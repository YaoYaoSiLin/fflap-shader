#version 130

#extension GL_EXT_gpu_shader4 : require
#extension GL_EXT_gpu_shader5 : require

#define SHADOW_MAP_BIAS 0.9

#define Enabled_TAA

#define SpecularityReflectionPower 2.0            //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]
#define Continuum2_Texture_Format

const int noiseTextureResolution = 64;

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform float viewWidth;
uniform float viewHeight;
uniform float rainStrength;

uniform int frameCounter;
uniform int isEyeInWater;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 upPosition;
uniform vec3 shadowLightPosition;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

in vec2 texcoord;

in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "libs/common.inc"
#include "libs/dither.glsl"
#include "libs/jittering.glsl"

#define Stage ScreenReflection
#define CalculateHightLight 1
#define depthOpaque depthtex0
#define reflectionSampler gaux2
//#define skyReflectionSampler gaux3

#include "libs/brdf.glsl"
#include "libs/atmospheric.glsl"
#include "libs/specular.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

void main() {
  vec4 albedo = texture2D(gcolor, texcoord);

  float id = round(texture2D(gdepth, texcoord).z * 255.0);
  //bool isSky = texture2D(gdepth, texcoord).z > 0.999;
  bool isSky = texture2D(gaux2, texcoord).a < 0.001;
  bool isTranslucentBlocks = albedo.a > 0.9;
  bool isPlants = id == 18.0 || id == 31.0;

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);

  float smoothness = texture2D(composite, texcoord).r;
  float metallic   = texture2D(composite, texcoord).g;
  float emissive   = texture2D(composite, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;
  //float IOR = 1.000293;
  //if(isTranslucentBlocks){
  //  float r = 1.0 + emissive;
  //  if(isEyeInWater > 0) IOR = r / IOR;
  //  else IOR = IOR / r;
  //}

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec3 color = texture2D(gaux2, texcoord).rgb;

  float depth = texture2D(depthtex0, texcoord).x;
  float depthSolid = texture2D(depthtex1, texcoord).x;

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0);
       vP /= vP.w;
  vec4 wP = gbufferModelViewInverse * vP;

  vec2 uvJittering = texcoord + haltonSequence_2n3[int(mod(frameCounter, 16))] * pixel;
  vec4 vPJittering = gbufferProjectionInverse * nvec4(vec3(uvJittering, texture2D(depthtex0, uvJittering)) * 2.0 - 1.0); vPJittering /= vPJittering.w;
  vec4 wPJittering = gbufferModelViewInverse * vPJittering;

  #ifdef Enabled_TAA
  vP = vPJittering;
  #endif
  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(nvP, normal);
  vec3 nrP = normalize(rP);
  //vec3 refractP = refract(nvP, normal, IOR);

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  //if(length(vP.xyz) < 100.0 || metallic > 0.5 || isTranslucentBlocks){
  if(!isSky){
    vec3 h = normalize(nrP - nvP);

    float ndoth = clamp01(dot(normal, h));

    float vdoth = 1.0 - clamp01(dot(-nvP, h));

    vec3 f = F(F0, pow5(vdoth));

    float ndotl = clamp01(dot(nrP, normal));
    float ndotv = 1.0 - clamp01(dot(-nvP, normal));

    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);
    float specularity = pow((1.0 - g) * clamp01(d), SpecularityReflectionPower);

    float dither = bayer_32x32(texcoord, resolution);
    dither *= 1.0;

    float r = clamp01(max(f.b, max(f.r, f.g)) * d);
          r = 1.0 - clamp01(r / (1.0 + 4.0 * abs(dot(normal, -nvP)) * abs(dot(normal, nrP))) * 5.0);

    vec3 skySpecularReflection = L2rgb(CalculateSky(nrP, sP, cameraPosition.y, 0.5));
         //skySpecularReflection = L2rgb(mix(skySpecularReflection, skyLightingColorRaw + sunLightingColorRaw * 0.1, r));

    vec4 ssR = vec4(0.0);
    //if((!isPlants && length(vP.xyz) < 100.0) || isTranslucentBlocks){
      ssR = raytrace(vP.xyz, rP, normal, dither * 0.01, r);
      //ssR.rgb = rgb2L(ssR.rgb);
    //  if(!isTranslucentBlocks) ssR.a *= clamp01((-length(vP.xyz) + 100.0 - 16.0) / 32.0);
    //}

    vec3 specularReflection = mix(skySpecularReflection, ssR.rgb, ssR.a);

    //color = vec3(0.0);
    //color = mix(color, specularReflection, f * specularity);
    color += specularReflection * f * specularity;
    //color = ssR.rgb;
    //color = vec3(clamp01(-dot((nvP), normal)));
  }

  //color = L2rgb(color);

/* DRAWBUFFERS:56 */
  gl_FragData[0] = vec4(color, 1.0);
  gl_FragData[1] = vec4(color, 1.0);
  //gl_FragData[1] = solidBlockSpecularReflection;
}

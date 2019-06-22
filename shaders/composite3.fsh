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
uniform sampler2D gaux1;
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

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

const bool gaux2MipmapEnabled = true;

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

  float skyLightMap = step(0.9, pow2(texture2D(gdepth, texcoord).y) * 4.0 - 0.01);

  int id = int(round(texture2D(gdepth, texcoord).z * 255.0));
  //bool isSky = texture2D(gdepth, texcoord).z > 0.999;
  bool isSky = id == 255;
  bool isPlants = id == 18 || id == 31 || id == 83;
  bool isTranslucentBlocks = albedo.a > 0.9;
  bool isWater = id == 8;
  bool isParticles = id == 253;

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);

  float smoothness = texture2D(composite, texcoord).r;
  float metallic   = texture2D(composite, texcoord).g;
  //float emissive   = texture2D(composite, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;

  float IOR = 1.0 + texture2D(composite, texcoord).b;
        IOR += step(0.5, metallic) * 49.0 + 2.5 * float(!isTranslucentBlocks);

  float ri = 1.000293;
  float ro = IOR;
  if(isEyeInWater == 1){ ri = 1.333; ro = 1.000293; }

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
  if(!isSky && smoothness > 0.015){
    vec3 h = normalize(nrP - nvP);

    float ndoth = clamp01(dot(normal, h));

    float vdoth = pow5(1.0 - clamp01(dot(-nvP, h)));

    vec3 f = F(F0, vdoth);

    float ndotl = clamp01(dot(nrP, normal));
    float ndotv = 1.0 - clamp01(dot(-nvP, normal));

    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);
    float specularity = pow((1.0 - g) * clamp01(d), SpecularityReflectionPower);

    float dither = bayer_32x32(texcoord, resolution);
    dither *= 1.0;

    #if Metal_Block_Reflection_Smoothness == 0
    //float ra = max(f.r, max(f.g, f.b)) * d * g;
    //      ra = 1.0 / (max(0.0, ra) / (4.0 * clamp01(dot(-nvP, normal))) * clamp01(dot(nrP, normal)));
    #else
    //float ra = 1.0 - ((min(f.b, min(f.r, f.g)) * clamp01(d * g) / (4.0 * clamp01(dot(-nvP, normal))) * clamp01(dot(nrP, normal)) + 1.0 )* 5.0);
    //float ra = 1.0 - ((clamp01(max(f.b, max(f.r, f.g)) * d * g) / (4.0 * clamp01(dot(-nvP, normal))) * clamp01(dot(nrP, normal)) + 1.0) * 5.0);
    #endif

    float s = (pow2(ro) * (1.0 - vdoth) * g * d) / pow2((ri) * clamp01(pow2(dot(-nvP, h)) + (ro) * clamp01(dot(nrP, h))));
    float sScreen = clamp01(1.0 / (s + 0.001) * 0.05);
          s = clamp01(1.0 / (s + 0.001));

    //vec3 noColor = color;

    float fMax = max(f.r, max(f.g, f.b));

    vec3 skySpecularReflection = CalculateSky(nrP, sP, cameraPosition.y, 0.5);
         skySpecularReflection = CalculateAtmosphericScattering(skySpecularReflection, -(mat3(gbufferModelViewInverse) * nrP).y + 0.15);
         skySpecularReflection = L2rgb(skySpecularReflection);
         skySpecularReflection = mix(skySpecularReflection, color, max(1.0 - skyLightMap, s));

    vec4 ssR = vec4(0.0);
    if((!isParticles && !isPlants && length(vP.xyz) < 64.0 && smoothness > 0.21) || isWater){
      ssR = raytrace(vP.xyz, rP, normal, dither * 0.01, sScreen);
      ssR.rgb = mix(ssR.rgb, skySpecularReflection, s);
    }

    //skySpecularReflection *= clamp01(pow2(skyLightMap) * 4.0 - 0.01);
    //skySpecularReflection *= 1.0 - g;
    //ssR.a = mix(1.0, ssR.a, clamp01(pow2(skyLightMap) * 4.0 - 0.01));

    vec3 specularReflection = mix(skySpecularReflection, ssR.rgb, max(ssR.a, 1.0 - skyLightMap));
         //specularReflection = mix(mix(color, skySpecularReflection, clamp01(g * d) * skyLightMap), specularReflection, clamp01(g * d));

    //color = mix(color, skySpecularReflection, clamp01(g * d) * skyLightMap);

    color += specularReflection * clamp01(pow3(g * d)) * f;

    //color = vec3(minComponent(albedo.rgb));

    //color = ssR.rgb;
  }

  //color = texture2D(composite, texcoord).rgb;

  //color = albedo.rgb * 255.0;

  //color = texture2D(composite, texcoord).rgb;

  //color = vec3(texture2D(gaux3, texcoord).r);
  //color = mix(color, texture2D(gaux2, texcoord).rgb, texture2D(gaux3, texcoord).r);

  color = mix(color, texture2D(gaux2, texcoord).rgb, texture2D(gaux1, texcoord).a);

  //color = vec3(maxComponent(albedo.rgb) - minComponent(albedo.rgb));

  //float cc = maxComponent(albedo.rgb) - minComponent(albedo.rgb);

  //color = vec3(clamp01(maxComponent(albedo.rgb - vec3(0.858, 0.827, 0.623) * 0.5)));

  //color.rgb = vec3(min(color.r, min(color.g, color.b)));

  //color = vec3(clamp01(texture2D(gaux3, texcoord).x * 1024.0 - length(vP.xyz))) * 0.1;
  //color = vec3(length(nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture2D(gaux3, texcoord).x) * 2.0 - 1.0)))) * 0.1;
  //if(texture)

  //color = albedo.rgb;

  //color = texture2D(gaux3, texcoord).rgb;

  //color = vec3(smoothness, metallic, 0.0);

  //color = L2rgb(color);

/* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color, 0.0);
  //gl_FragData[1] = solidBlockSpecularReflection;
}

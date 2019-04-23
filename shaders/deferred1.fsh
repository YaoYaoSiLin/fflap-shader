#version 130

#extension GL_EXT_gpu_shader4 : require
#extension GL_EXT_gpu_shader5 : require

#define SHADOW_MAP_BIAS 0.9

#define SpecularityReflectionPower 2.0            //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]

//#define Enabled_SSAO
    #define SSAO_Scale 0.5  //[0.5 0.70710677 1.0]

const int   noiseTextureResolution  = 64;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex6;

uniform sampler2D depthtex0;

uniform sampler2D noisetex;

uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform float rainStrength;
uniform float viewWidth;
uniform float viewHeight;

uniform vec3 sunPosition;
uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

uniform int isEyeInWater;
uniform int heldBlockLightValue;
uniform int heldBlockLightValue2;
uniform int frameCounter;

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

const bool shadowcolor0Mipmap = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / vec2(viewWidth, viewHeight);

#include "libs/common.inc"
#include "libs/dither.glsl"
#include "libs/jittering.glsl"

#define CalculateHightLight 1
#define CalculateShadingColor 1

#include "libs/brdf.glsl"
#include "libs/light.glsl"
#include "libs/atmospheric.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

float LinearlizeDepth(float depth) {
    return (far * (depth - near)) / (depth * (far - near));
}

float noise(vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = smoothstep(0.0, 1.0, f);

    vec2 uv = (p.xy+vec2(37.0, 17.0)*p.z) + f.xy;
    float v1 = texture2D( noisetex, (uv) / noiseTextureResolution, -100.0 ).x;
    float v2 = texture2D( noisetex, (uv + vec2(37.0, 17.0)) / noiseTextureResolution, -100.0 ).x;
    return mix(v1, v2, f.z);
}

vec4 GetViewPosition(in vec2 coord, in sampler2D depth){
  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, texture2D(depth, coord).x) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

vec4 GetViewPosition(in vec2 coord, in float depth){
  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

/*
vec3 CalculateRays(in vec4 wP, in bool isSky){
  vec3 light = vec3(0.0);
  int steps = 8;

  float dither = bayer_32x32(texcoord + haltonSequence_2n3[int(mod(frameCounter, 8))], resolution);

  if(!isSky){
    vec4 rP = wP;

    //float dither = bayer_16x16(texcoord, vec2(viewWidth, viewHeight));

    vec3 nwP = rP.xyz - vec3(0.0, playerEyeLevel, 0.0);
         nwP = normalize(nwP) * length(nwP) / float(steps) * 0.62;

    float vlCount = 0.0;

    for(int i = 0; i < steps; i++){
      rP.xyz -= nwP.xyz;
      vec4 rP2 = rP + vec4(nwP, 0.0) * fract(dither);

      float l = length(rP2.xyz);

      rP2.xyz = wP2sP(rP2);
      float shadow = float(texture2D(shadowtex1, rP2.xy).z + 0.0001 > rP2.z);

      light += vec3(shadow) * clamp01(l / 128.0 + 0.03);//mix(vec3(shadow), shadowColor.rgb, shadowAlp * shadow);
      //vlCount += 1.0;
    }

    //if(vlCount > 0.0) light /= vlCount;
    light /= 8.0;

    //light = min(light, vec3(1.0));
  }else{
    light = vec3(1.0);
  }

  return light;
}
*/
void main() {
  vec4 albedo = texture2D(colortex0, texcoord);
       albedo.rgb = pow(albedo.rgb, vec3(2.2));

  vec2 lightMap = texture2D(colortex1, texcoord).xy;
  float torchLightMap = clamp01(pow(1.0 / (lightMap.x / (lightMap.x + 0.012)) * lightMap.x, 2.0) * 1.003 - 0.003);
  float skyLightMap = max(pow(lightMap.y, 1.5) * 1.001 - 0.001, 0.0);

  bool isSky = int(texture2D(colortex1, texcoord).z * 255.0) == 255;

  vec3 normal = normalDecode(texture2D(colortex2, texcoord).xy);

  float smoothness = texture2D(colortex3, texcoord).r;
  float metallic   = texture2D(colortex3, texcoord).g;
  float emissive   = texture2D(colortex3, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float depth = texture2D(depthtex0, texcoord).x;

  vec3 sP = normalize(mat3(gbufferModelViewInverse) * sunPosition);

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0); vP /= vP.w;
  vec4 wP = gbufferModelViewInverse * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(normalize(vP.xyz), normal);
  vec3 nrP = normalize(rP.xyz);

  vec2 uvJittering = texcoord + haltonSequence_2n3[int(mod(frameCounter, 8))] * pixel;
  vec4 vPJittering = GetViewPosition(uvJittering, depthtex0);
  vec4 wPJittering = (gbufferModelViewInverse) * vPJittering;

  //float fading = texture2DLod(colortex6, pixel * vec2(2.0, 2.0), 0).r;
  //vec3 sunLightingColorRaw = texture2DLod(colortex6, pixel * vec2(6.0, 2.0), 0).rgb;
  //vec3 skyLightingColorRaw = texture2DLod(colortex6, pixel * vec2(10.0, 2.0), 0).rgb;
  vec3 torchLightingColor = vec3(1.022, 0.782, 0.344) * 0.33;

  vec3 color = vec3(0.0);

  vec3 shading = vec3(1.0);
  float ao = 1.0;

  vec4 skySpecularReflection = vec4(0.0);

  if(isSky){
    albedo.a = (0.0);
    //color = albedo.rgb;
  }else{
    //vec3 sunLightingColor = CalculateFogLighting(sunLightingColorRaw, playerEyeLevel + cameraPosition.y, wP.y + cameraPosition.y, sunLightingColorRaw, skyLightingColorRaw, fading);
    //vec3 skyLightingColor = CalculateFogLighting(skyLightingColorRaw, playerEyeLevel + cameraPosition.y, wP.y + cameraPosition.y, sunLightingColorRaw, skyLightingColorRaw, fading);

    #ifdef Enabled_SSAO
    vec4 aoSample = textureGather(colortex5, texcoord * SSAO_Scale);
    ao = pow(dot(aoSample, vec4(0.25)), 2.0);
    #endif

    vec4 sunDirctLighting = CalculateShading(shadowtex1, wP);
    shading = mix(shading, sunDirctLighting.rgb, sunDirctLighting.a) * texture2D(colortex2, texcoord).z;

    //vec3 sunLighting = albedo.rgb * sunLightingColorRaw * pow(fading, 5.0) * shading * clamp01(dot(normalize(shadowLightPosition), normal));

    //vec3 skyLighting = albedo.rgb * (clamp01(dot(normalize(upPosition), normal) * 0.5 + 0.5) * 0.85 + 0.15) * ao;
    vec3 skyLighting  = albedo.rgb * clamp01(dot(normalize(upPosition), normal));
         skyLighting += albedo.rgb * abs(dot(normalize(reflect(mat3(gbufferModelView) * vec3(1.0, 0.0, 0.0), normalize(upPosition))), normal)) * 0.48 * ao;
         skyLighting += albedo.rgb * abs(dot(normalize(reflect(mat3(gbufferModelView) * vec3(0.0, 0.0, 1.0), normalize(upPosition))), normal)) * 0.48 * ao;
         skyLighting += albedo.rgb * abs(dot(normalize(reflect(mat3(gbufferModelView) * vec3(0.0, 1.0, 0.0), normalize(upPosition))), normal)) * 0.21 * ao;

    vec3 fakeGI = albedo.rgb * clamp01(dot(normalize(reflect(normalize(shadowLightPosition), normalize(upPosition))), normal)) * 0.311
                + albedo.rgb * clamp01(dot(reflect(normalize(shadowLightPosition), mat3(gbufferModelView) * vec3(0.0, 0.0, -1.0)), normal)) * 0.216
                + albedo.rgb * clamp01(dot(-normalize(shadowLightPosition), normal)) * 0.216;
         fakeGI *= (skyLightingColorRaw + sunLightingColorRaw * pow(fading * clamp01(dot(sP, vec3(0.0, 1.0, 0.0))), 3.0)) * skyLightMap * skyLightMap * ao;

    color += skyLighting * skyLightMap * skyLightingColorRaw;
    color += fakeGI;

    vec3 h = normalize(nrP - nvP);

    float ndotv = 1.0 - clamp01(dot(-nvP, normal));
    float vdoth = 1.0 - clamp01(dot(-nvP, h));
    float ndoth = clamp01(dot(normal, h));
    float ndotl = clamp01(dot(nrP, normal));

    vec3  f = F(F0, pow5(vdoth));
    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);
    float specularity = pow(1.0 - clamp01(g), SpecularityReflectionPower);

    color *= rgb2L(vec3(1.0 - metallic));
    color *= rgb2L(clamp01(1.0 - f * specularity));

    skySpecularReflection.rgb = L2rgb(CalculateSky(nrP, sP, cameraPosition.y, 0.5)) * f * specularity;

    color += BRDF(albedo.rgb, normalize(shadowLightPosition), -nvP, normal, roughness, metallic, F0) * shading * sunLightingColorRaw * pow(fading, 5.0) * 4.0;
    //color = sunDirctLighting.rgb;
  }

  vec3 AtmosphericScattering = (CalculateSky(nvP, sP, cameraPosition.y, 0.5));
  color.rgb = mix(AtmosphericScattering, color.rgb, clamp01(albedo.a));
  //color.rgb = albedo.aaa;

  color = L2rgb(color);

/* DRAWBUFFERS:0156 */
  gl_FragData[0] = vec4(L2rgb(albedo.rgb), 0.0);
  gl_FragData[1] = vec4(torchLightMap, skyLightMap, texture2D(colortex1, texcoord).za);
  gl_FragData[2] = vec4(color, 1.0);
  gl_FragData[3] = vec4(0.0);
}


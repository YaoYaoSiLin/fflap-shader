#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;
uniform sampler2D gaux1;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

uniform float viewWidth;
uniform float viewHeight;

uniform int isEyeInWater;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define CalculateHightLight 0

#include "../libs/common.inc"
#include "../libs/atmospheric.glsl"
#include "../libs/brdf.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 RemovalColor(in vec3 image, in vec3 color, in float t){
  return (image - color) / (1.0 - t) + color;
}

vec3 GetScatteringCoe(in vec4 albedo){
  return pow2(albedo.a) * vec3(Pi);
}

vec3 GetAbsorptionCoe(in vec4 albedo){
  return (1.0 - albedo.rgb) * pow3(albedo.a) * Pi;
}

void main(){
  vec4 albedo = texture2D(gcolor, texcoord);

  vec2 unpackLightmap = unpack2x8(texture(gdepth, texcoord).x);
  float skyLightMap = unpackLightmap.y;
  float emissive   = texture2D(gdepth, texcoord).z;

  vec3 normal = normalDecode(texture2D(composite, texcoord).xy);
  //vec3 visibleNormal = normalDecode(texture2D(composite, texcoord).xy);
  float alpha = texture2D(gnormal, texcoord).x;

  vec2 unpackSpecular = unpack2x8(texture(composite, texcoord).b);

  float smoothness = unpackSpecular.x;
  float metallic   = unpackSpecular.y;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;

  vec3 F0 = vec3(metallic);
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float mask = round(texture2D(gnormal, texcoord).z * 255.0);
  bool isWater      = CalculateMaskID(8.0, mask);
  bool isIce        = CalculateMaskID(79.0, mask);
  bool isParticels = bool(step(249.5, mask) * step(mask, 250.5));
  bool emissiveParticels = bool(step(250.5, mask) * step(mask, 252.5));

  bool isGlass      = CalculateMaskID(20.0, mask);
  bool isGlassPane = CalculateMaskID(106.0, mask);
  bool isStainedGlass = CalculateMaskID(95.0, mask);
  bool isStainedGlassPane = CalculateMaskID(160.0, mask);
  bool AnyGlass = isGlass || isGlassPane || isStainedGlass || isStainedGlassPane;
  bool AnyGlassBlock = isGlass || isStainedGlass;

  vec4 color = texture2D(gaux2, texcoord);
       color.rgb = decodeGamma(color.rgb) * decodeHDR;

  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
  vec3 svP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));
  vec3 wP = mat3(gbufferModelViewInverse) * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 L = normalize(reflect(nvP, normal));

  float translucentBlockLength = length(svP - vP);

  vec3 worldSunPosition = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  albedo.rgb = decodeGamma(albedo.rgb);

  float skyVisiblity = max(0.0, skyLightMap - 0.07) / 0.93;

  if(bool(albedo.a) && !isParticels && !emissiveParticels){
    vec3 fogColor = albedo.rgb * skyLightingColorRaw;

    vec3 f = vec3(0.0);
    float g = 0.0, d = 0.0;
    FDG(f, g, d, L, -nvP, normal, L2Gamma(F0), roughness);
    float brdf = saturate(g * d) * 0.95;

    vec3 p = mat3(gbufferModelViewInverse) * L;
    vec3 eyePosition = vec3(0.0, cameraPosition.y - 63.0, 0.0);
    vec3 skySpecularReflection = CalculateInScattering(eyePosition, p, worldSunPosition, 0.76, ivec2(16, 2), vec3(1.0, 1.0, 0.0));
         skySpecularReflection = ApplyEarthSurface(skySpecularReflection, eyePosition, p, worldSunPosition);
         skySpecularReflection = (skySpecularReflection * pow3(skyVisiblity));

    //fogColor += skySpecularReflection * sqrt(f * brdf);

    //fogColor = (fogColor);

    color.rgb = RemovalColor(color.rgb, fogColor, color.a);
    //if(alpha > 0.99) image.rgb = vec3(0.0);
    //image.rgb *= 1.0 - max(0.0, image.a * 0.7071);
    //image.rgb = fogColor;
  }

  //vec3 backPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(gaux1, texcoord).x) * 2.0 - 1.0));
  //color.rgb = abs(length(vP) - texture(gaux1, texcoord).x * 544.0) * vec3(1.0 / 100.0);

  color.rgb = encodeGamma(color.rgb * encodeHDR);

  //color.rgb /= overRange;

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = color;
}

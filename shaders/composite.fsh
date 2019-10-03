#version 130

#define SpecularityReflectionPower 2.0            //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

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

#define Void_Sky

#include "libs/common.inc"
#include "libs/atmospheric.glsl"
#include "libs/brdf.glsl"

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

void main(){
  vec4 albedo = texture2D(gcolor, texcoord);
  bool isTranslucentBlocks = albedo.a > 0.99;

  bool isWater = int(round(texture2D(gdepth, texcoord).z * 255.0)) == 8;

  float skyLightMap = texture2D(gdepth, texcoord).y;

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);
  float alpha = texture2D(gnormal, texcoord).z;

  float smoothness = texture2D(composite, texcoord).r;
  float metallic   = texture2D(composite, texcoord).g;
  float emissive   = texture2D(composite, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;
  float r = texture2D(composite, texcoord).b;
  float IOR = 1.000293;
  if(isTranslucentBlocks){
    r = 1.0 + r;
    if(isEyeInWater > 0) IOR = r / IOR;
    else IOR = IOR / r;
  }

  vec4 color = texture2D(gaux2, texcoord);

  float depth = texture2D(depthtex0, texcoord).x;

  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0));
  vec3 wP = mat3(gbufferModelViewInverse) * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(nvP, normal);
  vec3 refractP = refract(nvP, normal, IOR);

  vec3 svP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  if(albedo.a > 0.99 && color.a > 0.01){
    //vec2 rcoord = nvec3(gbufferProjection * nvec4(vP.xyz + refractP)).xy * 0.5 + 0.5;

    //color = texture2D(gaux2, rcoord);

    vec3 F0 = vec3(max(0.02, metallic));
         F0 = mix(F0, albedo.rgb, step(0.5, metallic));

    vec3 h = normalize(rP - nvP);

    float ndoth = clamp01(dot(normal, h));
    float vdoth = 1.0 - clamp01(dot(-nvP, h));

    vec3 f = (F(F0, pow5(vdoth)));

    float ndotl = clamp01(dot(rP, normal));
    float ndotv = 1.0 - clamp01(dot(-nvP, normal));

    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);
    float specularity = pow(1.0 - g, SpecularityReflectionPower);

    specularity *= 1.0 - isEyeInWater * float(isWater) * step(dot(normal, normalize(upPosition)), 0.0);

    vec3 transColor = rgb2L(albedo.rgb) * skyLightingColorRaw;
         transColor = L2rgb(transColor) * skyLightMap;

    vec3 skySpecularReflection = CalculateSky(normalize(rP), sP, cameraPosition.y, 0.5);
         //skySpecularReflection = CalculateAtmosphericScattering(skySpecularReflection, -(mat3(gbufferModelViewInverse) * normalize(rP)).y + 0.15);

    vec3 transColorSpec = mix(transColor, L2rgb(skySpecularReflection), f * specularity * step(0.7, skyLightMap)) / overRange;

    color.rgb = RemovalColor(color.rgb, transColorSpec, color.a);
    //color.rgb += (transColorSpec - color.rgb) * pow5(color.a) * 0.3;
    //color.rgb = saturation(color.rgb, 1.0 - color.a * color.a * 0.9 * color.a);
    //color.rgb *= 1.0 - clamp01((color.a * 255.0 - 220.0) / 35.0);
    //color.rgb *= 1.0 - clamp01((color.a * 255.0 - 230.0) / 25.0);
  }

  //float t = sqrt(texcoord.x);

  //color.rgb = mix(color.rgb, vec3(0.5) * 0.25, t);
  //color.rgb = color.rgb - vec3(0.5) * 0.25;
  //color.rgb = (color.rgb - vec3(0.5) * 0.25) / (1.0 - t) + vec3(0.5) * 0.25;
  //color.rgb = (min(vec3(1.0), color.rgb * 4.0) - vec3(0.15)) / (0.1) + vec3(0.025);
  //color.rgb = min(vec3(1.0), color.rgb * overRange);

  //float lightDirect = clamp01(pow5(dot(normalize(svP), normalize(shadowLightPosition + svP - upPosition))));
  //color.rgb = vec3(pow5(lightDirect) / overRange) * 0.5;
  //CalculateFog(color.rgb, length(svP.xyz), svP.y + cameraPosition.y + playerEyeLevel, lightDirect);

  albedo.rgb = L2rgb(albedo.rgb);

/* DRAWBUFFERS:5 */
  gl_FragData[0] = color;
}

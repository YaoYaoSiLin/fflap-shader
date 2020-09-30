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

void main(){
  vec4 albedo = texture2D(gcolor, texcoord);

  float skyLightMap = texture2D(gdepth, texcoord).y;

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);
  float alpha = texture2D(composite, texcoord).x;

  float smoothness = texture2D(gnormal, texcoord).b;
  float metallic   = texture2D(composite, texcoord).b;
  float emissive   = texture2D(gdepth, texcoord).z;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;
  vec3 F0 = vec3(metallic);
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float mask = round(texture2D(composite, texcoord).z * 255.0);
  bool isWater      = CalculateMaskID(8.0, mask);
  bool isIce        = CalculateMaskID(79.0, mask);
  bool isParticels = bool(step(249.5, mask) * step(mask, 250.5));
  bool emissiveParticels = bool(step(250.5, mask) * step(mask, 252.5));

  vec4 image = texture2D(gaux2, texcoord);
       image.rgb *= overRange;

  albedo.rgb = rgb2L(albedo.rgb);

  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
  vec3 svP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));
  vec3 wP = mat3(gbufferModelViewInverse) * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 reflectionVector = normalize(reflect(nvP, normal));

  float translucentBlockLength = length(svP - vP);

  vec3 worldSunPosition = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  if(bool(albedo.a) && !isParticels && !emissiveParticels){
    vec3 color = albedo.rgb * skyLightingColorRaw;

    color = L2rgb(color);

    vec3 m = normalize(reflectionVector - nvP);
    float vdoth = pow5(1.0 - clamp01(dot(-nvP, m)));
    vec3 f = F(F0, vdoth);
    float brdf = min(1.0, CalculateBRDF(-nvP, reflectionVector, normal, roughness));

    vec3 skySpecularReflection = CalculateSky(reflectionVector, worldSunPosition, 0.0, 1.0);
    color *= 1.0 - brdf * max(f, vec3(step(0.5, metallic)));
    color += skySpecularReflection * sqrt(brdf * f);

    image.rgb = RemovalColor(image.rgb, color, image.a);
    if(image.a > 0.99) image.rgb = vec3(0.0);
    image.rgb *= 1.0 - max(0.0, image.a * 0.7071);
  }

  image.rgb /= overRange;


/* DRAWBUFFERS:5 */
  gl_FragData[0] = image;
}

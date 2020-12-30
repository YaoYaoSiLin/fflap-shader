#version 130

#define GetTranslucentBlocksDepth

//#define Continuum2_Texture_Format

#define tileResolution 16      //[0 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192]

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D gaux1;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform float viewWidth;
uniform float viewHeight;
uniform int isEyeInWater;

uniform vec3 upPosition;
uniform vec3 sunPosition;
uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;

uniform ivec2 atlasSize;

in float id;

in vec2 texcoord;
in vec2 lmcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

in vec3 tangent;
in vec3 binormal;
in vec3 normal;
in vec3 vP;

in vec4 biomesColor;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

vec2 screenCoord = gl_FragCoord.xy * pixel;

#include "../libs/common.inc"

#define CalculateHightLight 0
#define CalculateShadingColor 2

//#define Void_Sky

#include "../libs/brdf.glsl"
#include "../libs/atmospheric.glsl"
#include "../libs/water.glsl"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

vec3 GetScatteringCoe(in vec4 albedo){
  return pow2(albedo.a) * vec3(Pi);
}

vec3 GetAbsorptionCoe(in vec4 albedo){
  return (1.0 - albedo.rgb) * pow3(albedo.a) * Pi;
}

/* DRAWBUFFERS:012345 */

void main() {
  //float lastLayerDepth = texture2D()

  //if(texture2D(gaux1, screenCoord).x < gl_FragCoord.z + 0.00001 && texture2D(gaux1, screenCoord).x > 0.0) discard;
  //if(gl_FragCoord.z > texture2D(depthtex0, screenCoord).x) discard;
  //discard;

  float viewLength = length(vP);
  vec3 nvP = normalize(vP);
  bool backFace = dot(normal, nvP) > 0.0;

  vec3 worldSunPosition = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  vec3 vPSolidBlock = vec3(screenCoord, texture2D(depthtex1, screenCoord).x) * 2.0 - 1.0;
       vPSolidBlock = nvec3(gbufferProjectionInverse * nvec4(vPSolidBlock));

  float mask = round(id);
  bool isWater      = CalculateMaskID(8.0, mask);
  bool isIce        = CalculateMaskID(79.0, mask);
  bool isEwww       = CalculateMaskID(165.0, mask);

  bool isGlass      = CalculateMaskID(20.0, mask);
  bool isGlassPane = CalculateMaskID(106.0, mask);
  bool isStainedGlass = CalculateMaskID(95.0, mask);
  bool isStainedGlassPane = CalculateMaskID(160.0, mask);
  bool AnyGlass = isGlass || isGlassPane || isStainedGlass || isStainedGlassPane;
  bool AnyClearGlass = isGlass || isGlassPane;
  bool AnyStainedGlass = isStainedGlass || isStainedGlassPane;
  bool AnyGlassBlock = isGlass || isStainedGlass;
  bool AnyGlassPane = isGlassPane || isStainedGlassPane;

  if(isWater && !gl_FrontFacing) discard;

  if(!gl_FrontFacing) {
    gl_FragData[4] = vec4(length(vP) / 544.0, normalEncode(normal), 1.0);
    return;
  }

  vec4 albedo = texture2D(texture, texcoord) * biomesColor;

  vec3 normalTexture = texture2D(normals, texcoord).rgb;
  if(normalTexture.x + normalTexture.y + normalTexture.z < 0.001) normalTexture = vec3(0.5, 0.5, 1.0);
  normalTexture = normalTexture * 2.0 - 1.0;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  float smoothness = clamp(speculars.r, 0.001, 0.999);
  float metallic = max(0.02, speculars.g);

  float blockDepth = 1.0;
  float r = 1.333;

  if(isWater){
    albedo = biomesColor;
    albedo = CalculateWaterColor(albedo);

    smoothness = 0.99;
    metallic = 0.02;
    blockDepth = 4.0;
    //blockDepth = min(255.0, length(vPSolidBlock - vP));
  }

  if(AnyGlass){
    //if(albedo.a < 0.001) albedo.rgb = vec3(0.04);
    r = 1.52;
    blockDepth = 0.125 + 0.875 * float(AnyGlassBlock);

    if(speculars.a - speculars.r < 0.0001)
    smoothness = 0.96 - 0.64 * max(0.0, albedo.a - 0.9) * 10.0;

    metallic = 0.0425;
  }

  //blockDepth = min(length(vPSolidBlock - vP), blockDepth);

  float roughness = 1.0 - smoothness;
        roughness = roughness * roughness;

  vec3 F0 = vec3(metallic);
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  mat3 tbn = mat3(tangent, binormal, normal);

  vec3 texturedNormal = normalize(tbn * normalTexture);
  vec3 flatNormal = normal * sign(float(backFace) - 0.5);
  vec3 visibleNormal = flatNormal;
  if(bool(step(dot(-nvP, visibleNormal), 0.2))) visibleNormal = flatNormal;

  float backFaceNormal = -float(gl_FrontFacing);

  texturedNormal *= backFaceNormal;
  visibleNormal *= backFaceNormal;
  flatNormal *= backFaceNormal;

  vec3 L = normalize(reflect(nvP, visibleNormal));
  vec3 rayDirection = mat3(gbufferModelViewInverse) * L;

  float scatteringcoe = GetScatteringCoe(albedo).x;

  float solidPart = step(0.9, albedo.a);
  if(AnyGlass) scatteringcoe *= mix(0.1, 1.0, solidPart);

  float alpha = 1.0 - exp(-scatteringcoe * blockDepth);
        alpha = pow(alpha, 2.2);
  //albedo.rgb = L2Gamma(albedo.rgb);

  float skyVisiblity = saturate(lmcoord.y - 0.07) / 0.93;

  vec3 color = L2Gamma(albedo.rgb) * (skyLightingColorRaw * pow3(skyVisiblity));

  //alpha = blockDepth * alpha * Pi;
  //alpha = 1.0 - min(1.0, exp(-alpha));
  //alpha = max(alpha, (albedo.a - 0.95) * 20.0);
  /*
  vec3 m = normalize(L - nvP);
  float vdoth = pow5(1.0 - clamp01(dot(-nvP, m)));
  vec3 f = F(L2Gamma(F0), vdoth);
  float brdf = min(1.0, CalculateBRDF(-nvP, L, visibleNormal, roughness));
  */

  roughness = 1.0 - roughness;

  vec3 f = vec3(0.0);
  float g = 0.0;
  float d = 0.0;
  FDG(f, g, d, L, -nvP, visibleNormal, L2Gamma(F0), roughness);
  float brdf = saturate(g * d) * 0.95;

  vec3 eyePosition = vec3(0.0, cameraPosition.y - 63.0, 0.0);
  vec3 skySpecularReflection = CalculateInScattering(eyePosition, rayDirection, worldSunPosition, 0.76, ivec2(16, 2), vec3(1.0, 1.0, 0.0));
       skySpecularReflection = ApplyEarthSurface(skySpecularReflection, eyePosition, rayDirection, worldSunPosition);
       skySpecularReflection = (skySpecularReflection * pow3(skyVisiblity));

  //color.rgb *= 1.0 - brdf * mix(f, vec3(1.0), step(0.5, metallic));

  //color.rgb += skySpecularReflection * sqrt(f * brdf);
  //alpha = max(alpha, maxComponent(sqrt(f * brdf)));

  if((isWater && isEyeInWater == 1) || (isGlass && albedo.a > 0.95)) alpha = 0.0;

  color = G2Linear(color);
  alpha = pow(alpha, 1.0 / 2.2);

  color /= overRange;

  //albedo.rgb = G2Linear(albedo.rgb);

  float selfShadow = 1.0;
  float emissive = speculars.b;
  vec4 lightmap = vec4(pack2x8(lmcoord), selfShadow, emissive, 1.0);

  float materials = mask / 255.0;

  vec2 encodeNormal = normalEncode(visibleNormal);

  float specularPackge = pack2x8(smoothness, metallic);

  gl_FragData[0] = vec4(albedo.rgb, 1.0);
  gl_FragData[1] = lightmap;
  gl_FragData[2] = vec4(albedo.a, 0.0, materials, 1.0);
  gl_FragData[3] = vec4(encodeNormal, specularPackge, 1.0);
  gl_FragData[5] = vec4(color, alpha);

    //if(gl_FrontFacing)
    //gl_FragData[5] = vec4(color, alpha);

    //gl_FragData[4] = vec4(gl_FragCoord.z, 1.0, 0.0, float(!gl_FrontFacing));
}

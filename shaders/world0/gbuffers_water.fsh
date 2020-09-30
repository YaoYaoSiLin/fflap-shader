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

void main() {
  if(texture2D(gaux1, screenCoord).a < gl_FragCoord.z && texture2D(gaux1, screenCoord).a > 0.0) discard;
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
    blockDepth = min(255.0, length(vPSolidBlock - vP));
  }

  if(AnyGlass){
    //if(albedo.a < 0.001) albedo.rgb = vec3(0.04);
    r = 1.52;
    blockDepth = 0.125 + 0.875 * float(AnyGlassBlock);

    if(speculars.a - speculars.r < 0.0001)
    smoothness = 0.96 - 0.64 * max(0.0, albedo.a - 0.9) * 10.0;
  }

  //blockDepth = min(length(vPSolidBlock - vP), blockDepth);

  float roughness = 1.0 - smoothness;
        roughness = roughness * roughness;

  vec3 F0 = vec3(metallic);
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  mat3 tbn = mat3(tangent, binormal, normal);
  vec3 normalSurface = normalize(tbn * normalTexture);
  if(backFace) normalSurface = -normalSurface;
  vec3 normalVisible = normalSurface - (normalSurface - normal) * step(-0.15, dot(nvP, normalSurface));

  vec3 reflectP = normalize(reflect(nvP, normalVisible));
  vec3 surfaceReflectionVector = normalize(reflect(nvP, normalSurface));

  float alpha = albedo.a;
  albedo.rgb = L2Gamma(albedo.rgb);

  vec3 color = albedo.rgb * skyLightingColorRaw;

  //alpha = blockDepth * alpha * Pi;
  //alpha = 1.0 - min(1.0, exp(-alpha));
  //alpha = max(alpha, (albedo.a - 0.95) * 20.0);

  color = L2rgb(color);

  vec3 m = normalize(reflectP - nvP);
  float vdoth = pow5(1.0 - clamp01(dot(-nvP, m)));
  vec3 f = F(F0, vdoth);
  float brdf = min(1.0, CalculateBRDF(-nvP, reflectP, normalVisible, roughness));

  vec3 skySpecularReflection = L2rgb(CalculateSky(surfaceReflectionVector, worldSunPosition, 0.0, 1.0));
  color.rgb *= 1.0 - brdf * mix(f, vec3(1.0), step(0.5, metallic));
  color.rgb += skySpecularReflection * sqrt(f * brdf);
  alpha = max(alpha, maxComponent(sqrt(brdf * f)));

  if((isWater && isEyeInWater == 1) || (!isWater && albedo.a > 0.95)) alpha = 0.0;

  color /= overRange;

  normalVisible.xy = normalEncode(normalVisible);

  albedo.rgb = G2Linear(albedo.rgb);

/* DRAWBUFFERS:01235 */
  gl_FragData[0] = vec4(albedo.rgb, 1.0);
  gl_FragData[1] = vec4(lmcoord, id / 255.0, 1.0);
  gl_FragData[2] = vec4(normalVisible.xy, smoothness, 1.0);
  gl_FragData[3] = vec4(albedo.a, 0.0, metallic, 1.0);
  gl_FragData[4] = vec4(color, alpha);
  //gl_FragData[1] = vec4(torchLightMap / 15.0, skyLightMap, 0.0, 1.0);
  //gl_FragData[2] = vec4(normalEncode(normalTexture), 1.0, 1.0);
  //gl_FragData[3] = vec4(smoothness, metallic, id / 65535.0, 1.0);
  //gl_FragData[4] = vec4(color / overRange, alpha);
  //gl_FragData[5] = vec4(color, 1.0 - alpha);
}

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

#include "libs/common.inc"

#define CalculateHightLight 0
#define CalculateShadingColor 2

#define Void_Sky

#include "libs/brdf.glsl"
#include "libs/atmospheric.glsl"
#include "libs/water.glsl"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

vec2 OffsetCoord(in vec2 coord, in vec2 offset)
{
  vec2 tileResolutionFix = (tileResolution) / vec2(atlasSize);

	vec2 offsetCoord = coord + mod(offset.xy, tileResolutionFix);

	vec2 minCoord = vec2(coord.x - mod(coord.x, tileResolutionFix.x), coord.y - mod(coord.y, tileResolutionFix.y));
	vec2 maxCoord = minCoord + tileResolutionFix;

	if (offsetCoord.x > maxCoord.x) {
		offsetCoord.x -= tileResolutionFix.x;
	} else if (offsetCoord.x < minCoord.x) {
		offsetCoord.x += tileResolutionFix.x;
	}

	if (offsetCoord.y > maxCoord.y) {
		offsetCoord.y -= tileResolutionFix.y;
	} else if (offsetCoord.y < minCoord.y) {
		offsetCoord.y += tileResolutionFix.y;
	}

	//offsetCoord /= atlasSize;

	return offsetCoord;
}

void main() {
  if(texture2D(gaux1, screenCoord).a < gl_FragCoord.z && texture2D(gaux1, screenCoord).a > 0.0) discard;
  //if(dot(normal, normalize(vP)) > 0.0) discard;

  mat3 tbn = mat3(tangent, binormal, normal);

  vec4 albedo = texture2D(texture, texcoord) * biomesColor;
  vec3 normalTexture = texture2D(normals, texcoord).rgb * 2.0 - 1.0;
  vec4 speculars = texture2D(specular, texcoord);

  //float blankTexture = min(albedo.a, speculars.a);

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  float smoothness = clamp(speculars.r, 0.001, 0.999);
  float metallic = speculars.g;

  float blockID = int(round(id));

  bool isWater      = blockID == 8;
  bool isGlass      = blockID == 20;
  bool isGlassPlane = blockID == 106;
  bool isIce        = blockID == 79;

  //if(dot(normal, normalize(vP)) > 0.0) discard;

  float r = 1.333;

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);
  vec3 rP = reflect(normalize(vP.xyz), normal);
  vec3 nrP = normalize(rP);
  vec3 nvP = normalize(vP.xyz);

  vec3 solidVector = nvec3(gbufferProjectionInverse * nvec4(vec3(screenCoord, texture2D(depthtex1, screenCoord).x) * 2.0 - 1.0));
  vec3 halfVector = vP - solidVector;

  float halfVectorLength = length(halfVector);

  float blockDepth = 1.0;

  if(isWater) {
    albedo = vec4(biomesColor.rgb, 0.05);
    albedo = CalculateWaterColor(albedo);

    //smoothness = 0.99 + (biomesColor.b - (biomesColor.r + biomesColor.g) * 0.5) / min(biomesColor.r, biomesColor.g);
    smoothness = 0.9996;
    metallic = 0.0;
    blockDepth = 255;
  }

  if(isGlass || isGlassPlane){
    //if(isGlassPlane) albedo.a = min(0.9, albedo.a);

    if(albedo.a < 0.001) albedo.rgb = vec3(0.0425);
    smoothness = mix(0.8, smoothness, step(0.001, speculars.a));
    r = 1.52;
    blockDepth = 0.125 + 0.875 * float(isGlass);
  }

  if(isIce){
    r = 1.31;
    smoothness = 0.9;
  }

  //if()

  //if(int(round(id)) == 90.0){
  //  albedo.a *= 0.6;
  //}

  float alpha = clamp01(albedo.a);

  float torchLightMap = max(0.0, pow5(lmcoord.x) * 15.0 - 0.005 + max(0.0, pow5(lmcoord.x) * 15.0 - 5.0));
  float skyLightMap = max(pow2(lmcoord.y) * 1.004 - 0.004, 0.0);

  if(texture2D(normals, texcoord).a + texture2D(texture, texcoord).a < 0.001) normalTexture = vec3(0.0, 0.0, 1.0);
  normalTexture.xy *= step(0.0, -dot(normalize(vP.xyz), normalize(tbn * normalTexture)));
  normalTexture = normalize(tbn * normalTexture);

  float roughness = 1.0 - smoothness;
        roughness = roughness * roughness;

  vec3 F0 =  vec3((1.000293 - r) / (1.000293 + r));
       F0 = F0 * F0;
       F0 = vec3(max(F0.x, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec3 h = normalize(nrP - nvP);

  float ndoth = clamp01(dot(normalTexture, h));
  float ndotl = clamp01(dot(nrP, normalTexture));
  float ndotv = 1.0 - clamp01(dot(-nvP, normalTexture));

  float vdoth = clamp01(pow5(1.0 - dot(normalTexture, -nvP)));

  float d = DistributionTerm(pow2(1.0 - smoothness), ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);
  vec3 f = F(F0, vdoth);

  float specularity = clamp01(g * d);

  float shading = pow(clamp01(dot(normalize(shadowLightPosition), normal)), 0.2);

  albedo.rgb = rgb2L(albedo.rgb);

  vec3 color = albedo.rgb * skyLightingColorRaw * skyLightMap;
       //color += albedo.rgb * skyLightingColorRaw * sunLightingColorRaw * fading * 5.0;
       //color= albedo.rgb * sunLightingColorRaw * fading * shading;

  albedo.rgb = L2rgb(albedo.rgb);
  color = L2rgb(color);

  //if(albedo.a < 0.2) color = vec3(0.02);

  //color = color * (shading) * sunLightingColorRaw * 1.7 + color * skyLightingColorRaw;
  //color *= 0.1;

  //if(alpha > 0.85) alpha = 1.0 - max(0.0, alpha * 1.16 - 0.16);
  //vec3 wP = mat3(gbufferModelViewInverse) *


  //if(isEyeInWater == 0) alpha = 1.0 - pow5(clamp01(exp(-min(blockDepth, hVdistance) * (alpha * alpha))));

  //alpha = min(0.8, alpha);
  //albedo.a = 1.0 - pow5(clamp01(exp(-min(1.0 + float(isWater) * far * 0.5, length(halfVector)) * (albedo.a * albedo.a))));

  //alpha = min(alpha, 0.85);
  //alpha *= 0.2;
  //color *= 0.7;
  //alpha = 0.5;

  //(1.0 + clamp01(d)) * (alpha * alpha) * (alpha + min(blockDepth, halfVectorLength)) / Pi * 1.0
  float scatteringFactor = alpha * min(blockDepth, halfVectorLength) * pow3(alpha + 1.0);
        scatteringFactor = 1.0 - clamp01(exp(-scatteringFactor));

        //scatteringFactor = max(scatteringFactor, vdoth);

  alpha = scatteringFactor;
  //if(isWater && isEyeInWater == 1) alpha = 1.0 - length(refract(nvP, normal, 1.333 / 1.000293));
  //alpha *= 1.0 - isEyeInWater * float(isWater);

  vec3 skySpecularReflection = CalculateSky(normalize(rP), sP, cameraPosition.y, 0.5);
       //skySpecularReflection = CalculateAtmosphericScattering(skySpecularReflection, -(mat3(gbufferModelViewInverse) * nrP).y + 0.15);

  specularity *= 1.0 - isEyeInWater * float(isWater) * step(dot(normal, normalize(upPosition)), 0.0);

  if(albedo.a * 255.0 > 240.0) {
    alpha = 0.0;
    //color = saturation(color, 2.0);
  }

  color = mix(color, L2rgb(skySpecularReflection), (f * specularity) * step(0.7, skyLightMap));
  alpha = max(alpha, specularity * dot03(f) * specularity);
  //alpha *= skyLightMap;

  alpha = min(0.9411, alpha);

  //if(alpha * 255.0 > 240.0) alpha = 0.9411;
  //color = saturation(color, 0.1);
  //color = min(color, vec3(0.9411));

  //stop draw particles on gbuffers_water
  //vec4 particlesColor  = texture2D(gaux1, screenCoord);
  //float particlesDepth = length(nvec3(gbufferProjectionInverse * nvec4(vec3(screenCoord, texture2D(gaux3, screenCoord).z) * 2.0 - 1.0)));

  //if(length(vP) - particlesDepth - 0.05 > 0.0 && particlesColor.a > 0.01){
  //  color = mix(color, particlesColor.rgb, particlesColor.a);
  //  alpha = max(alpha, particlesColor.a);
  //}

  //alpha = 1.0;

  float cut = 1.0;
/*
  if(dot(normal, normalize(vP)) > 0.0 && !isWater && length(vP) - length(solidVector) < 0.0) {
    cut = 0.0;
    alpha = 0.0;
    albedo.a = 0.0;
  }
*/
  //if(dot(normal, normalize(vP)) > 0.0 && texture2D(gaux3, screenCoord).x > 0.0) discard;

  //if(dot(normal, normalize(vP)) > 0.0 && texture2D(gaux3, screenCoord).a  > 0.99) discard;

  //if(length(vP.xyz) > texture2D(gaux3, screenCoord).a * 1024.0 - 0.05){ alpha = 0.0; cut = 0.0; albedo.a = 0.0;}
  //color = albedo.rgb * 0.5;

  //if(isWater) alpha = length(nvec3(gbufferProjectionInverse * nvec4(vec3(gl_FragCoord.xy * pixel, texture(depthtex1, gl_FragCoord.xy * pixel).x) * 2.0 - 1.0)));
  //if(isWater) alpha = 1.0 - clamp01(length(vP - nvec3(gbufferProjectionInverse * nvec4(vec3(gl_FragCoord.xy * pixel, texture2D(depthtex1, gl_FragCoord.xy * pixel).x) * 2.0 - 1.0))) * 0.2);
  //vec3 halfVector = vP - nvec3(gbufferProjectionInverse * nvec4(vec3(gl_FragCoord.xy * pixel, texture2D(depthtex1, gl_FragCoord.xy * pixel).x) * 2.0 - 1.0));
  //alpha = clamp01(1.0 - exp(-length(halfVector) * alpha)) * pow5(clamp01(dot(normalize(vP), normalize(vP + refract(normalize(vP), normalTexture, 1.000293 / r)))));

  //if(albedo.a > 0.9) alpha = 0.0;

  //if(length(vP) < 1.0) discard;

/* DRAWBUFFERS:01235 */
  gl_FragData[0] = vec4(albedo.rgb, 1.0);
  gl_FragData[1] = vec4(torchLightMap / 15.0, skyLightMap, id / 255.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalTexture), albedo.a, 1.0);
  gl_FragData[3] = vec4(smoothness, metallic, r - 1.0, 1.0);
  gl_FragData[4] = vec4(color / overRange, alpha);
  //gl_FragData[5] = vec4(color, 1.0 - alpha);
}

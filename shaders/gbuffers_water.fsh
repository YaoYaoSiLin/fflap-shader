#version 130

#define Continuum2_Texture_Format

#define SpecularityReflectionPower 2.0  //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]

#define tileResolution 128

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D gaux2;
uniform sampler2D gaux3;

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

#define CalculateHightLight 1
#define CalculateShadingColor 2

#include "libs/brdf.glsl"
#include "libs/atmospheric.glsl"

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
  //float depth = nvec3(gbufferProjection * nvec4(vP)).z * 0.5 + 0.5;
  //if(depth > 0.999 || depth < 0.05) discard;

  mat3 tbn = mat3(tangent, binormal, normal);

  vec4 albedo = texture2D(texture, texcoord) * biomesColor;
  vec3 normalMap = texture2D(normals, texcoord).rgb;
  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  float smoothness = speculars.r;
  float metallic = speculars.g;

  bool isWater = int(round(id)) == 8;
  bool isGlass = int(round(id)) == 20;
  bool isIce   = int(round(id)) == 79;

  float r = 1.333;

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);
  vec3 rP = reflect(normalize(vP.xyz), normal);
  vec3 nrP = normalize(rP);
  vec3 nvP = normalize(vP.xyz);

  vec3 solidVector = nvec3(gbufferProjectionInverse * nvec4(vec3(screenCoord, texture2D(depthtex1, screenCoord).x) * 2.0 - 1.0));
  vec3 halfVector = vP - solidVector;

  if(isWater) {
    albedo = vec4(vec3(0.02), 0.01);
    r = 1.333;
    albedo.rgb = mix(biomesColor.rgb, albedo.rgb, pow5(dot(biomesColor.rgb, vec3(1.0)) * 0.3333));
    //albedo.rgb += mix(biomesColor.rgb, albedo.rgb, abs(biomesColor.rgb - 1.0) * biomesColor.a);
    //albedo.a = 1.0 - (clamp01(1.0 + biomesColor.b - (biomesColor.r * 0.621 + biomesColor.g * 0.379) * 0.48));
    //albedo.a = max(albedo.a, pow(clamp01(max(biomesColor.r - biomesColor.g, biomesColor.g - biomesColor.r)), 1.0));
    //albedo.a = max(albedo.a, pow((max(0.0, albedo.r - albedo.g) + max(0.0, albedo.g - albedo.r)), 0.5) * clamp01((1.0 - albedo.b) * 5.0));
    albedo.a = max(albedo.a, (max(0.0, albedo.r - albedo.g) + max(0.0, albedo.g - albedo.r)) * max((1.0 - biomesColor.b), albedo.a) * 2.0);
    albedo.rgb += (vec3(64.0, 86.0, 26.0) / 255.0 * 0.06 + skyLightingColorRaw * 0.62 * pow5(getLum(biomesColor.rgb))) * 16.0 * biomesColor.rgb;
    albedo.a = pow(albedo.a, 0.33);
    //albedo.a = mix(0.06, albedo.a, abs(dot(normalize(upPosition), normal)));
    //albedo.a = clamp01(albedo.a * max(albedo.a, 1.0 - biomesColor.b));
    //albedo.a = max(albedo.a, min(1.0, (max(0.0, albedo.r - albedo.g) + max(0.0, albedo.g - albedo.r)) * 4.0));
    albedo.rgb *= 0.24;
    //albedo.a = 1.0;
    //smoothness = clamp01(1.0 - (albedo.a - 0.08) * 0.17) * 0.9976;
    smoothness = 0.9998 - pow5(albedo.a) * 1.0;
    metallic = 0.0;
  }

  if(isGlass){
    //albedo = vec4(1.0);
    if(albedo.a < 0.001) albedo.rgb = vec3(0.0425);
    //metallic = 0.0425;
    smoothness = mix(0.9, smoothness, step(0.001, max(speculars.a, albedo.a)));
    r = 1.52;
    //albedo.a = 1.0;
  }

  if(isIce){
    r = 1.31;
  }

  if(albedo.a > 0.95){
    //r = 1.000293;
  }

  float alpha = albedo.a;
  //if(alpha > 0.98 && !isWater) alpha = 1.0 - clamp01((alpha - 0.0199) * 50.0);
  //else
  if(isEyeInWater == 0) alpha = 1.0 - pow5(clamp01(exp(-min(1.0 + float(isWater) * far * 0.5, length(halfVector)) * (albedo.a * albedo.a))));

  if(normalMap.x + normalMap.y <= 0.0) normalMap = vec3(0.5, 0.5, 1.0);
  normalMap = normalMap * 2.0 - 1.0;
  normalMap = normalize(tbn * normalMap);

  float roughness = 1.0 - smoothness;
        roughness = roughness * roughness;

  vec3 F0 = vec3(0.02);
  if(r != 1.333) {
       F0 = vec3((1.000293 - r) / (1.000293 + r));
       F0 = F0 * F0;
  }
       F0 = vec3(max(F0.x, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec3 color = albedo.rgb;
  //if(isWater) color.rgb += pow5(getLum(biomesColor.rgb)) * vec3(64.0, 86.0, 26.0) / 255.0 * 0.16 + vec3(0.0, 0.0, skyLightingColorRaw.b) * 0.54;
       color = rgb2L(color);

  if(isWater){
    //color.rgb += ((rgb2L(vec3(74.0, 140.0, 64.0) / 255.0) * 0.2 + vec3(0.0, 0.0, skyLightingColorRaw.b)) * 4.0 * (sunLightingColorRaw * pow5(fading) * skyLightingColorRaw)) * alpha * pow5(dot(biomesColor.rgb, vec3(1.0)) * 0.3333);
  }

  float shading = pow(clamp01(dot(normalize(shadowLightPosition), normal)), 0.2);

  color = color * sunLightingColorRaw * fading * shading + color * skyLightingColorRaw;
  color = L2rgb(color);

  //if(albedo.a < 0.2) color = vec3(0.02);

  //color = color * (shading) * sunLightingColorRaw * 1.7 + color * skyLightingColorRaw;
  //color *= 0.1;

  //if(alpha > 0.85) alpha = 1.0 - max(0.0, alpha * 1.16 - 0.16);
  //vec3 wP = mat3(gbufferModelViewInverse) *

  vec3 h = normalize(rP - nvP);

  float vdoth = 1.0 - clamp01(dot(-nvP, h));

  vec3 f = F(F0, pow5(vdoth));

  float ndotl = clamp01(dot(rP, normal));
  float ndotv = 1.0 - clamp01(dot(-nvP, normal));

  float d = DistributionTerm(roughness, clamp01(dot(normalize(rP -nvP), normal)));
  float g = VisibilityTerm(d, ndotv, ndotl);
  float specularity = pow(1.0 - g, SpecularityReflectionPower);

  vec3 skySpecularReflection = L2rgb(CalculateSky(normalize(rP), sP, cameraPosition.y, 0.5));

  //alpha = min(0.8, alpha);
  //albedo.a = 1.0 - pow5(clamp01(exp(-min(1.0 + float(isWater) * far * 0.5, length(halfVector)) * (albedo.a * albedo.a))));

  //alpha = min(alpha, 0.85);
  //alpha *= 0.2;
  //color *= 0.7;
  //alpha = 0.5;

  color = mix(color, skySpecularReflection, (f * specularity));
  alpha = max(alpha, mix(alpha, specularity, dot(f, vec3(1.0)) / 3.0));

  if(albedo.a > 0.95) alpha = 0.0;

  //if(isWater) alpha = length(nvec3(gbufferProjectionInverse * nvec4(vec3(gl_FragCoord.xy * pixel, texture(depthtex1, gl_FragCoord.xy * pixel).x) * 2.0 - 1.0)));
  //if(isWater) alpha = 1.0 - clamp01(length(vP - nvec3(gbufferProjectionInverse * nvec4(vec3(gl_FragCoord.xy * pixel, texture2D(depthtex1, gl_FragCoord.xy * pixel).x) * 2.0 - 1.0))) * 0.2);
  //vec3 halfVector = vP - nvec3(gbufferProjectionInverse * nvec4(vec3(gl_FragCoord.xy * pixel, texture2D(depthtex1, gl_FragCoord.xy * pixel).x) * 2.0 - 1.0));
  //alpha = clamp01(1.0 - exp(-length(halfVector) * alpha)) * pow5(clamp01(dot(normalize(vP), normalize(vP + refract(normalize(vP), normalMap, 1.000293 / r)))));

  //if(albedo.a > 0.9) alpha = 0.0;

/* DRAWBUFFERS:01235 */
  gl_FragData[0] = vec4(albedo.rgb, 1.0);
  gl_FragData[1] = vec4(0.0, 0.0, id / 255.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalMap), albedo.a, 1.0);
  gl_FragData[3] = vec4(smoothness, metallic, r - 1.0, 1.0);
  gl_FragData[4] = vec4(color, alpha);
  //gl_FragData[5] = vec4(color, 1.0 - alpha);
}

#version 130

#define GI_Rendering_Scale 0.5 //[0.353553 0.5]

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;
const bool  shadowHardwareFiltering = false;

uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;

uniform sampler2D depthtex2;

uniform sampler2D shadowtex0;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

uniform vec3 shadowLightPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;

in vec2 texcoord;

float shadowPixel = 1.0 / float(shadowMapResolution);
vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 wP2sP(in vec4 wP, out float bias){
	vec4 sP = shadowModelView * wP;
       sP = shadowProjection * sP;
       sP /= sP.w;

  bias = 1.0 / (mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95);

  //sP.xy *= bias;
  sP.z /= max(far / shadowDistance, 1.0);
  sP = sP * 0.5 + 0.5;

	return sP.xyz;
}

vec3 GetClosest(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  //coord.xy = jittering * pixel;
  float depth = texture(depthtex0, coord).x;

  //if(depth > 0.9999) closest.xyz = vec3(-1.0, depth);


  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      vec2 neighborhood = (vec2(i, j)) * pixel * GI_Rendering_Scale;
      float neighbor = texture(depthtex0, coord + neighborhood).x;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;

  return closest;
}

float linearizeShadowMapDepth(float depth) {
    return (2.0 * near) / (shadowDistance + near - depth * (shadowDistance - near));
}

float invFar = 1.0 / far;
#define MaxRange 8.0

vec4 Gather(in sampler2D tex, in vec2 coord){
  vec4 sampler;

  sampler += texture2D(gaux2, coord + vec2(pixel.x, 0.0));
  sampler += texture2D(gaux2, coord - vec2(pixel.x, 0.0));
  sampler += texture2D(gaux2, coord + vec2(0.0, pixel.y));
  sampler += texture2D(gaux2, coord - vec2(0.0, pixel.y));
  sampler *= 0.25;

  return sampler;
}

float Gather1(in sampler2D tex, in vec2 coord){
  float sampler;

  sampler += texture(gaux2, coord + vec2(pixel.x, 0.0)).x;
  sampler += texture(gaux2, coord - vec2(pixel.x, 0.0)).x;
  sampler += texture(gaux2, coord + vec2(0.0, pixel.y)).x;
  sampler += texture(gaux2, coord - vec2(0.0, pixel.y)).x;
  sampler *= 0.25;

  return sampler;
}

//vec2 fragCoord = floor(texcoord * resolution * GI_Rendering_Scale);
//float checkerBoard = (mod(fragCoord.x + fragCoord.y, 2));
/*
vec3 GetShadowNormal(in vec2 coord, in float checkerBoard){
  coord.x += pixel.x * checkerBoard;
  return (texture2D(gaux2, coord).xyz * 2.0 - 1.0);
}

float GetShadowDepth(in vec2 coord, in float checkerBoard){
  coord.x += pixel.x * checkerBoard;
  return texture2D(gaux2, coord).a;
}

vec3 GetShadowAlbedo(in vec2 coord, in float checkerBoard){
  coord.x += pixel.x * (1.0 - checkerBoard);

  vec3 albedo = texture2D(gaux2, coord).rgb;
  albedo = normalize(albedo) * sqrt(getLum(albedo));

  return albedo;
}
*/

vec3 GetShadowNormal(in vec2 coord){
  return (texture2D(gaux2, coord + vec2(0.5, 0.0)).xyz * 2.0 - 1.0);
}

float GetShadowDepth(in vec2 coord){
  return texture(gaux2, coord + vec2(0.0, 0.5)).x;
}

vec3 GetShadowAlbedo(in vec2 coord){
  vec3 albedo = texture2D(gaux2, coord).rgb;
       //albedo = normalize(albedo) * (getLum(albedo));

  return albedo;
}

void CalculateRSM(inout vec3 shadowMapColor, in vec3 shadowPosition, in vec2 offset, in vec3 shadowSpaceNormal, in vec3 shadowSpaceLightDirection, inout float totalWeight, in float maxRadius){
  vec2 coord = shadowPosition.xy * float(shadowMapResolution);

  vec3 albedo = decodeGamma(texelFetch(shadowcolor1, ivec2(coord), 0).rgb);

  float alpha = texelFetch(shadowcolor1, ivec2(coord), 0).a;
  if(bool(step(0.999, alpha))) albedo.rgb = vec3(sqrt(invPi));

  vec3 normal = mat3(shadowModelView) * (texelFetch(shadowcolor0, ivec2(coord), 0).rgb * 2.0 - 1.0);
  float depth = texelFetch(shadowtex0, ivec2(coord), 0).x;

  vec3 position = vec3(shadowPosition.xy + offset, depth) - shadowPosition.xyz;
  //if(position.z <= 0.0) position.z *= 4.0;
       position = mat3(shadowProjectionInverse) * position;

  vec3 halfPosition = position;
  vec3 halfDirection = normalize(halfPosition);
  float halfLength = length(halfPosition);

  float cosTheta = saturate(dot(shadowSpaceNormal, halfDirection));
  float visible = saturate(dot(normal, -halfDirection));

  float attenuation = min(1.0, 1.0 / (halfLength * halfLength) * 6.6);

  vec3 F0 = vec3(0.02);

  vec3 kS = mix(F0, vec3(1.0), pow5(1.0 - saturate(dot(-halfDirection, normal))));
  vec3 kD = 1.0 - kS;

  vec3 radiance = albedo * visible * attenuation;

  shadowMapColor += radiance * cosTheta * kD;
  totalWeight += 1.0;
}

float ComputeAO(in vec3 p, in vec3 n, in vec3 s, in float r){
  vec3 v = s - p;
  float vdotv = dot(v, v);
  float vdotn = dot(-normalize(v), n) * inversesqrt(vdotv);

  float falloff = 1.0 + 1.0 / sqrt(r * r - vdotv);

  return saturate(max(0.0, vdotn - 1e-5) * falloff);
}

vec3 CalculateCoarseRSM(in vec4 worldPosition, in vec3 normal){
  vec3 shadowMapColor = vec3(0.0);
  float viewLength = length(worldPosition.xz);

  //#define GI_Lighting_AO
  #define GI_Rendering_Distance 100.0
  #define GI_Rendering_Distance_Fallout 20.0

  #define GI_ShadowMapCutOut sqrt(GI_Rendering_Distance / shadowDistance) + 0.05

  if(viewLength > GI_Rendering_Distance) return vec3(0.0);

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  vec3 shadowSpaceLight = mat3(shadowModelView) * worldLightVector;

  float totalWeight = 0.0;

  vec4 shadowPosition = (worldPosition);
  vec3 shadowSpaceNormal = mat3(gbufferModelViewInverse) * (normal);
       //shadowSpaceNormal.z *= 1.0 / 2048.0;
       shadowSpaceNormal = mat3(shadowModelView) * shadowSpaceNormal;

  shadowPosition = shadowModelView * shadowPosition;
  shadowPosition = shadowProjection * shadowPosition;
  shadowPosition /= shadowPosition.w;

  //float distortion = mix(1.0, length(shadowPosition.xy), SHADOW_MAP_BIAS) / 0.95;
  //shadowPosition.xy /= distortion;

  shadowPosition.xyz = shadowPosition.xyz * 0.5 + 0.5;
  shadowPosition.z -= shadowPixel;

  vec2 fragCoord = floor(shadowPosition.xy * resolution);
  float checkerBoard = mod(fragCoord.x + fragCoord.y, 2);

  //shadowPosition.xy = floor(shadowPosition.xy * resolution) * pixel - pixel * 0.5;
  //shadowMapColor = GetShadowNormal(shadowPosition.xy, checkerBoard);

  vec2 jitter = jittering;

  //float dither = R2sq(texcoord.xx * resolution * 0.5);
  //float dither = R2sq(texcoord * resolution * GI_Rendering_Scale);
  float blueNoise = GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jitter * 1.0);
  float blueNoise2 = GetBlueNoise(depthtex2, (1.0 - texcoord), resolution.y * 0.5, jitter * 1.0);
  float dither = blueNoise2;

  float maxRadius = 128.0;

  float roughness = 0.99;
  roughness *= roughness;

  vec2 E = vec2(blueNoise2, blueNoise);
       E.y = mix(E.y, 0.0, 0.7);
       //E.x = mix(E.x, 1.0, 0.7);

  float CosTheta = sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  int steps = 1;
  int directionStep = 4;
  float invsteps = 1.0 / float(steps);

  float ao = 0.0;
  vec3 viewPosition = mat3(gbufferModelView) * worldPosition.xyz;
  float aoRadius = invsteps * 341.0 * inversesqrt(viewPosition.z * viewPosition.z);

  float rayStep = 1.0;
  float radius = 0.0004 * maxRadius * SinTheta * invsteps;

  for(int i = 0; i < steps; i++){
    for(int j = 0; j < directionStep; j++){
      float angle = (float(j) + E.x) / float(directionStep) * 2.0 * Pi;
      vec2 direction = vec2(cos(angle), sin(angle));
           //direction = RotateDirection(direction, vec2(blueNoise, blueNoise2));
           direction *= rayStep;

      #ifdef GI_Lighting_AO
        vec2 aoDirection = direction * aoRadius;

        vec2 aoCoord = texcoord + aoDirection * pixel;
        if(floor(aoCoord) != vec2(0.0)) continue;

        vec3 v = nvec3(gbufferProjectionInverse * nvec4(vec3(aoCoord, texture(depthtex0, aoCoord).x) * 2.0 - 1.0));
        ao += ComputeAO(v, normal, viewPosition, 14.0 / 12.0);
      #endif

      direction *= radius;

      float distortion = mix(1.0, length(shadowPosition.xy * 2.0 - 1.0 + direction), SHADOW_MAP_BIAS) / 0.95;

      vec3 stepCoord = shadowPosition.xyz * 2.0 - 1.0 + vec3(direction, 0.0);
      vec3 shadowCoord = (stepCoord / vec3(vec2(distortion), 1.0)) * 0.5 + 0.5;

      if(length(shadowCoord.xy * 2.0 - 1.0) > GI_ShadowMapCutOut) continue;

      CalculateRSM(shadowMapColor, shadowCoord, direction, shadowSpaceNormal, shadowSpaceLight, totalWeight, maxRadius);
    }

    rayStep += 1.0;
  }

  shadowMapColor /= max(1.0, totalWeight);
  shadowMapColor *= saturate((-viewLength + (GI_Rendering_Distance - GI_Rendering_Distance_Fallout)) / GI_Rendering_Distance_Fallout);

  ao /= float(directionStep * steps);
  ao = saturate(1.0 - ao);

  return shadowMapColor;
}

void main() {
  vec3 color = vec3(0.0);

  vec3 normal = normalDecode(texture2D(composite, texcoord).xy);
  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

  vec2 coord = texcoord;
  coord -= jittering * pixel;

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex0, coord).x) * 2.0 - 1.0);vP/=vP.w;
  vec4 wP = gbufferModelViewInverse * vP;

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  float viewndotl = dot(worldNormal, worldLightVector);

  float viewLength = length(vP.xyz);

  color = CalculateCoarseRSM(wP, normal);
  color = encodeGamma(color);

  /* DRAWBUFFERS:1 */
  gl_FragData[0] = vec4(color, texture(depthtex0, texcoord).x);
}

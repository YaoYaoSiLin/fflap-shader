#version 130

#define GI_Rendering_Scale 0.5 //[0.353553 0.5]

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;
const bool  shadowHardwareFiltering = false;

#define gnormal colortex2
#define composite colortex3
#define gaux2 colortex5

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

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"

float shadowPixel = 1.0 / float(shadowMapResolution);
vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

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

#define LowResolutionShadowMap

void CalculateRSM(inout vec3 shadowMapColor, in vec3 shadowPosition, in vec2 offset, in vec3 shadowSpaceNormal, in vec3 shadowSpaceLightDirection, inout float totalWeight, in float maxRadius){
  //vec3 testPosition = shadowPosition;// + vec3(offset * radius, 0.0);
  //vec2 coord = floor(testPosition.xy * resolution) * pixel - pixel * 0.5;
  //vec2 coord = shadowPosition.xy;//round(testPosition.xy * resolution * 0.5) * pixel;

  #ifdef LowResolutionShadowMap
  vec2 coord = round(shadowPosition.xy * resolution * 0.5) * pixel;
  #else
  vec2 coord = shadowPosition.xy * float(shadowMapResolution);
  #endif

  //vec3 albedo = GetShadowAlbedo(coord);
  //vec3 normal = GetShadowNormal(coord);
  //float depth = GetShadowDepth(coord);
  //vec3 albedo = GetShadowAlbedo(coord, checkerBoard);
  //vec3 normal = GetShadowNormal(coord, checkerBoard);
  //float depth = GetShadowDepth(coord, checkerBoard);

  #ifdef LowResolutionShadowMap
    vec3 albedo = GetShadowAlbedo(coord);
    vec3 normal = GetShadowNormal(coord);
    float depth = GetShadowDepth(coord);
  #else
    vec3 albedo = (texelFetch(shadowcolor1, ivec2(coord), 0).rgb);

    float alpha = texelFetch(shadowcolor1, ivec2(coord), 0).a;
    if(bool(step(0.999, alpha))) albedo.rgb = vec3(0.0);

    vec3 normal = mat3(shadowModelView) * (texelFetch(shadowcolor0, ivec2(coord), 0).rgb * 2.0 - 1.0);
    float depth = texelFetch(shadowtex0, ivec2(coord), 0).x;
  #endif

  //albedo.rgb = normalize(albedo.rgb);
  albedo.rgb = L2Gamma(albedo.rgb);

  vec3 position = vec3(shadowPosition.xy + offset, depth) - shadowPosition.xyz;
  //if(position.z <= 0.0) position.z *= 4.0;
       position = mat3(shadowProjectionInverse) * position;

  float l = length(position.xyz);
        l = l*l*l*l+1e-5;

  vec3 direction = normalize(position);

  #if 1
  float ndotl = clamp01(dot(shadowSpaceNormal, direction) * 4.0);
        ndotl *= clamp01(dot(normal, -direction) * 4.0);
  #else
  float ndotl = step(0.01, dot(shadowSpaceNormal, direction));
        ndotl *= step(0.01, dot(normal, -direction));
  #endif
  //ndotl = pow5(ndotl);

  float irrdiance = max(0.0, dot(normal, shadowSpaceLightDirection));

  float weight = ndotl;

  shadowMapColor += albedo * weight * min(1.0, 1.0 / l * 128.0 * 2.0);
  totalWeight += 1.0;
}

vec3 CalculateCoarseRSM(in vec3 viewPosition, in vec3 normal){
  vec3 shadowMapColor = vec3(0.0);
  float viewLength = length(viewPosition);

  //if(viewLength > shadowDistance) return vec3(0.0);

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  vec3 shadowSpaceLight = mat3(shadowModelView) * worldLightVector;

  float totalWeight = 0.0;

  vec4 shadowPosition = (gbufferModelViewInverse) * nvec4(viewPosition);
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


  //float dither = R2sq(texcoord.xx * resolution * 0.5);
  float dither = R2sq(texcoord * resolution * GI_Rendering_Scale);
  float blueNoise = GetBlueNoise(depthtex2, texcoord * GI_Rendering_Scale, resolution.y, jittering * 1.0);
  float blueNoise2 = GetBlueNoise(depthtex2, (1.0 - texcoord) * GI_Rendering_Scale, resolution.y, jittering * 1.0);
  //dither = blueNoise;

  vec2 rotateAngle = vec2(dither, 1.0 - dither) * 0.9 + 0.1;

  float maxRadius = 128.0;
  float radius = (1.0 / 2048.0) * maxRadius;

  float roughness = 0.99;
  roughness *= roughness;

  vec2 E = vec2(blueNoise2, blueNoise);
       E.y = mix(E.y, 0.0, 0.7);
       //E.x = mix(E.x, 1.0, 0.7);

  float CosTheta = sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  for(int i = 0; i < steps; i++){
    float angle = (float(i) + E.x) * invsteps * 2.0 * Pi;
    vec2 offset = vec2(cos(angle), sin(angle)) * SinTheta;
         //offset = RotateDirection(offset, vec2(blueNoise, 1.0 - blueNoise));

    //if(i != 7) continue;

    offset *= radius;

    float distortion = mix(1.0, length(shadowPosition.xy * 2.0 - 1.0 + offset), SHADOW_MAP_BIAS) / 0.95;

    #ifdef LowResolutionShadowMap
    vec3 shadowCoord = shadowPosition.xyz + vec3(offset, 0.0);
    #else
    vec3 shadowCoord = shadowPosition.xyz * 2.0 - 1.0 + vec3(offset, 0.0);
         shadowCoord = (shadowCoord / vec3(vec2(distortion), 1.0)) * 0.5 + 0.5;
    #endif

    CalculateRSM(shadowMapColor, shadowCoord, offset, shadowSpaceNormal, shadowSpaceLight, totalWeight, maxRadius);
    //shadowMapColor += texture2D(shadowcolor1, shadowCoord.xy).rgb;
    //totalWeight += 1.0;
  }

  //shadowMapColor *= invsteps;
  shadowMapColor /= max(1.0, totalWeight);
  shadowMapColor = G2Linear(shadowMapColor);

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

  color = CalculateCoarseRSM(vP.xyz, normal);

  /* DRAWBUFFERS:3 */
  gl_FragData[0] = vec4(color, 1.0);
}

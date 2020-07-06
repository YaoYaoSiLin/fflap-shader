#version 130

#define GI_Rendering_Scale 0.353553

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;
const bool  shadowHardwareFiltering = false;

#define composite colortex3
#define gaux2 colortex5

uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;

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
#if 0

  shadowPosition.z -= 0.003;
  float shadow = step(testPosition.z, shadowDepth);

  //float depth2 = step()

  vec3 position = vec3(shadowPosition - vec3(testPosition.xy, shadowDepth));
       //position.z = step(position.z, 0.0) * 2.0 - 1.0;
       //position.z *= 4.0;
       position.z = 1.0;
       if(bool(shadowSpaceNormal)){
       position.z *= -1.0;//sign(-shadowSpaceNormal.g);
       //shadowSpaceNormal.z *= sign(shadowSpaceNormal.g);
       }
       //shadowSpaceNormal.xy *= -1.0;
       /*
       //if(shadowSpaceNormal.g < 0.0)


       //shadowSpaceNormal.z *= -1.0;
       if(shadowSpaceNormal.g < 0.0){
         shadowNormal.z *= -1.0;
         shadowSpaceNormal.xz *= -1.0;
       }
*/
       position = mat3(shadowModelViewInverse) * mat3(shadowProjectionInverse) * position;

  vec3 direction = normalize(position);

  float l = length(position.xyz);
        l = l*l*l*l;

  //shadowSpaceNormal *= sign(shadowSpaceNormal.g);

  float ndotl = max(0.0, dot(direction, shadowNormal));
        //ndotl *= max(0.0, dot(-direction, shadowSpaceNormal));

  shadowMapColor += shadowColorSample;
  totalWeight += 1.0;

  //if(texcoord.y > 0.5){
  //  shadowMapColor += texcoord.x > 0.5 ? shadowSpaceNormal : shadowNormal;
  //}else{
  //  shadowMapColor += texcoord.x > 0.5 ? max(0.0, dot(-direction, shadowSpaceNormal)) : max(0.0, dot(direction, shadowNormal));
  //}
/*

  float d = shadowDepth - 0.4 * length(direction.xy);
  direction.z = (coord.z - d) / d;
  //if(direction.z < 0.01) return;

  float l = length(direction.xy);
        l = l*l*l*l;

  //vec3 shadowSpaceFacingUp = vec3(0.0, 1.0, 0.0);
  //vec3 p0 = normalize(reflect(normalize(shadowLightPosition), shadowSpaceFacingUp));

  //direction.z = sign(dot(p0, shadowSpaceNormal));
  l = min(l, max(-direction.z, 0.001));
  direction.z = -sign(shadowSpaceNormal.b);
  if(direction.z < 0.0) shadowNormal *= -1.0;
  direction = normalize(direction);

  float ndotl = clamp01(dot(shadowNormal, direction));
        ndotl *= clamp01(dot(shadowSpaceNormal, -direction));
        //ndotl *= clamp01(pow5(dot(shadowNormal, shadowSpaceLight)) * 32.0);
        //ndotl = 1.0;
*/
#endif

float linearizeShadowMapDepth(float depth) {
    return (2.0 * near) / (shadowDistance + near - depth * (shadowDistance - near));
}

vec2 RotateDirection(vec2 V, vec2 RotationCosSin) {
    return vec2(V.x*RotationCosSin.x - V.y*RotationCosSin.y,
                V.x*RotationCosSin.y + V.y*RotationCosSin.x);
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
       albedo = normalize(albedo) * (getLum(albedo));

  return albedo;
}

void CalculateRSM(inout vec3 shadowMapColor, in vec3 shadowPosition, in vec2 offset, in vec3 shadowSpaceNormal, in vec3 shadowSpaceLightDirection, inout float totalWeight, in float checkerBoard){
  float maxRadius = 8.0;

  float radius = 0.0003 * maxRadius;

  vec3 testPosition = shadowPosition + vec3(offset * radius, 0.0);
  //vec2 coord = floor(testPosition.xy * resolution) * pixel - pixel * 0.5;
  vec2 coord = round(testPosition.xy * resolution * 0.5) * pixel;

  vec3 albedo = GetShadowAlbedo(coord);
  vec3 normal = GetShadowNormal(coord);
  float depth = GetShadowDepth(coord);
  //vec3 albedo = GetShadowAlbedo(coord, checkerBoard);
  //vec3 normal = GetShadowNormal(coord, checkerBoard);
  //float depth = GetShadowDepth(coord, checkerBoard);

  vec3 position = vec3(shadowPosition.xy + offset, depth) - shadowPosition.xyz;
       position.z = position.z * 512.0 - 1.0;
       position = mat3(shadowProjectionInverse) * position;

  vec3 direction = normalize(position);

  position *= 0.1;
  float l = length(position.xyz * radius);
        l = l*l*l*l*0.75 + 0.25;

  float ndotl = clamp01(dot(shadowSpaceNormal, direction) * 10.0);
        ndotl *= max(0.0, dot(normal, -direction));

  float irrdiance = max(0.0, dot(normal, shadowSpaceLightDirection));

  shadowMapColor += albedo * ndotl * irrdiance / l;
  totalWeight += ndotl;
}

vec3 CalculateCoarseRSM(in vec3 viewPosition, in vec3 normal){
  vec3 shadowMapColor = vec3(0.0);
  float viewLength = length(viewPosition);

  //if(viewLength > shadowDistance) return vec3(0.0);

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  vec3 shadowSpaceLight = mat3(shadowModelView) * worldLightVector;

  float totalWeight = 0.0;

  vec4 shadowPosition = (gbufferModelViewInverse) * nvec4(viewPosition);
  vec3 shadowSpaceNormal = mat3(shadowModelView) * mat3(gbufferModelViewInverse) * (normal);
  //shadowSpaceNormal.xy = -shadowSpaceNormal.xy;

  shadowPosition = shadowModelView * shadowPosition;
  shadowPosition = shadowProjection * shadowPosition;
  shadowPosition /= shadowPosition.w;
  shadowPosition.xyz = shadowPosition.xyz * 0.5 + 0.5;
  shadowPosition.z -= 0.003;

  vec2 fragCoord = floor(shadowPosition.xy * resolution);
  float checkerBoard = mod(fragCoord.x + fragCoord.y, 2);

  //shadowPosition.xy = floor(shadowPosition.xy * resolution) * pixel - pixel * 0.5;
  //shadowMapColor = GetShadowNormal(shadowPosition.xy, checkerBoard);


  //float dither = R2sq(texcoord.xx * resolution * 0.5);
  float dither = R2sq(texcoord * resolution * 0.353553);
  vec2 rotateAngle = vec2(dither, 1.0 - dither) * 0.9 + 0.1;

  for(int i = 0; i < 40; ++i){
    float angle = mod(float(i), 9) * 0.125 * 2.0 * Pi;
    vec2 offset = vec2(cos(angle), sin(angle));
         offset *= 1.0 + floor(float(i) * 0.125);
         offset = RotateDirection(offset, rotateAngle);

    //if(i != 10) continue;

    CalculateRSM(shadowMapColor, shadowPosition.xyz, offset, shadowSpaceNormal, shadowSpaceLight, totalWeight, checkerBoard);
  }

  shadowMapColor /= max(1.0, totalWeight);
  shadowMapColor *= 0.125;


  //shadowPosition.x -= pixel.x;
  //shadowPosition.x += checkerBoard * pixel.x;

  //vec2 fragCoord = floor(shadowPosition.xy * resolution);
  //float checkerBoard = mod(fragCoord.x + fragCoord.y, 2);

  //shadowPosition.x -= pixel.x;
  //shadowPosition.x += pixel.x * checkerBoard;
  //shadowMapColor = texture2D(gaux2, shadowPosition.xy).rgb;
  //shadowMapColor = shadowSpaceNormal;

  //shadowPosition.xy = floor(shadowPosition.xy * resolution) * pixel - pixel * 0.5;
  //shadowPosition.x += pixel.x;

  //shadowMapColor = texture2D(gaux2, shadowPosition.xy).rgb;

  /*
  for(int i = 0; i < 32; ++i){
    float angle =


    //float r = float(8.0) - 4.0;
    //vec2 offset = vec2(0.0, r);

    CalculateRSM(shadowMapColor, shadowPosition.xyz, offset, shadowSpaceNormal, totalWeight);
  }

  */
  /*
  vec2 o = vec2(0.0, -4.0);

  shadowPosition.xy = shadowPosition.xy * 2.0 - 1.0;
  float distort = 1.0 / (mix(1.0, length(shadowPosition.xy), SHADOW_MAP_BIAS) / 0.95);
  shadowPosition.xy *= distort;
  shadowPosition.xy = shadowPosition.xy * 0.5 + 0.5;
  shadowPosition.z -= 0.003;

  vec3 shadowMapAlbedo = texture2D(shadowcolor1, shadowPosition.xy + o * 0.001 * distort).rgb;

  vec3 shadowMapNormal = mat3(shadowModelView) * (texture2D(shadowcolor0, shadowPosition.xy + o * 0.001 * distort).rgb * 2.0 - 1.0);

  float shadowMapDepth = texture(shadowtex0, shadowPosition.xy + o * 0.001 * distort).x;

  vec3 lightPosition = shadowPosition.xyz - vec3(shadowPosition.xy + o, shadowMapDepth);
       //lihgtPosition

  //-(step(shadowPosition.z - 0.004 - shadowMapDepth, 0.0) * 2.0 - 1.0)
  vec3 d = lightPosition;
  //d.z = -sign(shadowSpaceNormal.y);
  if(shadowSpaceNormal.y < 0.0)
  d.z = (step(d.z, 0.0) * 2.0 - 1.0) * 3.0;

  //d.z = sign(shadowSpaceNormal.y);
  d = mat3(shadowProjectionInverse) * (d);

  d = normalize(d);

  float shadow = step(shadowPosition.z, shadowMapDepth + 0.001);
        //shadow *= step(shadowMapDepth - 0.05, shadowPosition.z);

  shadowSpaceNormal.x = -shadowSpaceNormal.x;
  float normalAngle = max(0.0, dot(d, shadowSpaceNormal));
  //shadowMapColor = vec3((max(0.0, dot(-d, shadowSpaceNormal))));



  vec3 d2 = lightPosition;
       d2.z = d2.z * 128.0;
       //d2.z = sign(shadowSpaceNormal.y);
       d2 = mat3(shadowProjectionInverse) * d2;
       d2 = normalize(d2);

  //shadowMapNormal.xy *= -1.0;

  float indirectAngle = max(0.0, dot(d2, shadowMapNormal));

  //shadowMapNormal.z = -shadowMapNormal.z;
  shadowMapColor = shadowMapAlbedo * normalAngle;
  */
  //shadowMapColor = shadowMapNormal * vec3(1.0, 1.0, 1.0);
  //vec3 d2 = vec3(o, )

  //shadowMapColor = vec3(max(0.0, dot(d2, shadowMapNormal)));

  /*
  for(int i = 0; i < 32; ++i){
    float angle = mod(float(i), 9) * 0.125 * 2.0 * Pi;
    vec2 offset = vec2(cos(angle), sin(angle));
         offset *= 1.0 + floor(float(i) * 0.125);
         offset = -RotateDirection(offset, rotateAngle);

    if(i != 10) continue;

    CalculateRSM(shadowMapColor, shadowPosition.xyz, offset, shadowSpaceNormal, totalWeight);
  }
  */

  //vec2 offset =

  //CalculateRSM(shadowMapColor, shadowPosition.xyz, offset, shadowSpaceNormal, totalWeight);


  //vec3 testPosition = shadowPosition.xyz + vec3(offset * 0.001, 0.0);
  //vec3 coord = testPosition;
  //vec3 direction = vec3(0.0);


  //shadowMapColor *= clamp01(dot(shadowSpaceNormal, vec3(1.0, -1.0, -sign(shadowSpaceNormal.b))));

  //shadowMapColor += clamp01(dot(shadowSpaceNormal.xyz, vec3(0.0, 1.0, 0.0)));

  return shadowMapColor * 1.0;
}

void main() {
  vec3 color = vec3(0.0);

  vec3 normal = normalDecode(texture2D(composite, texcoord).xy);
  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

  //vec3 closest = GetClosest(texcoord); closest.xy -= jittering * pixel;
  vec3 closest = vec3(texcoord - jittering * pixel, texture(depthtex0, texcoord - jittering * pixel).x);
  vec4 vP = gbufferProjectionInverse * nvec4(closest * 2.0 - 1.0);vP/=vP.w;
  vec4 wP = gbufferModelViewInverse * vP;

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  float viewndotl = dot(worldNormal, worldLightVector);

  float viewLength = length(vP.xyz);

  float dither = R2sq(texcoord*resolution*GI_Rendering_Scale-jittering);

  int count = 0;

  vec3 shadowMapColor = vec3(0.0);

  float shadowBufferSize = 0.5;

  float distort = 0.0;

  vec3 back = normalize(nvec3(gbufferProjectionInverse * nvec4(vec3(vec2(0.5), 0.7) * 2.0 - 1.0)));
  vec3 pi = normalize(reflect(-back, worldNormal));
  vec3 halfi = normalize(pi + worldNormal);

  float ndotu = dot(vec3(0.0, 1.0, 0.0), worldNormal);

  vec4 shadowPosition = shadowModelView * wP;
  vec3 shadowSpaceNormal = mat3(shadowModelView) * worldNormal;
  vec3 shadowSpaceLight = mat3(shadowModelView) * worldLightVector;

  float shade = step(dot(normal, normalize(shadowLightPosition)), 0.1);
  //shadowSpaceNormal.z = -shadowSpaceNormal.z;

  float invFar = 1.0 / far;
  float maxRange = invFar * 8.0;

  int steps = 8;
  float invsteps = 1.0 / float(steps);
  float alpha = invsteps * 2.0 * Pi;

  float totalWeight = 0.0;

  //float largeIDstep = 1.0 + floor(viewLength * 0.03125);
  //float largeIDStart = 0.0;
  //float largeIDEnd = 8.0;
  //float largeAlpha = 1.0 / largeIDEnd * 2.0 * Pi;

  vec2 rotateAngle = vec2(dither * 0.8 + 0.2, 0.0);
       rotateAngle.y = 1.0 - rotateAngle.x;

  /*
  for (int i = 0; i < steps; ++i){
    if(count > 100) {
      shadowMapColor = vec3(1.0, 0.0, 0.0);
      break;
    }

    float angleLarge = float(i) * alpha;
    vec2 large = vec2(cos(angleLarge), sin(angleLarge));
         //if(i > 8)
         large *= Pi;
         //large = RotateDirection(large, rotateAngle);
    */
    float smallIDstep = 1.0;
    float smallIDStart = 0.0;
    float smallIDEnd = 8.0;
    float smallIDAlpha = 1.0 / smallIDEnd * 2.0 * Pi;
    /*
    while(smallIDStart < smallIDEnd){
      smallIDStart += smallIDstep;

      //if(smallIDStart != 8.0) continue;
      //if(count > 8) break;

      float angleSmall = smallIDStart * smallIDAlpha;
      vec2 small = vec2(cos(angleSmall), sin(angleSmall));
           small = RotateDirection(small, rotateAngle);

      //for(int i = 0; i < 4; i++){
      //  float angle = float(i + dither) * 0.25 * 2.0 * Pi;
      for(float i = -1.0; i <= 1.0; i += 1.0){
        for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 large = vec2(i, j) * Pi;
        //vec2 large = vec2(0.0);
        //large = vec2(cos(angle), sin(angle)) * Pi * 2.0;
        large = RotateDirection(large, rotateAngle);

        vec2 offset = (large + small);

        vec3 testPosition = shadowPosition.xyz + vec3(offset, 0.0);
        vec3 direction = testPosition.xyz - shadowPosition.xyz;
             direction = mat3(shadowModelView) * direction;
        //testPosition.x += 1.0;

        testPosition = nvec3(shadowProjection * vec4(testPosition, 1.0));
        vec3 coord = testPosition * 0.5 + 0.5;
        if(floor(coord) != vec3(0.0)) continue;

        count++;
        break;
        }
      }
    }
    */

  color = CalculateCoarseRSM(vP.xyz, normal);

  //color = vec3(dot(shadowSpaceNormal, vec3(0.0, 1.0, 0.0)));

  //color = -shadowSpaceNormal.xyz;


  /* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color, 1.0);
}

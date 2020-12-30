#version 130

#define LightShaft_Quality 0.5

uniform sampler2D noisetex;

uniform sampler2D depthtex0;
uniform sampler2D depthtex2;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor1;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform float biomeTemperature;
uniform float biomeRainFall;

uniform int frameCounter;

uniform vec3 sunPosition;
uniform vec3 shadowLightPosition;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

in vec2 texcoord;

in float fading;

in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/atmospheric.glsl"

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;
const bool  shadowHardwareFiltering = false;
float shadowPixel = 1.0 / float(shadowMapResolution);

const int noiseTextureResolution = 64;

vec4 GetViewPosition(in vec2 coord, in sampler2D depth){
  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, texture(depth, coord).x) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

vec3 wP2sP(in vec4 wP, out float bias){
	vec4 sP = shadowModelView * wP;
       sP = shadowProjection * sP;
       sP /= sP.w;

  bias = 1.0 / (mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95);

  sP.xy *= bias;
  sP.z /= max(far / shadowDistance, 1.0);
  sP = sP * 0.5 + 0.5;

	return sP.xyz;
}

float HG(in float m, in float g){
  return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}

vec4 CalculateSunVisibility(in vec4 rayOrigin, in vec3 rayDirection, in float dither){
  vec4 raysColor = vec4(0.0);

  int steps = 1;
  float invsteps = 1.0 / float(steps);

  rayDirection *= invsteps;

  vec4 testPoint = rayOrigin + vec4(rayDirection * dither - rayDirection, 0.0);

  float bias = 0.0;
  float diffthresh = shadowPixel * 1.0;

  //for(int i = 0; i < steps; i++){
    testPoint.xyz -= rayDirection;

    vec3 shadowMapCoord = wP2sP(testPoint, bias);
         shadowMapCoord.z -= diffthresh;

    float d0 = texture(shadowtex0, shadowMapCoord.xy).x;

    raysColor.a += step(shadowMapCoord.z, d0);
  //}

  return raysColor * invsteps;
}

vec4 CalculateRays(in vec4 wP, in float dither, in bool isSky){
  vec4 raysColor = vec4(0.0);

  if(isSky) return vec4(1.0);

  int steps = 12;
	float invsteps = 1.0 / float(steps);

	vec3 rayDirection = normalize(wP.xyz) * length(wP.xyz) * invsteps;
  vec4 rayStart = wP + vec4(rayDirection, 0.0) * dither;

  float bias = 0.0;
  float diffthresh = shadowPixel * 1.0;

  for(int i = 0; i < steps; i++){
    rayStart.xyz -= rayDirection.xyz;

    vec3 shadowMap = wP2sP(rayStart, bias);

		float d = texture2D(shadowtex1, shadowMap.xy).x + diffthresh;
    float d2 = texture2D(shadowtex0, shadowMap.xy).x + diffthresh;

    float sampleShading = step(shadowMap.z, d);
    float sampleShading2 = step(shadowMap.z, d2);

    //float step
		raysColor.a += sampleShading;// * (1.0 - exp(-length(rayStart.xyz) * 0.1));

    vec4 sampleTexture = texture2D(shadowcolor1, shadowMap.xy);
    sampleTexture.rgb *= exp(-sampleTexture.a * Pi * (1.0 - sampleTexture.rgb));
    raysColor.rgb += mix(sampleTexture.rgb, vec3(1.0), step(sampleShading, sampleShading2));
  }

  raysColor *= invsteps;

  return raysColor;
}

vec2 hash2(in vec2 p){
  return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

float voronoi(in vec2 x){
  vec2 n = floor(x);
  vec2 f = fract(x);

  float md = 8.0;

  for(int i = -1; i <= 1; i++) {
    for(int j = -1; j <= 1; j++) {
      vec2 g = vec2(i, j);
      vec2 o = hash2(n + g);

      vec2 r = g + o - f;
      float d = dot(r, r);

      md = min(d, md);
    }
  }

  return md;
}
/*
float noise( in vec3 x )
{
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    return mix(mix(mix( hash(i+vec3(0,0,0)),
                        hash(i+vec3(1,0,0)),f.x),
                   mix( hash(i+vec3(0,1,0)),
                        hash(i+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(i+vec3(0,0,1)),
                        hash(i+vec3(1,0,1)),f.x),
                   mix( hash(i+vec3(0,1,1)),
                        hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}

float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ),
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ),
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}
*/

float noise(in vec2 x){
  return texture(noisetex, (x + 0.5) / float(noiseTextureResolution)).x;
}

float noise(in vec3 x) {
  x = x.xzy;

  vec3 i = floor(x);
  vec3 f = fract(x);

	f = f*f*(3.0-2.0*f);

	vec2 uv = (i.xy + i.z * vec2(17.0)) + f.xy;
  //uv = round(uv * float(noiseTextureResolution)) / float(noiseTextureResolution);
  uv += 0.5;

	vec2 rg = vec2(texture(noisetex, (uv) / float(noiseTextureResolution)).x,
                 texture(noisetex, (uv+17.0) / float(noiseTextureResolution)).x);

	return mix(rg.x, rg.y, f.z);
}

float CloudsShape(in vec2 p, float density){
  float shape =  0.0;

  float scale = 0.00066;
  p *= scale;

  shape += noise(p) * 0.5;
  shape += noise(p * 2.0) * 0.25;
  shape += noise(p * 4.0) * 0.125;
  shape = rescale(density, 0.875, shape);

  return shape;
}

float ApplyCloudsNoise(in float shape, float i, in vec3 p){
  float scale = 0.005;
  p *= scale;

  float perlin = abs(noise(p)-0.5) * 0.5;
        perlin += abs(noise(p*2.0)-0.5) * 0.25;
        perlin += abs(noise(p*4.0)-0.5) * 0.125;
        perlin /= 0.875;

  float notWorley = noise(p) * 0.5;
        notWorley += noise(p * 2.0) * 0.25;
        notWorley = rescale(0.1, 0.75, notWorley);

  float n = (perlin * 2.0) + notWorley;

  return rescale(n * i, 1.0, shape);
}

float CloudsSelfShaow(in vec3 p, in vec3 d, in float density){
  float shadow = 1.0;



  return shadow;
}

void CalculateFarVolumetric(inout vec4 volumetric, in vec3 rayOrigin, in vec3 rayDirection, in vec3 lightingDirection, in float dist, float dither){
  vec4 clouds = vec4(0.0);

  rayOrigin.y += rE;

  float earth = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), rE).x;

  float end = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), rE + 3000.0).y;
  float start = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), rE + 1500.0).y;

  float end2 = RaySphereIntersection(rayOrigin, vec3(0.0, 1.0, 0.0), vec3(0.0), rE + 3000.0).y;
  float start2 = RaySphereIntersection(rayOrigin, vec3(0.0, 1.0, 0.0), vec3(0.0), rE + 1500.0).y;

  if(start > dist || rayDirection.y < 0.0) return;

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  float testPoint = (end - start) * invsteps;

  float testDepth = (end2 - start2) * invsteps;
  float opticalDepth = 0.0;

  vec3 b = vec3(0.001);

  vec3 testPosition = rayDirection * (start + testPoint * dither);

  vec3 lightingStart = rayDirection * end;
  vec3 lightingStep = lightingDirection * invsteps * testPoint * 0.00066;
  vec3 lightingTest = lightingStep * 7.0 + lightingStep * dither;

  //shadowShape *= 0.25;

  vec3 lightStep = lightingDirection * 0.25;

  for(int i = 0; i < steps; i++){
    float shape = max(0.0, floor(hash(round(testPosition.xz * 0.00066)) * 1.2));

    vec3 tau = b * (opticalDepth + dither * testDepth * shape);
    vec3 attenuation = exp(-tau);
    //attenuation = vec3(1.0);

    float sunVisibility = 0.0;

    float shadowShape = 0.0;
    vec3 lightTest = lightStep * dither;

    for(int i = 0; i < 4; i++){
      float shape = max(0.0, floor(hash(round(testPosition.xz * 0.00066 + lightTest.xz)) * 1.2));
      shadowShape += shape;
      lightTest += lightStep;
    }

    //shadowShape *= 0.25;

    float shadowDepth = length(rayDirection * end2 - rayDirection * (end2 - start2) * invsteps * (dither + float(i)));
          //shadowDepth = max(0.0, shadowDepth - rE - 1500.0);
    sunVisibility = exp(-shadowShape);//exp(-b.x * 0.03 * shadowDepth * shadowShape);
    //sunVisibility = exp(-shadowDepth * 0.1 * b.x * (1.0 - shadowShape * 0.25));

    //sunVisibility = exp(-b.x * 100.0 * shadowShape);

    //vec3 cloudsColor = sunVisibility * sunLightingColorRaw + skyLightingColorRaw;
    vec3 cloudsColor = vec3(invPi);

    clouds.a += (1.0 - shape * dot03(attenuation));
    clouds.rgb += shape * cloudsColor * (attenuation);

    testPosition += rayDirection * testPoint;
    opticalDepth += shape * testDepth;
  }

  clouds.a *= invsteps;
  //clouds.a = min(1.0, clouds.a);

  /*
  float sunVisibility = 1.0;

  vec3 lP = testPosition * 0.00066 + lightingDirection * dither * 0.25;

  for(int i = 0; i < 4; i++){
    float shape = max(0.0, floor(hash(round(lP.xz)) * 1.2));

    sunVisibility *= exp(-shape * 1.0);
    lP += lightingDirection * 0.25;
  }
  */
  /*
  float opticalDepth = shape * testPoint;

  vec3 tau = b * opticalDepth;
  vec3 attenuation = exp(-tau);

  //vec3 cloudsColor = sunVisibility * sunLightingColorRaw + skyLightingColorRaw;
  vec3 cloudsColor = vec3(invPi);

  clouds.a = 1.0;
  clouds.rgb = shape * attenuation * cloudsColor;
  */


  //clouds.rgba = vec4(max(0.0, floor(hash(round(testPosition.xz * 0.00066)) * 1.2)));

  volumetric = clouds;
}
#if 0
  vec4 clouds;

  rayOrigin.y = max(1.0, rayOrigin.y);
  rayOrigin.y += rE;

  float cloudsTop = 3000.0;
  float cloudsBottom = 1500.0;

  vec2 t = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), rE + cloudsTop);
  if(t.x > t.y) t = t.yx;
  t.x = max(0.0, t.x);
  vec2 tLower = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), rE + cloudsBottom);
  if(tLower.x > tLower.y) tLower = tLower.yx;
  tLower.x = max(0.0, tLower.x);
  vec2 tE = RaySphereIntersection(vec3(0.0, rE + 1.0, 0.0), vec3(0.0, -1.0, 0.0), vec3(0.0), rE);
  vec2 tL = RaySphereIntersection(vec3(0.0, rA + 1.0, 0.0), lightingDirection, vec3(0.0), rA);

  float cloudsMiddle = tLower.y + (t.y - tLower.y) * 0.5;

  if(tLower.y > dist || rayDirection.y < 0.0) return;
  t.y = min(t.y, dist);
  /*
  if(tE.x > 0.0) {
    t.y = min(tE.y, t.y);
    tLower.y = min(tLower.y, tE.y);
  }
  */

  mat2 r1 = mat2(1.6, 1.2, -1.2, 1.6);

  #if 1
  float start = tLower.y - tLower.x;
  float end = t.y - t.x;
  float test = start;

  float stepLength = (end - start) / 8.0;

  test += stepLength * mix(dither, 1.0, 0.);

  int count = 0;

  float opticalDepth = 0.0;

  vec3 b = vec3(0.0001);

  vec3 phase = max(b * invPi * 60.0 * fading, vec3(HG(dot(lightingDirection, rayDirection), 0.8)));
  vec3 sunVisibility = vec3(1.0);

  //dither = mix(dither, 1.0, 0.0);
  //rayDirection += start * rayDirection - start * rayDirection * dither;;

  //clouds.rgba = vec4(0.0, 0.0, 1.0, 1.0);

  vec3 attenuationSum = vec3(0.0);

  clouds.a = 0.0;
  float isClouds = 0.0;
  vec3 cloudsExtinction = vec3(0.0);

  //rayDirection.xz = r1 * rayDirection.xz;


  //vec3 Tr = bR * exp(-1500.0 / Hr);
  //vec3 Tm = bM * exp(-1500.0 / Hm);

  //float h = max(0.0, length(tLower.y * rayDirection * vec3(0.0, 1.0, 0.0)));
  //float s = max(0.0, length(tLower.y * rayDirection));

  //vec3 atmospheric = InScattering(vec3(0.0, 0.0, 0.0), rayDirection, lightingDirection, h, s, 0.76, dot(lightingDirection, rayDirection));

  //if(tL.x > tL.y) tL = tL.yx;
  //tL.x = max(0.0, tL.x);

  //float fading2 = saturate((max(0.0, tL.y - tL.x) - max(0.0, tLower.x)) * 0.1);
  float fading2 = 1.0;//saturate(max(0.0, -tL.x) - max(0.0, tE.x));
  //float fading2 = minComponent(Extinction(vec3(0.0, 1500.0, 0.0), lightingDirection));

  //float d = max(stepLength - start, 0.0);
  float d = max(0.0, stepLength) * 0.125;

  //while(test <= end && count < 8){

  for(int i = 0; i < 8; i++){
    vec3 testPoint = (rayDirection) * test;

    float h = testPoint.y;
    float s = length(testPoint);
    //float d = max(0.0, test - start);

    float shape = CloudsShape(testPoint.xz, 0.4);

    float density = exp(-abs(2250.0 - length(h)) / 600.0);
    shape *= density;
    shape = ApplyCloudsNoise(shape, 0.2 / (1.0 + density), testPoint);

    shape = max(0.0, floor(hash(round(testPoint.xz * 0.00066)) * 1.2));
    float shape2 = max(0.0, floor(hash(round(testPoint.xz * 0.00066 + lightingDirection.xz * 0.16)) * 1.2));
    shape2 = 1.0 - exp2(-shape2 * 1000.0);

    opticalDepth += shape * d;

    //float stepv = exp(-CloudsShape(testPoint.xz + lightingDirection.xz * 0.1 * test, 0.5 * 1.1) * 1024.0 * b.x);
    //sunVisibility -= stepv / float(count + 1) * invPi;
    float svclouds = CloudsShape(testPoint.xz + lightingDirection.xz * 0.1 * test, 0.4 * 1.1);
    //vec3 stepVisibility = exp(-svclouds * 8192.0 * b) * exp(8192.0 * b);
    vec3 stepVisibility = vec3(1.0);//exp(-(svclouds) * b * 1200.0);
    //sunVisibility = min(sunVisibility, stepVisibility);
    //sunVisibility = (sunVisibility + stepVisibility) * 0.5;
    //sunVisibility += (stepVisibility * 0.0625);//mix(sunVisibility, vec3(1.0), stepVisibility);

    vec3 tau = b * opticalDepth;
    vec3 attenuation = exp(-tau);

    vec3 beerspowder = (1.0 - exp(-stepVisibility * 2.0)) * exp(-stepVisibility);

    vec3 sunLighting = sunLightingColorRaw * sunVisibility * phase * beerspowder * 12.0 * fading2;
    vec3 skyLighting = skyLightingColorRaw * attenuation;
    //vec3 earthLighting = InScattering(vec3(0.0, h, 0.0), vec3(0.0, -1.0, 0.0), lightingDirection, 0.0, h, 0.76, dot(lightingDirection, vec3(0.0, -1.0, 0.0)));

    vec3 Tr = bR * exp(-h / Hr);
    vec3 Tm = bM * exp(-h / Hm);
    vec3 extinction = exp(-(Tr + Tm) * s);
    vec3 atmospheric = InScattering(vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), lightingDirection, h, s, 0.76, dot(lightingDirection, rayDirection));

    vec3 cloudsColor  = (sunLighting + skyLighting) * attenuation;
    //     cloudsColor += earthLighting * (1.0 - attenuation);
         cloudsColor *= b * 1000.0 * extinction;
    //     cloudsColor += atmospheric * (sunLightingColorRaw) * fading;

    cloudsColor = sunLightingColorRaw * (1.0 - shape2) + skyLightingColorRaw * Pi;
    cloudsColor *= (1.0 - attenuation) * 100.0;

    clouds.a += (1.0 - shape * maxComponent(attenuation));
    clouds.rgb += cloudsColor * shape;

    test += stepLength;
    count++;
  }

  clouds.a = clouds.a / 7.0;

  //clouds.a *= 2.0;

  float h = length(tLower.y * rayDirection * vec3(0.0, 1.0, 0.0));
  float s = max(0.0, length(tLower.y * rayDirection));

  vec3 earthLighting = InScattering(vec3(0.0, h, 0.0), vec3(0.0, -1.0, 0.0), lightingDirection, h, h, 0.76, dot(lightingDirection, vec3(0.0, -1.0, 0.0)));
  //clouds.rgb += cloudsExtinction * earthLighting * dot03(sunLightingColorRaw) * fading;

  //clouds.rgb = InScattering(vec3(0.0, h, 0.0), vec3(0.0, -1.0, 0.0), lightingDirection, h, h, 0.76, dot(lightingDirection, vec3(0.0, -1.0, 0.0)));

  //vec3 earthLighting = InScattering(vec3(0.0, h, 0.0), vec3(0.0, -1.0, 0.0), lightingDirection, h, h, 0.76, dot(lightingDirection, vec3(0.0, -1.0, 0.0)));
  //clouds.rgb  = CalculateInScattering(vec3(0.0, h, 0.0), rayDirection, lightingDirection, 0.76, ivec2(3, 1));

  //clouds.rgb += (1.0 - clouds.a) * earthLighting * dot03(sunLightingColorRaw) * step(0.5, texcoord.x);

  //clouds.rgb += (skyLightingColorRaw + sunLightingColorRaw * 0.0001) * max(0.0, 1.0 - clouds.a) * float(isClouds) * 0.1;
  //clouds.rgb *= 10000.0 * b;

  //clouds.rgb += atmospheric * cloudsExtinction * 0.125;

  /*
  float h = 1500.0;
  float s = 1500.0;

  vec3 Tr = bR * exp(-h / Hr);
  vec3 Tm = bM * exp(-h / Hm);

  vec3 extinction = 1.0 - exp(-(bR + bM) * s);
  clouds.rgb = extinction;
  */

  //clouds.rgb = exp(-(start - tE.y) * (Tr+Tm));
  //clouds.rgb += exp(-(e))
  //clouds.rgb = vec3(0.0, 0.0, 1.0) * clouds.a * 0.125;
  //clouds.a *= maxComponent(b);
  //clouds.rgb += vec3(0.0, 0.0, 1.0) * (clouds.a);

  //clouds.a = maxComponent(attenuationSum);

  //if(clouds.r < 0.0) clouds.rgb = vec3(1.0, 0.0, 0.0);
  //clouds.rgb = vec3(sunVisibility);
  //clouds

  #else
  vec3 rayStart = rayDirection * tLower.y;

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  vec3 rayEnd = rayDirection * t.y;
  vec3 increment = (rayEnd - rayStart) * invsteps;

  vec3 testPoint = rayStart - increment * dither * 1.0;

  for(int i = 0; i < steps; i++){
    testPoint += increment;

    float shape = CloudsShape(testPoint.xz, 0.375);

    float density = exp(-abs(testPoint.y - 2250.0) / 1100.0);
    shape *= density;

    shape = ApplyCloudsNoise(testPoint, shape, 0.1);

    clouds += shape;
  }
  #endif

  volumetric = clouds;
  //if(rayDirection.y < 0.0) volumetric = vec4(0.0);
  //if(length(testPoint) > length(wP) * tA.y) volumetric = vec4(0.0);
}
#endif

void CalculateNearVolumetric(inout vec4 volumetric, in vec4 rayOrigin, in vec3 rayDirection, in float dither){
  float mu = dot(normalize(rayOrigin.xyz), rayDirection);
  float phaseM = HG(mu, 0.76);

  int steps = 4;
  float invsteps = 1.0 / float(steps);

  vec4 rayStep = vec4(normalize(rayOrigin.xyz), 0.0) * length(rayOrigin.xyz) * invsteps;
  vec4 rayStart = rayOrigin - rayStep * (dither + 0.05);

  vec3 m = vec3(0.0);
  vec3 r = vec3(0.0);

  vec3 Tm = bM * 10.0;
  vec3 Tr = bR * 10.0;

  for(int i = 0; i < steps; i++){
    float distort = 0.0;
    vec3 shadowCoord = wP2sP(rayStart, distort);

    float visibility = step(shadowCoord.z, texture(shadowtex0, shadowCoord.xy).x + shadowPixel);

    float currentmu = dot(rayDirection, normalize(rayStart.xyz));
    float phaseM = HG(currentmu, 0.76);
    float phaseR = 0.0596831 * (1.0 + currentmu * currentmu);

    vec3 scattering = 1.0 - exp(-(Tm + Tr) * length(rayStart.xyz));
    //vec3 extinction = vec3(step(100.0));//exp(-bM * 100.0 * );

    m += visibility * scattering * phaseM;

    rayStart -= rayStep;
  }

  vec3 scattering = m;
       scattering *= invsteps * sunLightingColorRaw;

  volumetric.rgb += scattering;

  //vec3 m = (sunLightingColorRaw) * scattering * 40.0;
}

void main() {
  vec2 coord = texcoord - jittering * pixel * 1.0;
	vec4 vP = GetViewPosition(coord, depthtex0);
	vec4 wP = gbufferModelViewInverse * vP;
  vec3 lightingDirection = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  float viewLength = length(vP.xyz);

  vec3 lightDirection = normalize(shadowLightPosition);

	float dither = 1.0;
  //dither = R2sq((texcoord * resolution - jittering) * LightShaft_Quality);
  dither = GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jittering);
  //dither = mix(dither, 1.0, 0.3);

  float isSky = step(0.9999, texture(depthtex0, texcoord).x);
  /*
  float isSky = 1.0;

  for(float i = -2.0; i <= 2.0; i += 1.0){
    for(float j = -2.0; j <= 2.0; j += 1.0){
      isSky = min(isSky, texture(depthtex0, texcoord + vec2(i, j) * pixel * LightShaft_Quality).x);
    }
  }

  isSky = step(0.9999, isSky);
  */
  vec4 volumetric = vec4(vec3(0.0f), 1.0f);

  vec2 t = RaySphereIntersection(vec3(0.0, rE, 0.0), normalize(wP.xyz), vec3(0.0), rA);
  if(t.x > t.y) t = vec2(t.y, t.x);

  vec3 direction = wP.xyz * mix(1.0, t.y, isSky);

  //if(bool(1.0 - isSky))
  CalculateNearVolumetric(volumetric, wP, lightingDirection, dither);
  //CalculateFarVolumetric(volumetric, vec3(0.0, (cameraPosition.y - 63.0) * 0.0, 0.0), normalize(wP.xyz), lightingDirection, length(direction), dither);

  //volumetric.rgb = InScattering(vec3(0.0, cameraPosition.y, 0.0), normalize(wP.xyz), lightingDirection, 1500.0, length(vP.xyz) * 100.0, 0.76, dot(lightingDirection, normalize(wP.xyz)));
  //volumetric.rgb *= sunLightingColorRaw;

  /*
  vec4 rays = CalculateRays(wP, dither, isSky);

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  vec3 rayDirection = normalize(wP.xyz);

  vec3 rayStart = cameraPosition;
  vec3 rayEnd = rayDirection * 64.0;
  float stepLength = 64.0 * invsteps;
  vec3 rayStep = rayDirection * stepLength;

  vec3 testPoint = rayStart + rayStep - rayStep * dither;

  vec4 ray = rays;
  rays = vec4(0.0);
  int count = 0;

  float averageDepth = 0.0;

  vec3 br = vec3(0.14, 0.18, 0.2);
  vec3 bm = vec3(1.1);

  vec3 scatteredLight = vec3(1.0);

  vec2 opticalDepth = vec2(0.0);

  vec3 r = vec3(0.0);
  vec3 m = vec3(0.0);

  for(int i = 0; i < steps; i++){
    vec3 x = testPoint - rayStart;
    //float aTox = min(length(viewLength), length(x));

    if(length(x) > length(wP.xyz)) continue;

    float shape = 0.0;

    vec3 p = testPoint * 0.0156;

    shape += abs(noise(p.xz)-0.5);
    shape += abs(noise(p.xz * 2.0)-0.5) * 0.5;
    shape += abs(noise(p.xz * 4.0)-0.5) * 0.25;
    shape += abs(noise(p.xz * 8.0)-0.5) * 0.125;
    shape /= 1.875;

    float n = abs(noise(testPoint * 0.5) - 0.5);
          //n *= 0.5;
          //n += abs(noise(testPoint * 2.0 * 0.5) - 0.5) * 0.25;
          //n /= 0.75;

    float clouds = shape + n * 0.4;
          clouds = rescale(0.05, 1.0, clouds);
          //clouds = rescale(0.2, 1.0, clouds);

    float height = (testPoint.y - 63.05) + 4.0;

    vec2 h2 = vec2(stepLength) * 0.0001;
    //vec2 h2 = vec2(exp(-height / 3.0), exp(-height / 2.0)) * stepLength;
    opticalDepth += h2;

    vec3 tau = br * opticalDepth.x + bm * opticalDepth.y;
    vec3 attenuation = exp(-tau);

    vec4 visibility = vec4(1.0);
    if(!isSky) visibility = CalculateSunVisibility(wP, x, 1.0);
    attenuation *= visibility.a;

    r += attenuation * h2.x;
    m += attenuation * h2.y;

    testPoint += rayStep;
  }


  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  float mu = dot(sP, rayDirection);

  float phase = HG(dot(sP, rayDirection), 0.8);
  float phaseR = .0596831 * (1.0 + mu*mu);

  vec3 backscatter = rescale(vec3(0.0), vec3(1.0), bm) * invPi;

  rays.rgb = sunLightingColorRaw * (r * br + m * bm * max(vec3(phase), backscatter * 1.0)) * 20.0;// * ray.rgb * ray.a;
  */
  //rays.rgb = rays.rgb * br * sunLightingColorRaw * 20.0;// * 4.0 * sunLightingColorRaw * ray.rgb * ray.a * max(vec3(phase), backscatter * 100.0);
  //rays.rgb *= invsteps;

  //rays.rgb = rays.rgb * sunLightingColorRaw * br * 10.0;

  //if(count > 0)
  //rays.a /= float(count);
  //rays.a = min(rays.a, 1.0);
  //rays.a *= invsteps * Pi;
  //rays.rgb /= max(1.0, float(count));

  //rays.rgb = vec3(HG(dot(rayDirection, sP), 0.8));


  //rays.rgb *= sunLightingColorRaw;

  //float m = dot(lightDirection, normalize(vP.xyz));
  //rays.a *= HG(m, 0.2);

/* DRAWBUFFERS:4 */
  gl_FragData[0] = volumetric;
}

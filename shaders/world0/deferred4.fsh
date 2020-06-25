#version 130

//#extension GL_EXT_gpu_shader4 : require
//#extension GL_EXT_gpu_shader5 : require

//#define Enabled_SSAO

#define Enabled_ScreenSpace_Shadow
//#define Fast_Normal

#define AtmosphericScattering_Steps 8
#define AtmosphericScattering_Stepss 8

const int noiseTextureResolution = 64;

#define gcolor colortex0
#define gdepth colortex1
#define gnormal colortex2
#define composite colortex3
#define gaux1 colortex4
#define gaux2 colortex5

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D noisetex;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform float frameTimeCounter;
uniform float rainStrength;
uniform float viewWidth;
uniform float viewHeight;
uniform float nightVision;
uniform float aspectRatio;

uniform vec3 sunPosition;
uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

uniform int isEyeInWater;
uniform int heldBlockLightValue;
uniform int heldBlockLightValue2;
uniform int frameCounter;

uniform ivec2 eyeBrightness;
uniform ivec2 eyeBrightnessSmooth;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowProjectionInverse;
uniform mat4 shadowModelViewInverse;

uniform sampler2D depthtex2;
uniform int moonPhase;
uniform int worldTime;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColor;
in vec3 skyLightingColor;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / vec2(viewWidth, viewHeight);
#define Gaussian_Blur
#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"

#define CalculateHightLight 1
#define CalculateShadingColor 1

#include "../libs/brdf.glsl"
#include "../libs/light.glsl"
#include "../libs/atmospheric.glsl"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

float LinearlizeDepth(float depth) {
    return (far * (depth - near)) / (depth * (far - near));
}

float GetDepth(in vec2 coord){
  float depth = texture2D(depthtex0, coord).x;
  float depthParticle = texture2D(gaux1, coord).w;

  if(0.0 < depthParticle && depthParticle < depth) depth = depthParticle;
  return depth;
}

vec4 GetViewPosition(in vec2 coord){
  float depth = GetDepth(coord);

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

vec4 GetViewPosition(in vec2 coord, in sampler2D depth){
  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, texture2D(depth, coord).x) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

vec4 GetViewPosition(in vec2 coord, in float depth){
  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

#ifdef Enabled_SSAO
float CalculateAOBlur(){
  vec2 fragCoord = floor(texcoord * resolution);
  float checker = mod(fragCoord.x * fragCoord.y, 2);

  vec2 coord = texcoord * 0.5;

  float ao = 0.0;

  if(checker < 0.5){
    ao += texture(gaux2, coord + vec2(pixel.x, 0.0)).r;
    ao += texture(gaux2, coord - vec2(pixel.x, 0.0)).r;
    ao += texture(gaux2, coord + vec2(pixel.y, 0.0)).r;
    ao += texture(gaux2, coord - vec2(pixel.y, 0.0)).r;
    ao *= 0.25;

    return ao;
  }

  return texture(gaux2, coord).r;
}
#endif
/*
vec4 textureChecker(in sampler2D sampler, in vec2 coord){
  float renderScale = 0.5;
  coord *= renderScale;

  vec2 fragCoord = floor(texcoord * resolution);
  float checker = mod(fragCoord.x * fragCoord.y, 2);

  if(checker < 0.5){
    vec4 sampleTop = texture2D(sampler, coord + vec2(0.0, pixel.y));
    vec4 sampleRight = texture2D(sampler, coord + vec2(pixel.x, 0.0));
    vec4 sampleLeft = texture2D(sampler, coord - vec2(pixel.x, 0.0));
    vec4 sampleBottom = texture2D(sampler, coord - vec2(0.0, pixel.y));

    return (sampleTop + sampleRight + sampleLeft + sampleBottom) * 0.25;
    //return vec4(0.0);
  }

  return texture2D(sampler, coord);
}
*/
/*
float RayMarchingShade(in vec3 view, in vec3 rayDirection){
  int steps = 32;
  float isteps = 1.0 / steps;

  rayDirection = -(rayDirection) - view * length(rayDirection);
  vec3 d = normalize(rayDirection) * isteps * length(view);

  vec3 testPoint = view - d * 0.5;
  float shade = 1.0;

  for(int i = 0; i < steps; i++){
    testPoint += d;

    vec3 samplePosition = nvec3(gbufferProjection * nvec4(testPoint)) * 0.5 + 0.5;
    if(floor(samplePosition.xy) != vec2(0.0)) break;
         samplePosition.z = GetDepth(samplePosition.xy);
         samplePosition = nvec3(gbufferProjectionInverse * nvec4(samplePosition * 2.0 - 1.0));

    if(samplePosition.z > testPoint.z && samplePosition.z < testPoint.z + 1.0625) shade = 0.0;
  }

  return shade;
}
*/

vec2 hash2(in vec2 p){
  return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

float voronoi(in vec2 x){
  vec2 n = floor(x);
  vec2 f = fract(x);

  float md = 8.0;

  for(int i = -1; i <= 1; i++) {
    for(int j = -1; j <= 1; j++) {
      vec2 g = vec2(float(i), float(j));
      vec2 o = hash2(n + g);

      vec2 r = g + o - f;
      float d = dot(r, r);

      md = min(d, md);
    }
  }

  return md;
}

float noise(in vec2 uv) {
  float n = texture2D(noisetex, uv / noiseTextureResolution).x;
  //return n*n*(3.0-2.0*n);
  return n;
}

float noise3D(in vec3 x){
  vec3 p = floor(x);
  vec3 f = fract(x);
  f = f*f*(3.0-2.0*f);

  vec2 uv = (p.xz+vec2(37.0,17.0)*p.y) + f.xz;
  vec2 rg = texture2D(noisetex, (uv + 0.5) / noiseTextureResolution).yx;

  return mix( rg.x, rg.y, f.y );
}

#if 0
float CloudMap(in vec3 p){
  float n = 0.0;

  //n = fbm1(p);
  //clouds = ;

  float t = frameTimeCounter * 0.004;

  p.x -= t;

  n += (p * 2.0) * 0.5; p.x -= n * 0.5; p.x -= t * 1.414;
  n += (p * 4.0) * 0.25; p.x -= n * 0.25; p.x -= t * 1.732;
  n += (p * 8.0) * 0.125; p.z += n * 0.125; p.x += t * 2.0;
  n += (p * 16.0) * 0.0625; p.z -= n * 0.0625; p.x += t * 2.236;

  //float worley = 1.0 - pwn(p.xz / noiseTextureResolution * 128.0);
  //n *= mix(worley, 1.0, n);

  n += (p * 64.0) / 64.0;
  n += (p * 128.0) / 128.0;

  /*
  float worley = pwn(p.xz / noiseTextureResolution * 32.0);
  float perlin = fbm1(p * 8.0);
  //n -= perlin * worley * (1.0 - n) * 0.33;

  //n += worley * 0.33;
  n += worley;
  n -= perlin * 0.9 * (1.0 - worley);
  */

  return n;
}
#endif
float HGPF(in float phase, in float g){
  float g2 = g*g;

  return 0.0795774 * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * phase, 1.5));
}

float GetCloudsLighting(in float phase, in float density){
  float g = 0.8;
  float g2 = g*g;

  float powereffect = max(1.0 - pow(phase, 1.0 / (density * 2.0)), 0.0);
  float beerslaw = 1.0 - pow(2.0, 1.0 / density);
  float HGF = 0.0795774 * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * density, 1.5));

  float beerpower = (1.0 - exp2(-density * 2.0));

  return HGF * beerpower;
}

float fbm(in vec3 p){
  float noise = 0.0;

  for(float i = 0.0; i < 4.0; i += 1.0){
    float r = exp2(i + 1.0);
    float weight = 1.0 / r;

    noise += texture(noisetex, p.xz * r / noiseTextureResolution).x * weight;
  }

  return noise;
}

float fbm3D(in vec3 p){
  float noise = 0.0;

  for(float i = 0.0; i < 4.0; i += 1.0){
    float r = exp2(i + 1.0);
    float weight = 1.0 / r;

    noise += noise3D(p * r) * weight;
  }

  return noise;
}

float remap(in float clouds, float original_min, float original_max, float new_min, float new_max){
  return new_min + (clouds-original_min) / (original_max-original_min) * (new_max - new_min);
}

float WorleyPerlinNoise(in vec3 p, float density){
  float clouds = 0.0;

  float worley = 1.0 - voronoi(p.xz);
  float perlin = fbm(p);

  clouds = remap(perlin, -worley, 1.0, density, 1.0);

  return max(0.0, clouds);
}

float CloudsLinging(in float mu){
  float g = 0.8;
  float g2 = g * g;

  float beerlaw = 1.0 - exp(-mu * 0.33);
  float powersugareffect = exp(-mu * 0.33 * 2.0);
  float hg = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * pow(mu, 0.02), 1.5));

  return pow(hg * beerlaw * powersugareffect * 2.0, 2.2);
}

void CalcCirrus(inout vec3 color, in vec3 wP, in vec3 P, in vec3 lP){
  float cloudsAltitude = 1500.0;
  float cloudsThickness = 300.0;
  float cloudsMiddle = cloudsAltitude + cloudsThickness * 0.5;

  float dither = R2sq(texcoord * resolution + jittering);
        dither = mix(dither, 1.0, 1.0);

  //color = vec3(0.0);

  vec3 position = vec3(0.0, rE, 0.0);
  vec3 direction = normalize(wP.xyz);

  float playerPosition = 0.0;

  float lighting = clamp01(pow(dot(normalize(lP), normalize(upPosition)), 0.5));

  vec3 nlP = normalize(mat3(gbufferModelViewInverse) * lP);
  lP = mat3(gbufferModelViewInverse) * lP;

  float g = 0.8;
  float g2 = g*g;
  //float mu = dot(direction, normalize(nlP+direction));
  //float HG = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu,1.5));

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  float i = float(steps) - 1.0;

  lP /= 1500.0;

  while(i > 0){

    //if(i != 4.0){
    //  i -= 1.0;
    //  continue;
    //}

    float layerHight = (i + 1.0) * invsteps * cloudsThickness;

    vec3 p = position + vec3(0.0, cloudsAltitude + layerHight, 0.0);
    float h = length(p) - rE - playerPosition;

    //if(playerPosition > cloudsAltitude && wP.y > 0.0) continue;
    //if(playerPosition < cloudsAltitude && wP.y < 0.0) continue;

    p = position + vec3(0.0, cloudsAltitude, 0.0);
    p += direction * (h) / direction.y;
    p /= 1500.0;

    float clouds = WorleyPerlinNoise(p, 0.0);

    float highfreqNoise = WorleyPerlinNoise(p * 16.0, 0.0) * 0.0625;
    clouds = remap(clouds, -(1.0 - highfreqNoise), 1.0, 0.0, 1.0);

    float noise = abs(fbm3D(p * 8.0) * 2.0 - 1.0) * 0.125;
    clouds = remap(clouds, noise * 0.5, 1.0, -1.5, 1.0);

    float layerThickness = abs(layerHight - cloudsThickness * 0.5) / cloudsThickness * 2.0;

    float density = -0.64;
          density -= 0.2 * pow2(layerThickness);

    float coverage = 2.0;

    clouds = clamp01((clouds + density) / (1.0 + density) * coverage);

    vec3 pS = p + lP / 1500.0;

    float shading = WorleyPerlinNoise(pS, 0.0);
    shading = remap(shading, 0.0, 1.0, 0.0, 1.0);

    float density2 = -0.2 + 0.19 * float(i);
    shading = clamp01((shading + density) / (1.0 + density) * coverage);
    shading = 1.0 - shading;
    shading *= lighting;

    vec3 cloudsColor = skyLightingColorRaw * sqrt(2.0);
         cloudsColor *= pow(exp(-clouds * 0.33), 2.2);

    vec3 sunlighting = sunLightingColorRaw * 30.0;
         sunlighting *= CloudsLinging(shading);

    cloudsColor += sunlighting;

    //cloudsColor = sunLightingColorRaw * pow(beerlaw * powersugareffect * HG * 2.0, 2.2);
    //cloudsColor += skyLightingColorRaw * 16.0 * pow(beerlaw * powersugareffect * 2.0, 2.2);

    clouds *= clamp01(direction.y * 3.0 - 0.2);
    //clouds = min(1.0, clouds * coverage);
    //clouds = step(0.01, clouds);

    //cloudsColor.r = vec3(shading * shading).r;

    color = mix(color, cloudsColor, pow(clouds, 2.2));
    i -= 1.0;
  }

  //color = vec3(shading);

  //color = mix(color, sum.rgb, sum.a);
}

#if 0
float GetCirrus(in vec3 p){
  float t = frameTimeCounter * 0.004;
//t = 0.0;
  p.x += t;

  t *= 2.0;

  //float as
  float distort = 3.14;

  //p.x /= length(p.xz);
  //p *= 2.0;
  p.x *= 0.7071;
  float clouds = 0.0;
  clouds += noise(p.xz * 2.0) * 0.5;

  float n2 = 0.0;
  n2 += 1.0 - noise(p.xz * 2.0 + vec2(16.0));
  n2 += 1.0 - noise(p.xz * 2.0 - vec2(16.0));
  n2 += 1.0 - noise(p.xz * 2.0 + vec2(16.0, -16.0));
  n2 += 1.0 - noise(p.xz * 2.0 - vec2(16.0, -16.0));
  n2 *= 0.25;

  clouds += n2 * 0.5;
  clouds *= 0.5;

  p.x += clouds * 0.25 * distort;    p.x += t * 0.5;

  clouds += noise(p.xz * 4.0) * 0.25;     p.xz += clouds * 0.125 * distort;   p.x += t * 0.25;
  clouds += noise(p.xz * 8.0) * 0.125;    p.xz += vec2(1.0, -1.0) * clouds * 0.0625 * distort;  p.x -= t * 0.125;
  clouds += noise(p.xz * 16.0) * 0.0625;  p.xz += vec2(1.0, -1.0) * clouds * 0.03125 * distort; p.x -= t * 0.25;
  clouds += noise(p.xz * 32.0) * 0.03125;

  p.x *= 1.414;
  p.x -= t * 0.25;

  float n = noise(p.xz * 64.0 * vec2(2.0, 0.5)) / 64.0; /*p.xz += n * 0.5;  p.z += t * 0.007;*/ p.x -= t * 0.3125;
  n += noise(p.xz * 128.0 * vec2(2.0, 0.5)) / 128.0;    //p.xz += n * 0.25; //p.z -= t * 0.004; //p.x += t * 0.01;
  n += noise(p.xz * 256.0 * vec2(2.0, 0.5)) / 256.0;

  clouds -= n * (1.0 - clouds) * 1.414;

  float coverage = 0.5;

  clouds = clamp01((clouds - coverage) / (1.0 - coverage) * 1.414);
  clouds = clouds * clouds * (3.0 - 2.0 * clouds);

  return clouds;
}

void (inout vec3 color, in vec3 wP, in vec3 P, in vec3 lP){
  float cloudsAltitude = 8000.0;

  float dither = bayer_16x16(texcoord, resolution);

  vec3 position = vec3(0.0, rE, 0.0);
  vec3 direction = normalize(wP.xyz);

  float playerPosition = (P.y) * 1.0;

  vec3 p = position + vec3(0.0, cloudsAltitude, 0.0);
  float h = length(p) - rE - playerPosition;

  //p = direction * (0.0 - cameraPosition.y - 63.0) / direction.y;
  //if(direction.y + wP.y < wP.y) return;
  //p /= 1.0;

  if(playerPosition > cloudsAltitude && wP.y > 0.0) return;
  if(playerPosition < cloudsAltitude && wP.y < 0.0) return;

  //if((wP.y + 80.0) < 80.0) return;
  p = position + vec3(0.0, cloudsAltitude, 0.0);
  p += direction * (h) / direction.y;
  //if(length(p.xz / cloudsAltitude) < length(direction.xz)) return;
  //p.xz += P.xz;
  p /= 8000.0;

       //p.x *= 0.7071;

  vec3 lightingVector = normalize(mat3(gbufferModelViewInverse)*lP);

  float clouds = GetCirrus(p);

  vec3 cloudsColor = vec3(1.0);

  int steps = 4;
  float invsteps = 1.0 / steps;

  //lightingVector /= max(0.2, lightingVector.y);
  vec3 rayDirection = lightingVector * invsteps * 0.2;

  float cloudsShading = 0.0;

  vec3 rayStart = p + rayDirection;

  //p -= lightingVector * dither;
  rayDirection *= dither;

  float prevShading = 0.0;
/*
  for(int i = 0; i < steps; i++){
    float currShading = GetCirrus(rayStart);
    rayStart += rayDirection;

    //cloudsShading = max(currShading, cloudsShading);
    //cloudsShading += currShading;
    cloudsShading = max(cloudsShading, currShading);
  }

  //cloudsShading *= invsteps;
  //cloudsShading = min(1.0, cloudsShading * Pi);*/
  cloudsShading = 1.0 - cloudsShading;

  //color = mix(color, vec3(1.0, vec2(cloudsShading)), clouds);
  //return;

  vec3 sunlight = sunLightingColorRaw;
  sunlight *= cloudsShading;

  float mu = dot(lightingVector, direction);

  float hg = HGPF(mu, 0.8);

  sunlight *= hg;

  vec3 skylight = skyLightingColorRaw*0.32;

  cloudsColor = sunlight + skylight;
  //cloudsColor *= sunLightingColorRaw;
  //cloudsColor *= GetCloudsLighting(sunphase, d) * Pi;

  float t2 = mix(1.0 - clouds, 1.0, 0.1) * 2.0;

  float beerpower = 1.0 - exp(-t2);
  cloudsColor *= beerpower;
  cloudsColor *= 30.0;
  cloudsColor *= mix(vec3(1.0), skyLightingColorRaw, 0.6811);
  //cloudsColor *= sqrt(sunLightingColorRaw);


  clouds = min(clouds * Pi, 1.0);
  clouds *= clamp01((direction.y - 0.07) * Pi * 2.0);
}
#endif
vec3 shadowMapGI(){


  return vec3(0.0);
  /*
  vec3 rsmBlur = vec3(0.0);

  vec2 fragCoord = floor(texcoord * resolution);
  float cb = step(mod(fragCoord.x * fragCoord.y, 2), 0.5);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      rsmBlur += rgb2L(texture2D(gaux2, texcoord*0.5+vec2(i, j)*pixel*1.0).rgb);
    }
  }

  rsmBlur *= 1.0/9.0;

  return rgb2L(texture2D(gaux2, texcoord*0.17685185).rgb);
  */
}

vec3 invKarisToneMapping(in vec3 color){
	//https://graphicrants.blogspot.com/2013/12/tone-mapping.html
	float a = 0.001;
	float b = 0.01;
	float c = 1.0;

	color *= c;
	float luma = maxComponent(color);
	color = color/luma*((a*a-(2.0*a-b)*luma)/(b-luma));

	return max(vec3(0.0), color);
}

vec3 ClosestSmooth(in sampler2D sampler, in vec2 coord){
  vec3 result = vec3(0.0);
  float depth = linearizeDepth(texture(depthtex0, coord).x);

  int count = 1;

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 samplePos = vec2(i, j) * pixel;
      vec3 sampleColor = texture2D(sampler, coord + samplePos).rgb;
      float sampleDepth = linearizeDepth(texture(depthtex0, coord + samplePos).x);

      if(abs(sampleDepth - depth) > 0.00003) continue;
      count++;
      result += sampleColor;
    }
  }

  result += texture2D(sampler, coord).rgb;
  result /= count;

  return result;
}

vec4 LowDetailSampler0p5x(in sampler2D sampler, in vec2 coord, int decode){
  vec4 tex;
  int count;

  int decodenormal = 1;
  int linearDepth = 2;

  for(int i = 0; i <= 2; i++){
    for(int j = 0; j <= 2; j++){
      vec2 offset = vec2(i, j);
      vec2 coordoffset = floor((texcoord * resolution + offset + 1.0));
      if(mod(coordoffset.x * coordoffset.y, 2) < 0.5) continue;

      vec4 sampleTex = texture2D(sampler, offset * pixel + texcoord);
      if(decode == decodenormal) sampleTex.rgb = normalDecode(sampleTex.rg);
      if(decode == linearDepth) sampleTex = vec4(linearizeDepth(sampleTex.x));

      tex += sampleTex;

      //tex += texture2D(sampler, offset * pixel + texcoord).rgb;
      count++;
    }
  }

  return tex / float(count);
}

vec3 UpSampleRSM(in vec2 coord, in vec3 normal1x, in float depth1x){
  vec3 tex = vec3(0.0);

  coord *= 0.353553;
  //coord = floor(coord * resolution) * pixel;

  depth1x = linearizeDepth(depth1x);

  //vec3 normalLD = LowDetailSampler0p5x(composite, texcoord, 1).xyz;
  //float depthLD = LowDetailSampler0p5x(depthtex0, texcoord, 2).x;

  int count = 0;
  float weights = 0.0;

  coord *= resolution;

  for(int i = 0; i <= 6; i++){
    for(int j = 0; j <= 6; j++){
      vec2 offset = (vec2(i, j) - 3.0);
      float weight = 1.0;

      vec2 coordoffset = round(coord + offset) * pixel;
      if(coordoffset.x > 0.5 - pixel.x || coordoffset.y > 0.5 - pixel.y) continue;

      vec3 sampleColor = texture2D(gaux2, coordoffset).rgb;
      float sampleDepth = linearizeDepth(texture(gaux2, coordoffset + vec2(0.5)).x);
      vec3 sampleNormal = texture2D(gaux2, coordoffset + vec2(0.5, 0.0)).rgb * 2.0 - 1.0;

      float x = (depth1x - sampleDepth) * 512.0;
      weight *= exp(-(x*x) / (2.0 * 0.64));

      x = max(0.0, pow5(dot(normal1x, sampleNormal)));
      weight *= x;
      //weight = 1.0;

      tex += sampleColor * weight;
      weights += weight;
      count++;
    }
  }

  //tex /= float(count);
  if(weights > 0.0) tex /= weights;

  //tex = texture2D(gaux2, texcoord * 0.5 + vec2(0.0, 0.5)).ggg;

  return tex;
}

float UpSampleAO(in vec2 coord, in vec3 normal){
  float tex = 0.0;

  //coord = floor(coord * 0.5 * resolution) * pixel;
  coord = round(coord * 0.5 * resolution);

  float weights = 0.0;

  float depth = linearizeDepth(texture(depthtex0, texcoord).x);

  for(int i = 0; i <= 6; i++) {
    for(int j = 0; j <= 6; j++) {
      vec2 offset = (vec2(i, j) - 3.0);

      vec2 position = (offset + coord);
           position = position * pixel + vec2(0.0, 0.5);

      float sampleDepth = linearizeDepth(texture(gaux2, position).y);
      float x = (depth - sampleDepth) * 1024.0;
      float weight = exp(-(x*x) / (2.0 * 0.64));
            //weight = 1.0;

      float sampler = texture(gaux2, position).x;
      tex += sampler * weight;
      weights += weight;
    }
  }

  if(weights > 0.0) tex /= weights;
  else tex = 1.0;

  //tex = texture(gaux2, texcoord * 0.5 + vec2(0.0, 0.5)).x;

  return tex;
}

float NormalizedDiffusion(in float r, in float d){
  return (exp(-r/d) * exp(-r*0.333/d)) / (8.0*Pi*d*r);
}

void main() {
  vec2 jitteringCoord = texcoord - jittering * pixel;
  #if MC_VERSION < 11202
    jitteringCoord = clamp(jitteringCoord, pixel, 1.0 - pixel);
  #endif

  vec4 albedo = texture2D(gcolor, texcoord);

  vec4 lightingData = texture2D(gdepth, texcoord);
  float blocksLightingMap = lightingData.r;
  float skyLightingMap = clamp01((lightingData.g - 0.03125) * 1.0723);
        skyLightingMap = pow3(skyLightingMap);
  float emissive = lightingData.a;
  float selfShadow = texture2D(gnormal, texcoord).a;
  #if MC_VERSION > 11404
    selfShadow = 1.0;
    emissive = 0.0;
  #endif

  vec3 normalVisible = normalDecode(texture2D(gnormal, texcoord).xy);
  vec3 normalSurface = normalDecode(texture2D(composite, texcoord).xy);
  vec3 blockNormal = normalVisible;

  float smoothness = texture2D(gnormal, texcoord).b;
  float metallic = texture2D(composite, texcoord).b;
  float roughness = 1.0 - smoothness; roughness *= roughness;
  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float mask = round(lightingData.z * 255.0);
  bool isSky = bool(step(254.5, mask));
  bool isParticels = bool(step(249.5, mask) * step(mask, 250.5));
  bool emissiveParticels = bool(step(250.5, mask) * step(mask, 252.5));
  bool isLeaves = CalculateMaskID(18.0, mask);
  bool isGrass = CalculateMaskID(31.0, mask);
  bool isWool = CalculateMaskID(35.0, mask);

  float depth = texture2D(depthtex0, texcoord).x;
  float depthParticle = texture2D(gaux2, texcoord).a;
  if(0.0 < depthParticle && depthParticle < depth) depth = depthParticle;

  vec4 vP = GetViewPosition(texcoord, depth);
  vec4 wP = gbufferModelViewInverse * vP;
  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);
  vec3 nvP = normalize(vP.xyz);
  vec3 reflectP = normalize(reflect(nvP, normalVisible));

  #ifdef Enabled_TAA
    vec4 vPJ = GetViewPosition(jitteringCoord, texture(depthtex0, jitteringCoord).x);
    vec4 wPJ = gbufferModelViewInverse * vPJ;
  #else
    vec4 vPJ = vP;
    vec4 wPJ = wP;
  #endif

  float ndotv = dot(nvP, normalSurface);
  if(-0.15 > ndotv) normalVisible = normalSurface;

  if(isParticels) {
    vec3 normal = nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.7) * 2.0 - 1.0));
         normal = normalize(-normal);

    normalVisible = normal;
    normalSurface = normal;
  }

  float viewLength = length(vP.xyz);

  vec3 centerupPosition = normalize(upPosition);
  vec3 eastPosition = normalize(mat3(gbufferModelView) * vec3(1.0, 0.0, 0.0));
  vec3 southPosition = normalize(mat3(gbufferModelView) * vec3(0.0, 0.0, 1.0));
  vec3 nsunPosition = normalize(sunPosition);
  vec3 nshadowLightPosition = normalize(shadowLightPosition);
  vec3 centerviewPosition = GetViewPosition(vec2(0.5), 0.9).xyz;

  vec4 vPjittering = GetViewPosition(jitteringCoord);

  vec3 color = vec3(0.0);

  vec3 albedoL = albedo.rgb;
  albedo.rgb = rgb2L(albedo.rgb);

  vec3 lightPosition = sunPosition;
  if(sP.y < -0.1) lightPosition = -lightPosition;
  vec3 nlightPosition = normalize(lightPosition);

  vec3 torchLightingColor = vec3(1.049, 0.5821, 0.0955);

  if(!isSky){
    float ao = UpSampleAO(texcoord, normalSurface);
          ao = L2Gamma(ao).x;

    #ifdef Enabled_SSAO
      //ao = CalculateAOBlur();
      //ao = pow(ao, 2.2);
    #endif

    float translucentShadowMaps = 0.0;
    float shadowMapIrrdiance = 0.0;

    //vec3 sss = vec3(1.0);
    //float sssRadius = NormalizedDiffusion(length(normalize(shadowLightPosition) - normalize(vP.xyz)), 1.0);

    vec4 sunShading = CalculateShading(shadowtex1, shadowtex0, wP, selfShadow, normalSurface, translucentShadowMaps, shadowMapIrrdiance);
    vec3 shading = mix(vec3(1.0), sunShading.rgb, sunShading.a) * selfShadow;

    //vec3 woolScattering = exp(-sss.x * 0.25) * albedoL;
         //woolScattering = L2Gamma(woolScattering);
    /*
    sss += sqrt(sss * sss + pow2(1.0 / sssRadius) * 0.01);
    sss = exp(-sss * (1.0 + maxComponent(albedo.rgb) * 7.0));
    //sss *= sssRadius;
    sss *= (albedoL);
    sss = L2Gamma(sss);
    */

    #ifdef Enabled_ScreenSpace_Shadow
      float sdfShadow = ScreenSpaceShadow(shadowLightPosition, vP.xyz, blockNormal);
      //shading = vec3(1.0);
      shading *= sdfShadow;
    #endif

    vec3 sunLighting = BRDF(albedoL, nlightPosition, -nvP, normalVisible, normalSurface, roughness, metallic, F0);
         sunLighting *= shading * fading;
         sunLighting = rgb2L(sunLighting) * sunLightingColorRaw;
    if(isEyeInWater != 1) sunLighting *= smoothstep(0.7333, 0.8, max(float(eyeBrightness.y) / 60.0, lightingData.y));
         //sunLighting = sunLighting * vec3(0.008, 0.02, 0.09);
    color = sunLighting * SunLight;

    //vec3 heldLighting = BRDF(albedoL, -nvP, -nvP, normalVisible, normalSurface, roughness, metallic, rgb2L(F0));

    vec3 diffuse = vec3(0.0);

    vec3 m = normalize(reflectP - nvP);

    float vdoth = pow5(1.0 - clamp01(dot(-nvP, m)));
    float ndotm = clamp01(dot(m, normalVisible));
    float ndotu = dot(normalSurface, centerupPosition);
    float ndotl = dot(normalSurface, nlightPosition);

    vec3 f = F(F0, vdoth);
    float brdf = min(1.0, CalculateBRDF(-nvP, reflectP, normalVisible, roughness));
    float d = DistributionTerm(roughness * roughness, ndotm);
    float nd = NormalizedDiffusion(length(nlightPosition - nvP), d);

    float FD90 = clamp01(0.5 + 2.0 * roughness * ndotm * ndotm);
    float FdV = 1.0 + (FD90 - 1.0) * vdoth;
    float FdL = 1.0 + (FD90 - 1.0) * pow5(clamp01(ndotl));

    vec3 skyLighting = albedo.rgb * (skyLightingColorRaw) * max(0.0, ndotu * 0.5 + 0.5);

    float indirectLightDown = clamp01(dot(normalize(reflect(nshadowLightPosition, centerupPosition)), normalSurface));
    float indirectLightBack = clamp01(dot(normalize(reflect(nshadowLightPosition, -nshadowLightPosition)), normalSurface));
    float NdotL = dot(centerupPosition, normalSurface);
    vec3 fakeGI = albedo.rgb * (skyLightingColorRaw * sunLightingColorRaw) * skyLightingMap * 2.0;//mix(skyLightingColorRaw * 2.0 * Pi, sunLightingColorRaw, clamp01(0.7 + NdotL))

    float ldotu = max(0.0, dot(nshadowLightPosition, centerupPosition));
    fakeGI *= (indirectLightDown + indirectLightBack) * 0.5 * (pow2(ldotu));

    diffuse = skyLighting * 0.54 * 1.2275;
    diffuse += fakeGI;
    diffuse *= ao * skyLightingMap;
    //diffuse += albedo.rgb * torchLightingColor * heldLightingGIFading * ao;

    //vec3 gi = texture2D(gaux2, texcoord*0.5).rgb;
    //gi = rgb2L((gi));
    //gi = gi * albedo.rgb * sunLightingColorRaw * SunLight;

    //diffuse += gi * 0.168;

    diffuse *= 0.7;

    //diffuse = albedo.rgb * (skyLightingColorRaw);// * (sqrt(max(0.0, ndotu)) * pow2(skyLightingMap) + abs(ndotl) * 0.33 * step(-0.1, ndotu));
    //diffuse *= skyLightingMap;
    //diffuse *= ao;

    vec3 ambient = albedo.rgb * skyLightingColorRaw * (ao + ndotu + 1.0) * 0.333;
    //vec3 ambient = albedo.rgb * skyLightingColorRaw * (ao + ndotu * 0.5 + 0.5) / 1.5;
    vec3 bounce = albedo.rgb * skyLightingColorRaw * ao * clamp01(-ndotu) * pow2(ldotu) * skyLightingMap;

    diffuse = (ambient) * skyLightingMap;

    vec3 indirect = UpSampleRSM(texcoord, normalSurface, depth).rgb * (1.0 - metallic);
    //indirect = sqrt(indirect * getLum(indirect));
    indirect = L2Gamma(indirect * invPi);

    //indirect *= normalize(albedo.rgb) * sqrt(getLum(albedo.rgb));
    diffuse += indirect * 4.0 * albedo.rgb * sunLightingColorRaw * SunLight * fading;

    translucentShadowMaps = exp(-translucentShadowMaps * 0.33 / d);

    vec3 transmittance = albedoL * translucentShadowMaps;
         transmittance *= float(isLeaves || isGrass || isWool);
         transmittance *= FD90 * FdV * FdL * invPi * max(0.0, 1.0 + -ndotl) * 0.5 / max(1.0, d);
         transmittance = L2Gamma(transmittance);

    diffuse += transmittance * SunLight * fading * sunLightingColorRaw;

    diffuse *= rgb2L(1.0 - max(f, metallic) * brdf);

    color += diffuse;

    vec3 Fin = F(F0, clamp01(pow5(ndotl)));

    //color += sss * sunLightingColorRaw * f * Fin * invPi * SunLight * (1.0 + 2499.0 * step(metallic, 0.5));
    //color += woolScattering * SunLight * sunLightingColorRaw * clamp01(-ndotl + 0.5) * invPi * float(isWool || isLeaves);
    //color = sss;

    float lightingRadius = max(heldBlockLightValue, heldBlockLightValue2);
    //float heldLightingFading = max((lightingDistance * 0.5 - viewLength) * float(heldBlockLightValue) * 0.0074, 0.0);
    //      heldLightingFading *= heldLightingFading;

    vec3 heldLightingBRDF = BRDF(albedoL, -nvP, -nvP, normalVisible, normalSurface, roughness, metallic, rgb2L(F0));
         heldLightingBRDF = L2Gamma(heldLightingBRDF);

    //vec3 heldLighting = 1.0 - exp(-(torchLightingColor) * 0.125 * heldLightingFading);
         //heldLighting = heldLighting * heldLightingBRDF * 8.0;

    float heldLightingDistance = max(0.0, lightingRadius - viewLength) / lightingRadius;
          heldLightingDistance = pow3(heldLightingDistance);
    vec3 heldLighting = clamp01(1.0 - exp(-torchLightingColor * heldLightingDistance * 0.125));
         heldLighting *= heldLightingBRDF * 8.0;

    blocksLightingMap = max(blocksLightingMap - 0.06667,0.0) * 1.07143;
    blocksLightingMap -= blocksLightingMap * step(0.5, metallic) * brdf;
    blocksLightingMap = blocksLightingMap / (1.0+blocksLightingMap) * 2.0;
    blocksLightingMap = pow3(blocksLightingMap);

    vec3 lightMapLighting = clamp01(1.0 - exp(-(torchLightingColor) * 0.125 * blocksLightingMap));
         //torchLignting *= pow5(lightingmu.x * 0.5 + lightingmu.y * 0.5);
         lightMapLighting *= albedo.rgb;

    vec3 torchLignting = (lightMapLighting * 0.0 + heldLighting);
    torchLignting *= exp(-maxComponent(sunLightingColorRaw + skyLightingColorRaw) * Pi * skyLightingMap);

    color += (torchLignting);

    //vec3 reflectionSky = CalculateSky(reflectP, sP, 0.0, 1.0);
    //color += rgb2L(brdf * brdf * f) * reflectionSky;

    if(emissiveParticels) color *= 1.0 - emissive;
    color += emissive * albedo.rgb * 4.56;

    //color = vec3((nd));

    //color = albedo.rgb;
    //color = gi * 0.5;
    //color = UpSampleGI(gaux2, texcoord, normalSurface, depth).rgb;
    //if(texcoord.x > 0.5) color = texture2D(gcolor, texcoord).rgb;
    //color = texture2D(gaux2, texcoord).rgb * 0.1;
    //color = L2Gamma(color) * 0.5;

    albedo.a = 1.0;
    //color = sss;
    //color = vec3(ao) * 0.01;
    //color = albedo.rgb / maxComponent(albedo.rgb) * getLum(albedo.rgb) * (pow2(1.0 - maxComponent(albedo.rgb)) * 0.99 + 0.01);
    //color = vec3(dot(direction + rayStart))
    //color = gi * 1.0;
    //color = indirect;// * getLum(albedo.rgb) * 1.0;// * invPi * invPi;

    //vec3 direction = vec3(cos(2.0 * Pi * 0.5), sin(2.0 * Pi * 0.5), 0.0);
    //vec3 worldNormal = mat3(gbufferModelViewInverse) * normalSurface;
    //color = vec3(dot(direction, worldNormal));

    //color = vec3(ComputeCoarseAO(vP.xyz, normalSurface)) * 0.1;

  }else{
    albedo.a = 0.0;

    vec3 atmoScattering = CalculateSky(normalize(vP.xyz), sP, 0.0, 1.0);

    vec3 skyColor = atmoScattering;

    float top = clamp01(wP.y / length(wP.xyz) * 100.0);
    float sundotv = clamp01(dot(nsunPosition, nvP));
    float moondotv = clamp01(dot(-nsunPosition, nvP));

    skyColor += clamp01((sundotv - 0.999) * 1000.0) * overRange * sunLightingColorRaw * top;
    //skyColor = mix(skyColor, min(sunLightingColorRaw * overRange, vec3(pow(overRange, 1.0 / 2.2))), top * max(0.0, sundotv - 0.999) * 1000.0);

    float moon = HGPF(moondotv, 0.96) * 0.00006;
          //moon += clamp01(moonLight - 0.9995) * 2000.0;
    skyColor += moon * vec3(1.022, 0.782, 0.344) * top;

    vec2 moonAtlasScale = vec2(textureSize(depthtex2, 0));
    float moonAspectRatio = 1.0 / (moonAtlasScale.x / moonAtlasScale.y);

    vec3 worldSkyPosition = mat3(gbufferModelViewInverse) * normalize(vP.xyz);

    vec2 moonPhases = vec2(0.25, 0.25);
         moonPhases.x += 0.5 * mod(moonPhase, 4);
         if(moonPhase > 3) moonPhases.y += 0.5;
         moonPhases.x *= moonAspectRatio;

    vec2 moonDrawPosition = worldSkyPosition.xy;
         moonDrawPosition /= 1.0 + worldSkyPosition.z;
         moonDrawPosition = (sP.xy / (1.0 - sP.z) + moonDrawPosition);
         moonDrawPosition *= 1.0 + worldSkyPosition.z;
         moonDrawPosition.x *= moonAspectRatio;
         moonDrawPosition *= 4.0;

    vec3 moonTexture = texture2D(depthtex2, moonDrawPosition + moonPhases).rgb;
         moonTexture = rgb2L(moonTexture) * 0.4;

    if(floor(moonDrawPosition * vec2(4.0, 2.0) + 0.5) != vec2(0.0) || worldSkyPosition.z < -0.5) moonTexture = vec3(0.0);

    skyColor += vec3(1.022, 0.782, 0.344) * dot03(moonTexture) * top;

    float r = frameTimeCounter * 0.000277;
          r = -r;
          //r *= 4.167;

    //mat2 rotate = mat2(cos(r), -sin(r), sin(r), cos(r));

    vec3 seed = mat3(gbufferModelViewInverse) * normalize(vP.xyz);
         seed.xz /= sqrt(seed.y + 1.0);
         seed.xz -= r * 1.0;
         //seed.xyz -= sP.xyz * 0.0277;
         //seed.xz *= rotate;
         seed.xz *= 512.0;
         seed.xz = floor(seed.xz);

    float stars = clamp01(hash(seed.xz) - 0.999) * 1000.0;
          stars *= (max(0.0, worldSkyPosition.y));

    skyColor += max(0.0, stars - maxComponent(skyColor) * 324.0) * top * 6.0;

    //color = mix(skyColor, color, albedo.a);
    color = skyColor;

    //color = vec3(HGPF(moondotv, 0.96) * 0.0016);

    //CalcCirrus(color, wPJ.xyz, cameraPosition, lightPosition);
    //color =
  }

  //shadowMapColor = clamp01(shadowMapColor);
//shadowMapColor = texture2D(gaux2, (texcoord + vec2(1.0,0.0))*0.17665).rgb*0.1;
  //color += rgb2L(shadowMapColor) * sunLightingColorRaw * albedo.rgb * SunLight;

  //
  //color = rgb2L(texture2D(shadowcolor0, ))
  //if(floor(texcoord*1.1) == vec2(0.0))
  //color = rgb2L((texture2D(gaux2, texcoord*0.5).rgb));
  //color = mat3(gbufferModelViewInverse) * normalDecode(texture2D(shadowcolor0, texcoord).yz) * 0.5 + 0.5;
  //

  //color = AtmosphericScattering(vec3(0.0), vec3(texcoord.x * 2.0 - 1.0, 1.0 - length(texcoord*2.0-1.0), texcoord.y * 2.0 - 1.0), vec3(0.0, 0.0, 0.0), 1.0);

  //color = texture2D(gaux2, texcoord*0.5).rgb*albedo.rgb*sunLightingColorRaw;

  //shadowMapColor *= albedo.rgb * sunLightingColorRaw;
  //color = shadowMapColor;

       //rsm.xy = rsm.xy * 0.5 + 0.5;
  //color = texture2D(shadowcolor1, rsm.xy).rgb * 0.01 * shadow;

  //if(skyLightingMap < 0.001) color += vec3(1.0, 0.0, 0.0);

  color = L2rgb(color);

  //if(texcoord.x < 0.1 && isEyeInWater == 1){
  //  color = vec3(float(eyeBrightness.y) / 240.0);
  //}

  //if(maxComponent(color) > 1.0) color = vec3(0.0);

  //color = vec3(dot(vec3(1.0, 0.0, 0.0), texture2D(shadowcolor0, texcoord).xyz * 2.0 - 1.0));
  /*
  float bias = 0.0;
  vec3 shP = wP2sP(wP, bias);

  float d = texture(shadowtex0, shP.xy).x + 0.04;
  float z = shP.z + 0.1;
  color = vec3((z - d) / d) * 1.0;
  //if(color.z > 1.0) color = vec3(1.0, 0.0, 0.0);
  //if(color.z < 0.0) color = vec3(0.0, 0.0, 1.0);
  */
/*
  color = mat3(shadowModelViewInverse) * normalDecode(texture2D(shadowcolor0, shP.xy).gb);
  color = vec3(dot(color, vec3(0.0, 1.0, 0.0)));
  //if(color.g > 0.5) color.b -= color.g*2.0;
  //color.b -= color.r;
  //color.b -= color.r + color.g;
  //if(color.r > 0.0 || color.g > 0.0) color.b -= max(color.r, color.g);
  //if(color.r < 0.0 || color.g < 0.0) color.b -= min(color.r, color.g);
*/
  color /= overRange;

  //color = mat3(gbufferModelViewInverse) * normalVisible;

/* DRAWBUFFERS:05 */
  gl_FragData[0] = vec4(albedo.rgb, 0.0);
  //gl_FragData[1] = vec4(smoothness, metallic, 0.0, 1.0);
  gl_FragData[1] = vec4(color, 1.0);
}
                                                                                           

#version 130

#define LightShaft_Quality 0.5

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D depthtex2;

uniform sampler2D noisetex;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform mat4 shadowProjectionInverse;
uniform mat4 shadowModelViewInverse;

uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;
uniform vec3 sunPosition;
uniform vec3 upPosition;

uniform vec2 jitter;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform float biomeTemperature;
uniform float biomeRainFall;

uniform int frameCounter;
uniform int isEyeInWater;
uniform int heldBlockLightValue;
uniform int heldBlockLightValue2;

uniform ivec2 eyeBrightness;
uniform ivec2 eyeBrightnessSmooth;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

in vec4 eyesWaterColor;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);

const int noiseTextureResolution = 64;

#define Gaussian_Blur
#define CalculateHightLight 1

#include "../libs/common.inc"
#include "../lib/packing.glsl"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/brdf.glsl"
#include "../libs/light.glsl"
#include "../libs/atmospheric.glsl"

float HG(in float m, in float g){
  return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

float NormalizedDiffusion(in float r, in float d){
  return min(1.0, (exp(-r/d) * exp(-r*0.333/d)) / (8.0*Pi*d*r));
}

bool totalInternalReflection(in vec3 i, inout vec3 o, in vec3 n, in float eta){
  float cosi = dot(i, n);
  float cost2 = pow2(cosi);

  float TIR = 1.0 - eta * eta * (1.0 - pow2(cosi));

  o = normalize(refract(-i, n, eta));
  if(TIR < 0.0) o = vec3(0.0);

  return bool(step(TIR, 0.0));
}

vec4 CalculateSunDirectLighting(in vec3 viewPosition, in vec3 normal){
    float viewLength = length(viewPosition);
    if(viewLength > shadowDistance) return vec4(1.0);

    vec4 shadowPosition = gbufferModelViewInverse * vec4(viewPosition, 1.0);
         shadowPosition = shadowModelView * shadowPosition;
         shadowPosition = shadowProjection * shadowPosition;
         shadowPosition /= shadowPosition.w;
         shadowPosition = shadowPosition;

    float distort = 1.0 / (mix(1.0, length(shadowPosition.xy), SHADOW_MAP_BIAS) / 0.95);
    vec3 shadowCoord = shadowPosition.xyz * vec3(vec2(distort), 1.0) * 0.5 + 0.5;

    shadowCoord.xy *= 0.8;

    float d = texture(shadowtex1, shadowCoord.xy).x;
    float d0 = texture(shadowtex0, shadowCoord.xy).x;

    float shading = step(shadowCoord.z, d + shadowPixel * 2.0);

    return vec4(shading);
}

void CalculateSSS(inout vec3 color, in vec3 rayOrigin, in vec3 rayDirection, in vec4 albedo, in vec3 normal, in float inv, in float sss){
  /*
  int steps = 16;
  float invsteps = 1.0 / float(steps);

  float dither = R2sq(texcoord * resolution - jittering);

  rayDirection *= invsteps;
  vec3 rayStart = rayOrigin + rayDirection * dither;

  vec3 scattering = vec3(0.0);
  vec3 attenuation = vec3(1.0);


  float dist = 0.0;

  for(int i = 0; i < 16; ++i){
    vec3 test = rayStart + rayDirection * float(i);
    vec3 coord = nvec3(gbufferProjection * nvec4(test)) * 0.5 + 0.5;

    float d = linearizeDepth(texture(depthtex0, coord.xy).x);
    float z = linearizeDepth(coord.z);
    float dif = (d - z) * -inv;

    if(dif * inv < -0.0001 || dif > 0.1) continue;
      scattering += 1.0;
  }

  scattering = 1.0 - scattering * invsteps;
  scattering = 1.0 - exp(-scattering);

  //dist = 1.0 - dist * invsteps;
  //dist = 1.0 - exp(-dist * 0.5);
  */
  //rayOrigin = normalize(rayOrigin);
  //rayDirection = normalize(shadowLightPosition);
  vec3 m = normal;//normalize(rayDirection - rayOrigin);

  float ndotl = clamp01(dot(rayDirection, m));
  float ndotv = clamp01(1.0 - dot(-rayOrigin, m));

  vec3 Fo = F(vec3(0.02), pow5(ndotv));
  vec3 Fi = F(vec3(0.02), pow5(ndotl));


  vec3 Pin = normalize(reflect(rayDirection, normal));
  if(inv > 0.0) Pin = normalize(rayDirection);
  //Pin = normalize(rayDirection + normalize(refract(rayDirection, -normal, 1.0 / 1.333)));

  vec3 Pout = rayOrigin;

  //scattering = Fo * Fi * 100.0 * (1.0 - exp(-scattering * 0.5)) * albedo.rgb;

  float d = NormalizedDiffusion(length(Pin - Pout), pow3(albedo.a)) * 1.0;

  color += d * Fo * Fi * 2500.0 * albedo.rgb * sss * sunLightingColorRaw;// * (max(inv, 0.0) * 0.9 + 0.1);
  //color = vec3(sss);

  //color = vec3(dot(normalize(shadowLightPosition), rayOrigin));
  //color += vec3(d);
  //color = vec3(dot(Pin, normalize(-rayOrigin)));
}

vec4 CalculateRayMarchingScattering(in vec4 rayOrigin, in vec4 rayIn, in float dither, in vec3 Ta, in vec3 Ts, in vec3 albedo, in float maxDepth){
  vec3 sunColor = G2Linear(sunLightingColorRaw) * fading;
  vec3 sunLightingScattering = vec3(0.0);

  vec4 raysColor = vec4(0.0);

  vec3 lightingDirection = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  float g = 0.2;

  float mu = dot(normalize(rayOrigin.xyz), lightingDirection);
  float phase = HG(mu, g);

  int steps = 8;
	float invsteps = 1.0 / float(steps);

  float diffthresh = shadowPixel;

  vec3 rayDirection = normalize(rayOrigin.xyz) * invsteps * length(rayOrigin.xyz);
  rayOrigin.xyz += rayDirection * dither;

  float rayInLength = length(rayIn);

  for(int i = 1; i <= steps; i++){
    vec4 testPoint = rayOrigin - vec4(rayDirection, 0.0) * float(i);

    float testLength = length(testPoint.xyz);
    float diff = step(rayInLength, testLength);
    if(isEyeInWater > 0) diff = step(testLength, rayInLength);

    if(!bool(diff)) break;

    float viewDepth = length(testPoint.xyz - rayIn.xyz * step(float(isEyeInWater), 0.5));
    //if(viewDepth > maxDepth && maxDepth < 2.0) continue;
          //viewDepth = min(maxDepth, viewDepth);

    float bias = 0.0;
    vec3 shadowCoord = wP2sP(testPoint, bias);
    float shadowDepth = texture2D(shadowtex1, shadowCoord.xy).x;
    float shadowDepth2 = texture2D(shadowtex0, shadowCoord.xy).x;

    vec4 pI = shadowProjectionInverse * vec4(vec3(shadowCoord.xy, shadowDepth2) * 2.0 - 1.0, 1.0);
    vec4 p = shadowModelView * testPoint;

    vec3 waterAlbedo = texture2D(shadowcolor1, shadowCoord.xy).rgb;

    float translucentDepth = length(p.z - pI.z) * 0.5;
          translucentDepth = min(translucentDepth, maxDepth);

    vec3 extinction = exp(-(Ta + Ts) * translucentDepth);
    //float visibility = (1.0 - step(shadowDepth, shadowDepth2)) * (1.0 - step(shadowCoord.z, shadowDepth2 + diffthresh));
    //float visibility = (1.0 - step(shadowDepth, shadowDepth2)) * step(length(rayIn.xyz), length(testPoint.xyz));
    float visibility = step(shadowCoord.z, shadowDepth + diffthresh);// * (1.0 - step(shadowCoord.z, shadowDepth2 + diffthresh));

    visibility = min(1.0, visibility / (translucentDepth + 1e-5));

    //float mu = dot(lightingDirection, normalize(testPoint.xyz));
    //float phase = HG(mu, g);

    vec3 sampleNormal = texture2D(shadowcolor0, shadowCoord.xy).xyz * 2.0 - 1.0;
    float caustics = 1.0 / pow5(1.0 - saturate(dot(sampleNormal, -lightingDirection)));

    vec3 extinction2 = exp(-(Ts + Ta) * viewDepth);

    sunLightingScattering += visibility * viewDepth * extinction * extinction2;
  }

  //sunLightingScattering = sunLightingScattering * invsteps * Ts * albedo;
  sunLightingScattering *= invsteps * Ts * albedo * invPi;
  sunLightingScattering *= sunColor * phase;

  raysColor.rgb = sunLightingScattering * 36.0;

  return raysColor;
}

vec3 GetScatteringCoe(in vec4 albedo){
  return pow2(albedo.a) * vec3(Pi) * 2.0;
}

vec3 GetAbsorptionCoe(in vec4 albedo){
  if(bool(step(0.95, albedo.a))) return vec3(0.0);

  return (1.0 - albedo.rgb) * pow5(albedo.a) * 10.47 / (length(albedo.rgb + 0.005));
}

void ApplyWaterScattering(inout vec3 color, in vec3 Ts, in vec3 Ta, vec3 lightingColor, in vec3 albedo, in float s){
  vec3 Te = Ta + Ts;
  vec3 extinction = exp(-Te * s);

  color *= extinction;

  vec3 limitDistance = min(vec3(s), exp((Ta + Ts)));

  //vec3 scattering = 1.0 - exp(-Ts * albedo.rgb * s);
  //     scattering *= limitDistance * exp(-Te * limitDistance);
  //     scattering *= Ts * albedo.rgb * 100.0;

  //color += scattering * lightingColor;

  float eyeSkyLight = float(eyeBrightness.y) / 240.0 * 15.0;

  //exp(-(Ta + Ts) * (15.0 - eyeSkyLight) * 30.0) *

  vec3 scattering = 1.0 - exp(-Ts * s * albedo);
  //     scattering *= limitDistance * exp(-(Ta + Ts) * limitDistance);

  //color += scattering * lightingColor * Ts * albedo;
  color += scattering * lightingColor * albedo * min(vec3(1.0), Ts);

  //color += scattering * (1.0 - exp(-(Ts * albedo.rgb) * 1.0)) * limitDistance * exp(-Te * limitDistance) * 6.0;

  //vec3 scattering = 1.0 - exp(-Ts * s * albedo.rgb);
       //scattering *= exp(-Te * min(exp(Te), vec3(s)));
       //color += scattering * lightingColor * min(Ts * s, normalize(albedo.rgb)) * albedo.rgb;

  //scattering *= Ts * albedo.rgb * limitDistance * exp(-(Ta + Ts) * limitDistance) * 300.0;
  //color += scattering * lightingColor;
}

vec3 CalculateTranslucent(in vec3 rayOrigin, in vec3 normal, in float rayMax, in vec4 albedo, in float materials, in float IOR, inout vec2 coord, inout bool TIR){
  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y, jittering);

  bool isSky = bool(step(254.5, materials));
  bool isWater = CalculateMaskID(8, materials);
  bool isIce = CalculateMaskID(79, materials);

  bool isGlass      = CalculateMaskID(20.0, materials);
  bool isGlassPane = CalculateMaskID(106.0, materials);
  bool isStainedGlass = CalculateMaskID(95.0, materials);
  bool isStainedGlassPane = CalculateMaskID(160.0, materials);
  bool AnyGlass = isGlass || isGlassPane || isStainedGlass || isStainedGlassPane;
  bool AnyClearGlass = isGlass || isGlassPane;
  bool AnyStainedGlass = isStainedGlass || isStainedGlassPane;
  bool AnyGlassBlock = isGlass || isStainedGlass;
  bool AnyGlassPane = isGlassPane || isStainedGlassPane;

  vec3 rayDirection = normalize(-rayOrigin);

  vec3 opaquePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));

  float blockDepth = rayMax;
  //float blockDepth = min(rayMax, length(translucentPosition));

  float eta1 = 1.000293;
  float eta2 = IOR;
  if(bool(isEyeInWater) && !isWater) {
    eta1 = IOR;
    eta2 = 1.333;
  }

  float eta = eta1 / eta2;
  if(bool(isEyeInWater)) eta = eta2 / eta1;

  //float eta = 1.333;
  //if(isEyeInWater == 0) eta = 1.000293 / IOR;

  vec3 frontPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
  vec3 solidPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));
  float backFaceLength = texture(gaux1, texcoord).x * 544.0;

  float hL = length(frontPosition - solidPosition);
  float bL = abs(length(frontPosition) - backFaceLength);
  float d = min(hL, bL);

  vec3 refractDirection = vec3(0.0);
  TIR = totalInternalReflection(rayDirection, refractDirection, normal, eta);
  refractDirection *= min(100.0 - 99.0 * float(isWater), d);

  vec3 secRefraction = vec3(0.0);//normalize(refract(rayDirection, normal, eta));
  bool TIRout = totalInternalReflection(rayDirection, secRefraction, normalDecode(texture2D(gaux1, texcoord).xy), eta2 / eta1);

  if(TIRout)
  refractDirection += secRefraction * max(0.0, hL - bL - 0.05);

  vec2 refracted = nvec3(gbufferProjection * nvec4(rayOrigin.xyz + refractDirection)).xy * 0.5 + 0.5;
  if(bool(step(texture2D(gcolor, refracted).a, 0.99))) refracted = texcoord;
  //refracted = texcoord;
  coord = refracted;

  vec3 rP = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(depthtex0, refracted).x) * 2.0 - 1.0));
  vec3 rPO = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(depthtex1, refracted).x) * 2.0 - 1.0));

  //blockDepth = max(min(rayMax, length(rPO - rP)), blockDepth * step(texture2D(gcolor, refracted).a, 0.99));
  //if(!bool(step(texture2D(gcolor, refracted).a, 0.99))) blockDepth = min(rayMax, length(rPO - rP));
  //blockDepth = length(rP - rPO) + albedo.a * albedo.a;
  //blockDepth = min(blockDepth, rayMax + max(0.0, 1.0 - rayMax) * albedo.a);


  //vec3 backPosition  = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(gaux1, texcoord).x) * 2.0 - 1.0));

  blockDepth = min(length(rP - rPO), bL + 1000.0 * float(isWater)) + albedo.a * albedo.a;

  vec3 color = decodeGamma(texture2D(gaux2, refracted).rgb) * decodeHDR;
  if(TIR) color = vec3(0.0);

  //color = vec3(step(10.0, max(0.0, hL - bL - 0.05)));

  //color = L2Gamma(color);

  //color = vec3(1.0);
  /*
  vec3 scatteringcoe = vec3(pow3(albedo.a) * Pi * 4.0);
  vec3 absorptioncoe = pow2(albedo.a) * (1.0 - (albedo.rgb)) * Pi;

  if(AnyGlass) {
    float solidPart = step(0.9, albedo.a);

    scatteringcoe *= solidPart * 0.999 + 0.001;
    absorptioncoe *= (1.0 - solidPart) * 0.999 + 0.001;
  }else{
    if(!isWater) {
      absorptioncoe *= 0.01;
      //scatteringcoe *= 10.0;
    }else{
        absorptioncoe *= 2.0;
    //  absorptioncoe = vec3(albedo.g) / maxComponent(albedo.rgb) * invPi;
    //  scatteringcoe = vec3(albedo.a) * Pi;
    }
  }

  //absorptioncoe = vec3(0.0, 0.0, 0.0);

  vec3 absorption = exp(-absorptioncoe * blockDepth);
  vec3 scattering = 1.0 - exp(-(scatteringcoe * blockDepth));
  vec3 extinction = exp(-(scatteringcoe + absorptioncoe) * blockDepth);

  color = color * extinction;

  vec3 simpleScatering = scattering;
  vec3 scatteringExtinction = exp(-(scatteringcoe) * min(exp(scatteringcoe), vec3(blockDepth)));
  scatteringExtinction = albedo.rgb;
  if(isWater) scatteringExtinction = (albedo.rgb) / mix(maxComponent(albedo.rgb), 1.0, 0.5);

  vec3 lightingColor = G2Linear(skyLightingColorRaw);

  color += lightingColor * simpleScatering * scatteringExtinction;
  */

  //float sss = (1.0);
  //vec4 shading = CalculateSunDirectLighting(rayOrigin, normal, sss);

  //color = L2Gamma(color);

  vec3 Ts = GetScatteringCoe(albedo);
  vec3 Ta = GetAbsorptionCoe(albedo);

  //albedo.rgb = L2Gamma(albedo.rgb);

  float solidPart = step(0.9, albedo.a);

  if(AnyGlass){
    Ts *= mix(0.159, 1.0, solidPart);
    //Ta *= mix(0.333 / sum3(albedo.rgb), 0.0, solidPart);
  }

  if(!AnyGlass && !isWater) {
    Ta *= 0.01;
  }

  //albedo.rgb = L2Gamma(albedo.rgb);

  //if(isWater) scattering *= exp(-extinctioncoe * min(exp(extinctioncoe), vec3(blockDepth))) * normalize(albedo.rgb);
  //if(isWater)
  //scattering *= min(normalize(albedo.rgb), max(vec3(0.001), extinction) * blockDepth * Ts * 200.0);
  //else scattering *= albedo.rgb;

  //scattering *= min(Ts * blockDepth, (albedo.rgb)) * min((extinction + 0.07) * 12.0, vec3(1.0));
  //scattering *= min(Ts * blockDepth, normalize(albedo.rgb)) * min(vec3(1.0), (extinction + 0.05) * 10.0);
  //scattering *= Ts * min(vec3(blockDepth), vec3(exp(Ts))) * 1.0 * max(extinction, exp(-Ts));
  //if(isWater) scattering *= 5.0;

  if((bool(isEyeInWater) && !isWater) || (!bool(isEyeInWater))){
    //color *= extinction;
    //color += scattering * (G2Linear(skyLightingColorRaw));

    vec3 lightingColor = (skyLightingColorRaw);
    //     lightingColor = normalize(lightingColor) * 0.5;
    //if(isWater) lightingColor = vec3(dot03(lightingColor));
    //color = L2Gamma(color);
    //albedo.rgb = L2Gamma(albedo.rgb);

    ApplyWaterScattering(color, Ts, Ta, lightingColor, albedo.rgb, blockDepth);
    //color = G2Linear(color);


    //CalculateSunLightingScattering

    vec4 p1 = gbufferModelViewInverse * nvec4(rPO);//gbufferModelViewInverse * vec4(rPO, 1.0);
    vec4 p0 = gbufferModelViewInverse * nvec4(rayOrigin);

    //color += (p1, p0, dither, Ta, Ts, albedo.rgb, sqrt(rayMax * 2.0)).rgb;
    //color += scattering * G2Linear(sunLightingColorRaw * fading) * HG(dot(rayDirection, normalize(shadowLightPosition)), 0.2) * shading.rgb;
  }

  //color = G2Linear(color);

  //color *= min(vec3(1.0), absorption);

  //color *= max(normalize(albedo.rgb), absorption);

  /*
  vec3 simpleScatering = scattering;
  vec3 scatteringExtinction = exp(-(scatteringcoe) * min(exp(scatteringcoe), vec3(blockDepth)));
  vec3 lightingColor = G2Linear(skyLightingColorRaw);

  color += lightingColor * simpleScatering * scatteringExtinction * scatteringcoe * 4.0;

  color *= max(normalize(albedo.rgb), absorption);
  */

  //color += scattering * G2Linear(skyLightingColorRaw) * exp(-(scatteringcoe + absorptioncoe) * min(exp(scatteringcoe + absorptioncoe), vec3(blockDepth)));
  //color *= max(normalize(albedo.rgb), absorption);

  //color *= mix(normalize(albedo.rgb), vec3(1.0), absorption);
  //color = color * extinction;
  //color = mix(color, albedo.rgb * G2Linear(skyLightingColorRaw), scattering);

  //color += scattering;

  /*
  if(!(isEyeInWater == 1 && isWater)){
    float scatteringFactor = 1.0 - exp(-blockDepth * sqrt(albedo.a) * 0.5);
          scatteringFactor *= dontScattering;
    color = mix(color, albedo.rgb * G2Linear(skyLightingColorRaw), scatteringFactor);

    color -= color * (1.0 - exp(-(albedo.a * blockDepth + albedo.a * 0.5) * Pi)) * exp(-albedo.rgb);
  }
  */
    /*
  vec2 specularPackge = unpack2x8(texture(composite, texcoord).b);

  float smoothness = specularPackge.r;
  float metallic = specularPackge.g;
  float roughness = pow2(1.0 - smoothness);

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float noDiffuse = max(1.0 - solidPart, metallic);
  //if(isIce) noDiffuse = smoothness;

  float g,d;
  vec3 f;

  vec3 reflectDirection = reflect(normalize(rayOrigin), normal);
  vec3 m = normalize(reflectDirection - normalize(rayOrigin));

  FDG(f, d, g, normalize(-rayOrigin), reflectDirection, normal, F0, roughness);
  color.rgb *= 1.0 - f;

  if(isWater) albedo.rgb = vec3(0.02);

  color = L2Gamma(color);

  //PBR lighting but disable diffuse lighting
  vec3 sunLighting = BRDFLighting(L2Gamma(albedo.rgb), normalize(shadowLightPosition), rayDirection, normal, normal, L2Gamma(F0), roughness, noDiffuse);
  //vec3 sunLighting = BRDFLighting(L2Gamma(albedo.rgb * Ts * albedo.a * albedo.a), normalize(shadowLightPosition), rayDirection, normal, normal, L2Gamma(F0), roughness, noDiffuse);
  color += sunLighting * sunLightingColorRaw * shading.rgb * fading;

  color = G2Linear(color);
*/
  vec3 lightingDirection = normalize(shadowLightPosition);
  float lightViewSpaceVisible = step(dot(normal, lightingDirection), 0.0) * 2.0 - 1.0;
  vec3 lightViewNormal = normal * lightViewSpaceVisible;

  rayOrigin = normalize(rayOrigin);
  //lightingDirection = normalize(lightingDirection + normalize(rayOrigin + refractDirection));

  //sss = 1.0 - exp(-pow5(1.0 - sss));

  //color = vec3(1.0 - exp(-(1.0 - sss)));
  //if(color.r > 1.0) color = vec3(1.0, 0.0, 0.0);
  //CalculateSSS(color, rayOrigin, lightingDirection, albedo, normal, lightViewSpaceVisible, sss);

  return color;
}

void CalculateSSS(inout vec3 color, in vec3 rayOrigin, in vec3 rayDirection, in vec3 normal, in vec4 albedo){
  //rayDirection -= normal * 0.05;
  //rayDirection = normalize(rayDirection);

  //vec2 coord = nvec3(gbufferProjection * nvec4(rayOrigin + rayDirection)).xy * 0.5 + 0.5;
  //color = texture2D(gaux2, coord).rgb * overRange;


  int steps = 8;
  float invsteps = 1.0 / float(steps);

  float backFace = sign(dot(rayDirection, normal));

  rayDirection = normalize(refract(rayDirection, backFace * normal, 1.0 / 1.333));

  vec3 rayStart = rayOrigin;
  vec3 rayStep = rayDirection * invsteps * 0.1;

  float depth = 0.0;

  for(int i = 0; i < steps; i++){
    rayStart += rayStep;

    vec3 coord = nvec3(gbufferProjection * nvec4(rayStart)).xyz * 0.5 + 0.5;
    if(floor(coord.xy) != vec2(0.0)) break;

    float d = texture(depthtex0, coord.xy).x;
    float diff = (d - coord.z);

    //if(diff * backFace > 0.0 || diff < -0.001) continue;
    depth += texture2D(gaux2, coord.xy).x * overRange;

    rayStep *= 1.1;
  }

  color = vec3(depth * invsteps);

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

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void CalculateParticleNormal(inout vec3 result){
  vec3 facingToPlayer = nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.8) * 2.0 - 1.0));
  result = normalize(-facingToPlayer);
}

vec2 unpackUnorm2x4(in uint v){
  uvec2 x = (uvec2(v) >> uvec2(0, 4)) & uvec2(0xf);
  return vec2(x) / 15.0;
}
#if 0 
vec3 CalculateSubSurfaceLighting(in vec3 albedo, in vec3 viewPosition, in float metallic){
  vec4 worldPosition = gbufferModelViewInverse * nvec4(viewPosition);

  vec4 shadowCoord = shadowProjection * shadowModelView * worldPosition; shadowCoord /= shadowCoord.w;
  //     shadowCoord.xy /= mix(1.0, length(shadowCoord.xy), SHADOW_MAP_BIAS) / 0.95;
       shadowCoord = shadowCoord * 0.5 + 0.5;
  //     shadowCoord.xy *= 0.8;
       //shadowCoord.z -= (2.0 / 2048.0);

  float radius = 2.0;

  //float d = 0.0;
  
  vec3 L = normalize(shadowLightPosition);
  vec3 V = normalize(-viewPosition);
  vec3 N = normalDecode(texture2D(composite, texcoord).xy);

  float ndotv = dot(V, N);
  float ndotl = dot(L, N);
  /*
  vec3 f = F(vec3(0.02), V, N * sign(dot(V, N)));
  */
  vec3 F0 = mix(vec3(max(metallic, 0.02)), albedo, metallic);
  
  vec3 fout = F(F0, V, N * sign(ndotv));

  vec3 Lo = vec3(0.0);

  int count = 0;
  
  vec2 dither = vec2(GetBlueNoise(depthtex2, texcoord, resolution.y, jitter) ,
                     GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y, jitter));

  //if(bool(step(0.0, ndotl))) return vec3(0.0);

  

  vec4 start = gbufferModelViewInverse * nvec4(viewPosition);

  float stepLength = 0.125;

  vec3 direction = normalize(start.xyz) * stepLength;

  vec4 position = start + vec4(direction * dither.x, 0.0);

  vec3 shadowMapCoord = wP2sP(start);

  for(int i = 0; i < 8; i++){
    position.xyz -= direction;

    //if(length(position.xyz) <= stepLength) break;

    //if(length(position.xyz) > length(start.xyz)) break;

    vec3 coord = wP2sP(position);
    if(coord.x > 0.8 || coord.y > 0.8) continue;

    //Lo += pow(decodeGamma(texture2D(shadowcolor0, coord.xy).rgb), vec3(4.0)) * 0.1 * 0.125;//
    //Lo += 0.125 * step(coord.z, texture(shadowtex0, coord.xy).x + 0.0001);// * decodeGamma(texture2D(shadowcolor0, coord.xy).rgb);

    float depth = texture(shadowtex0, coord.xy).x;
    //if(depth > shadowMapCoord.z) continue;

    float visibility = step(coord.z, depth + 0.0001);

    vec3 normal = mat3(gbufferModelView) * (texture2D(shadowcolor1, coord.xy).rgb * 2.0 - 1.0);

    float cosTheta = saturate(dot(normal, L));

    vec3 fin = F(F0, L, normal);

    //float d = length(position.xyz)

    vec3 Lin = nvec3(shadowProjectionInverse * nvec4(vec3(coord.xy, depth) * 2.0 - 1.0));
    vec3 Lout = nvec3(shadowProjectionInverse * nvec4(vec3(coord.xyz) * 2.0 - 1.0));

    float translucentLength = length(Lout) - length(Lin);
    if(translucentLength < 1e-5 || translucentLength > 1.0) break;

    float leng = exp(-(translucentLength) * 100.0);

    float diff = coord.z - depth;

    Lo += fin / (translucentLength * 5.0);
    //Lo += step(diff, 0.0) * step(-0.005, diff);
    //Lo += fin * cosTheta * step(leng, 1.0) * 0.125 * (1.0 - visibility);//fin * cosTheta 
  }
  
  //Lo = fout;

  Lo *= 0.125;
  Lo *= fout * invPi * albedo.rgb * 1000.0 * fading * sunLightingColorRaw;

  //Lo = length(-L - V) * vec3(0.01);

  /*
  for(float i = -radius; i <= radius; i += 1.0){
    for(float j = -radius; j <= radius; j += 1.0){
      vec2 direction = vec2(i, j);
           direction = RotateDirection(direction, dither);

      vec2 coord = shadowCoord.xy + direction * (1.0 / 2048.0);
           coord = coord * 2.0 - 1.0;
           coord /= mix(1.0, length(coord), SHADOW_MAP_BIAS) / 0.95;
           coord = coord * 0.5 + 0.5;
           coord *= 0.8;

      //vec3 normal = mat3(gbufferModelView) * (texture2D(shadowcolor1, coord).rgb * 2.0 - 1.0);

      d += step(shadowCoord.z, texture(shadowtex0, coord).x);
    }
  }
  */
      /*
  d /= pow2(radius * 2.0 + 1.0);


  float diff = shadowCoord.z - texture(shadowtex0, coord).x;

  Lo = vec3(1.0) * step(diff, 0.0) * step(-0.0005, diff);
  */
  /*
  vec3 fin = F(F0, L, -N * sign(-ndotl));

  vec2 coord = shadowCoord.xy * 2.0 - 1.0;
  coord /= mix(1.0, length(coord), SHADOW_MAP_BIAS) / 0.95;
  coord = coord * 0.5 + 0.5;
  coord *= 0.8;

  vec2 direction = normalize(mat3(gbufferModelViewInverse) * shadowLightPosition).xy;

  float d = 0.0;

  shadowCoord.z -= length(direction) * shadowPixel + shadowPixel / max(0.01, abs(ndotl));
  d += GetShadow(shadowtex0, vec3(coord + direction * shadowPixel, shadowCoord.z));

  shadowCoord.z -= length(direction) * shadowPixel * 2.0;
  d += GetShadow(shadowtex0, vec3(coord + direction * shadowPixel * 3.0, shadowCoord.z));

  shadowCoord.z -= length(direction) * shadowPixel * 2.0;
  d += GetShadow(shadowtex0, vec3(coord + direction * shadowPixel * 5.0, shadowCoord.z));

  shadowCoord.z -= length(direction) * shadowPixel * 2.0;
  d += GetShadow(shadowtex0, vec3(coord + direction * shadowPixel * 7.0, shadowCoord.z));

  shadowCoord.z -= length(direction) * shadowPixel * 2.0;
  d += GetShadow(shadowtex0, vec3(coord + direction * shadowPixel * 9.0, shadowCoord.z));

  shadowCoord.z -= length(direction) * shadowPixel * 2.0;
  d += GetShadow(shadowtex0, vec3(coord + direction * shadowPixel * 11.0, shadowCoord.z));

  d /= 7.0;
  d = 1.0 - d;
*/
  /*
  float d = shadowGatherOffset(shadowtex0, vec3(coord, shadowCoord.z), vec2(1.0, 1.0))
          + shadowGatherOffset(shadowtex0, vec3(coord, shadowCoord.z), vec2(1.0, -1.0))
          + shadowGatherOffset(shadowtex0, vec3(coord, shadowCoord.z), vec2(-1.0, -1.0))
          + shadowGatherOffset(shadowtex0, vec3(coord, shadowCoord.z), vec2(-1.0, 1.0));
        d *= 0.25;
        d = 1.0 - d;
        d *= 1.0;
        */

  //vec3 half = -L - V;

  //Lo = invPi * fin * fout / (d * 30.0) * albedo.rgb * 10.0 * pow5(1.0 - d);//invPi * albedo.rgb / sqrt(dot(half, half) + pow2(d));
  //Lo = 1.0 / vec3(d * 30.0) * pow5(1.0 - d);

  /*
  vec3 Ni = N * sign(ndotl);//mat3(gbufferModelView) * (texture2D(shadowcolor1, coord).rgb * 2.0 - 1.0);
  vec3 fi = F(F0, L, Ni);

  Lo = fi * fo * (d) * albedo.rgb * sunLightingColorRaw * saturate(dot(L, Ni)) * 100.0;
  Lo *= 1.0 - metallic;

  Lo *= invPi;
  */

  return Lo;
}
#endif
void main(){
  vec4 color = texture2D(gaux2, texcoord);
       color.rgb = decodeGamma(color.rgb) * decodeHDR;

  float alpha = texture2D(gnormal, texcoord).x;
  vec4 albedo = texture2D(gcolor, texcoord);
       albedo.rgb = decodeGamma(albedo.rgb);

  float depth = texture(depthtex0, texcoord).x;
  vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0));
  vec3 worldPosition = mat3(gbufferModelViewInverse) * viewPosition;

  vec3 nvP = normalize(viewPosition);

  float viewLength = length(viewPosition);

  vec3 flatNormal = normalDecode(texture2D(gnormal, texcoord).xy);
  vec3 texturedNormal = normalDecode(texture2D(composite, texcoord).xy);
  vec3 visibleNormal = texturedNormal;
  if(bool(albedo.a))flatNormal = texturedNormal;
  if(!bool(albedo.a) && dot(-normalize(viewPosition), visibleNormal) < 0.2) visibleNormal = flatNormal;

  float materials = round(texture(gnormal, texcoord).b * 255.0);
  bool isSky = bool(step(254.5, materials));
  bool isWater = CalculateMaskID(8, materials);
  bool isParticels = bool(step(249.5, materials) * step(materials, 250.5));
  bool emissiveParticels = bool(step(250.5, materials) * step(materials, 252.5));
  bool isIce      = CalculateMaskID(79.0, materials);
  bool isGlass      = CalculateMaskID(20.0, materials);
  bool isGlassPane = CalculateMaskID(106.0, materials);
  bool isStainedGlass = CalculateMaskID(95.0, materials);
  bool isStainedGlassPane = CalculateMaskID(160.0, materials);
  bool AnyGlass = isGlass || isGlassPane || isStainedGlass || isStainedGlassPane;
  bool isLeaves = CalculateMaskID(18.0, materials);
  bool isGrass = CalculateMaskID(31.0, materials);

  vec2 specularPackge = unpack2x8(texture(composite, texcoord).b);
  float smoothness = specularPackge.r;
  float metallic = specularPackge.g;
  float roughness = pow2(1.0 - smoothness);

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec3 torchLightingColor = vec3(1.049, 0.5821, 0.0955);

  vec2 lightmapPackge = unpack2x8(texture(gdepth, texcoord).x);
  float torchLightMap = lightmapPackge.x;
  float skyLightMap = lightmapPackge.y;
  float emissive = texture(gdepth, texcoord).b;

  float blockDepth = 1.0;
  if(isWater) blockDepth = 255.0;
  if(isGlassPane) blockDepth = 0.125;

  vec2 coord = texcoord;
  bool TIR = false;

  if(isParticels || emissiveParticels){
    CalculateParticleNormal(texturedNormal);
    CalculateParticleNormal(visibleNormal);
    CalculateParticleNormal(flatNormal);
  }

  //float IOR = 1.333;
  //if(isGlass || isGlassPane) IOR = 1.5;

  //float n1 = 1.0;
  //float n2 = 1.333;
  //float f0 = pow2((n1 - n2) / (n1 + n2));

  float IOR = 1.0 / ((2.0 / (sqrt(0.04) + 1.0)) - 1.0);

  if(bool(albedo.a)){
      if(!emissiveParticels) color.rgb = CalculateTranslucent(viewPosition, texturedNormal, blockDepth, vec4(albedo.rgb, alpha), materials, IOR, coord, TIR);
      else{
        //todo translucnet particles
        vec3 particlesColor = albedo.rgb * skyLightingColorRaw * pow3(max(0.0, skyLightMap - 0.07) / 0.93);

        float particlesAlpha = albedo.a;
              particlesAlpha *= step(texture(depthtex0, texcoord).x, texture(depthtex1, texcoord).x);

        //color.rgb = mix(color.rgb, particlesColor, particlesAlpha);
      }

    vec3 L = normalize(shadowLightPosition);
    vec3 V = -normalize(viewPosition);

    vec3 h = normalize(L + V);

    vec3 f = F(F0, L, h);
    vec3 kS = f;
    vec3 kD = (1.0 - f) * (1.0 - metallic);

    vec4 shading = CalculateSunDirectLighting(viewPosition, flatNormal);

    vec3 sunDirectLight = SpecularLight(albedo.rgb, normalize(shadowLightPosition), -nvP, visibleNormal, texturedNormal, F0, roughness, metallic);
    if(bool(step(0.95, alpha))) sunDirectLight += DiffuseLight(albedo.rgb, normalize(shadowLightPosition), -nvP, visibleNormal, texturedNormal, F0, roughness, metallic);

    color.rgb += shading.rgb * sunDirectLight * sunLightingColorRaw * fading;
  }

  //if(bool(albedo.a) || isLeaves || isGrass)
  //color.rgb += CalculateSubSurfaceLighting(albedo.rgb, viewPosition, metallic);
  /*
  #if MC_VERSION > 11499
    if(bool(albedo.a)){
      if(!emissiveParticels) color.rgb = CalculateTranslucent(viewPosition, texturedNormal, blockDepth, vec4(albedo.rgb, alpha), materials, IOR, coord, TIR);
      else{
        //todo translucnet particles
        vec3 particlesColor = L2Gamma(albedo.rgb) * skyLightingColorRaw * pow3(max(0.0, skyLightMap - 0.07) / 0.93);
             particlesColor = G2Linear(particlesColor);

        float particlesAlpha = albedo.a;
              particlesAlpha *= step(texture(depthtex0, texcoord).x, texture(depthtex1, texcoord).x);

        color.rgb = mix(color.rgb, particlesColor, particlesAlpha);
      }

      color.rgb = decodeGamma(color.rgb);

      float noDiffuse = max(1.0 - step(0.99, alpha), metallic);
      if(isIce) noDiffuse = smoothness;

      vec3 rayDirection = -normalize(viewPosition);
      vec3 L = normalize(reflect(normalize(viewPosition), visibleNormal));

      float g,d;
      vec3 f;
      FDG(f, d, g, rayDirection, L, visibleNormal, (F0), roughness);
      float brdf = saturate(g * d);

      color.rgb *= (1.0 - brdf * max(f, vec3(metallic)));

      
      //float sss = (1.0);
      vec4 shading = CalculateSunDirectLighting(viewPosition, flatNormal, sss);
      //vec3 sunLighting = BRDFLighting(decodeGamma(albedo.rgb), normalize(shadowLightPosition), rayDirection, visibleNormal, texturedNormal, (F0), roughness, noDiffuse);
      //color.rgb += sunLighting * sunLightingColorRaw * shading.rgb * fading;

      color.rgb *= 1.0 - emissive;
      color.rgb += emissive * decodeGamma(albedo.rgb) * 4.56;

      color.rgb = encodeGamma(color.rgb);
    }
  #endif

  #if MC_VERSION < 11499
    if(bool(albedo.a)) {
      color.rgb = CalculateTranslucent(viewPosition, texturedNormal, blockDepth, vec4(albedo.rgb, alpha), materials, IOR, coord, TIR);

      color.rgb = decodeGamma(color.rgb);

      float noDiffuse = max(1.0 - step(0.99, alpha), metallic);
      if(isIce) noDiffuse = smoothness;

      vec3 rayDirection = -normalize(viewPosition);
      vec3 L = normalize(reflect(normalize(viewPosition), visibleNormal));

      float g,d;
      vec3 f;
      FDG(f, d, g, rayDirection, L, visibleNormal, (F0), roughness);
      float brdf = saturate(g * d);

      color.rgb *= (1.0 - brdf * max(f, vec3(metallic)));

      float sss = (1.0);
      vec4 shading = CalculateSunDirectLighting(viewPosition, flatNormal, sss);
      vec3 sunLighting = BRDFLighting(decodeGamma(albedo.rgb), normalize(shadowLightPosition), rayDirection, visibleNormal, texturedNormal, (F0), roughness, noDiffuse);
      color.rgb += sunLighting * sunLightingColorRaw * shading.rgb * fading;

      color.rgb *= 1.0 - emissive;
      color.rgb += emissive * decodeGamma(albedo.rgb) * 4.56;

      color.rgb = encodeGamma(color.rgb);
    }
  #endif
  */
  vec2 density = vec2(exp(-1.0 / Hr), exp(-1.0 / Hm));

  vec3 Tr = bR * 1.0;
  vec3 Tm = bM * 1.0;

  //color.rgb = L2Gamma(color.rgb);
  //color.rgb = decodeGamma(color.rgb);

  if(!isSky){
    float mu = dot(nvP, normalize(shadowLightPosition));
    float phaseR = 0.0596831 * (1.0 + mu * mu);
    float phaseM = HG(mu, 0.76);

    vec3 extinction = exp(-(Tr + Tm) * viewLength);
    vec3 r = (1.0 - exp(-(Tr) * viewLength)) * phaseR;
    vec3 m = (1.0 - exp(-Tm * viewLength)) * phaseM;

    color.rgb *= extinction;
    color.rgb += r * sunLightingColorRaw;
    color.rgb += m * sunLightingColorRaw;
  }

  if(isEyeInWater == 1){
    vec4 biomeWaterColor = eyesWaterColor;

    vec3 Ta = GetAbsorptionCoe(biomeWaterColor);
    vec3 Ts = GetScatteringCoe(biomeWaterColor);

    biomeWaterColor.rgb = decodeGamma(biomeWaterColor.rgb);

    //if(TIR) ApplyWaterScattering(color.rgb, Ts, Ta, sunLightingColorRaw, biomeWaterColor.rgb, 1000.0);

    float sDistnce = viewLength + biomeWaterColor.a * biomeWaterColor.a;

    //ApplyWaterScattering(color.rgb, Ts, Ta, skyLightingColorRaw, biomeWaterColor.rgb, sDistnce);
  }

  //color.rgb = encodeGamma(color.rgb);

  vec4 rayColor = texture2D(gaux1, texcoord * LightShaft_Quality);
  //color.rgb += rayColor.rgb;
  /*
  if(bool(albedo.a)){
    vec3 frontPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
    vec3 solidPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));
    vec3 backPosition  = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(gaux1, texcoord).x) * 2.0 - 1.0));

    float hL = length(frontPosition - solidPosition);
    float bL = length(frontPosition - backPosition);
    float d = min(hL, bL);//min(bL, hL);

    color.rgb = vec3(d * 0.001);//max(0.0, length(nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(gaux1, texcoord).x) * 2.0 - 1.0)))) * vec3(1.0 / 1000.0);
  }
  */
  color.rgb = encodeGamma(color.rgb * encodeHDR);

  /*
  if(isEyeInWater == 1){
    vec4 waterAlbedo = eyesWaterColor;

    float sDistance = pow2(viewLength) + waterAlbedo.a * waterAlbedo.a;

    vec3 scatteringcoe = vec3(1.0);
    vec3 absorptioncoe = 1.0 - waterAlbedo.rgb;

    vec3 Ts = scatteringcoe;
         Ts *= pow2(waterAlbedo.a) * Pi;

    vec3 Ta = absorptioncoe;
         Ta *= pow3(waterAlbedo.a) * Pi;

    //vec3 torchLightDirection = normalize(viewPosition);
    //vec3 normalFaceToPlayer = normalize(-nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.8) * 2.0 - 1.0)));


    //float lightingRadius = max(float(heldBlockLightValue), float(heldBlockLightValue2));
    //float heldLightingDistance = lightingRadius / 15.0 / min(3.0, sDistance);
          //heldLightingDistance = pow2(heldLightingDistance * heldLightingDistance);

    //vec3 lightingFog = DisneyDiffuse(-torchLightDirection, -torchLightDirection, normalFaceToPlayer, 0.99, L2Gamma(waterAlbedo.rgb));
    //lightingFog *= clamp01(1.0 - exp(-torchLightingColor * heldLightingDistance * 0.125)) * Pi;
    //     lightingFog = G2Linear(lightingFog);

    //color.rgb += lightingFog;

    vec3 lightingColor = G2Linear(skyLightingColorRaw);

    float eyeBrightnessToDepth = (float(eyeBrightnessSmooth.y) / 240.0) * 15.0;

    float skyLightDepth = max(15.0 - (worldPosition.y + eyeBrightnessToDepth), 0.05);
          //skyLightDepth = min(skyLightDepth, 15.0 - eyeBrightnessToDepth);
          skyLightDepth = 15.0 - eyeBrightnessToDepth;
          skyLightDepth = pow2(skyLightDepth * 0.5) + waterAlbedo.a * waterAlbedo.a;
    vec3 extinction = exp(-(Ta + Ts) * skyLightDepth * 4.0);

    //lightingColor *= extinction;

    vec3 torchLignting = G2Linear(torchLightingColor) * (max(0.0, torchLightMap - 0.0667) * 1.071);
    float torchLightMap0to15 = max(0.0, 16.0 * torchLightMap - 1.0);
    float torchLightDistance = 15.0 - torchLightMap0to15;
    color.rgb += torchLignting * exp(-(Ta + Ts) * torchLightDistance) * torchLightDistance * Ts * waterAlbedo.rgb;

    vec3 refractDirection = vec3(0.0);
    bool TIR = totalInternalReflection(normalize(-viewPosition.xyz), refractDirection, normal, 1.333 / 1.000293);

    //lightingColor *= saturate(eyeBrightnessToDepth);

    if(!TIR && isWater) color.rgb *= saturate(eyeBrightnessToDepth);
    if(TIR && isWater) color.rgb = normalize(waterAlbedo.rgb) * lightingColor * waterAlbedo.rgb;

    //(color.rgb, Ts, Ta, lightingColor, waterAlbedo.rgb, sDistance);

    float eyeTorchLight = float(eyeBrightnessSmooth.x) / 240.0 * 15.0;
    color.rgb += G2Linear(torchLightingColor) * waterAlbedo.rgb * (eyeTorchLight / 15.0) * exp(-(Ta + Ts) * (15.0 - eyeTorchLight)) * Ts;

    float heldLight = max(float(heldBlockLightValue), float(heldBlockLightValue2));
    vec3 heldLighting = G2Linear(torchLightingColor) * heldLight / 15.0;// * (max(0.0, (heldLight - viewLength - 2.0) / 15.0));
    //color.rgb += heldLighting * exp(-(Ta + Ts) * min(heldLight, viewLength) * 100.0);// * min(heldLight, viewLength) * Ts * waterAlbedo.rgb;

    //ApplyWaterSunLightScattering(color.rgb, viewPosition, Ts, Ta, waterAlbedo.rgb);

    //vec3 fakeWaterFog =

    //color.rgb = L2Gamma(color.rgb);
    //color.rgb += DisneyDiffuse(-torchLightDirection, -torchLightDirection, normalFaceToPlayer, 0.99, L2Gamma(waterAlbedo.rgb));
    //color.rgb = G2Linear(color.rgb);

  }
  */

  //color.rgb *= step((texture(depthtex0, texcoord).x), 0.7);

  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y, jittering);

  vec4 albedo2 = vec4(albedo.rgb, alpha);
  float maxDepth = blockDepth;

  if(isEyeInWater == 1 && !bool(albedo.a)) {
    albedo2 = eyesWaterColor;
    maxDepth = 255.0;
  }

  vec3 lightingColor = G2Linear(skyLightingColorRaw);
  //     lightingColor = normalize(lightingColor) * 0.5;
  //if(isWater) lightingColor = vec3(dot03(lightingColor));

  vec4 p1 = gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex1, coord).x) * 2.0 - 1.0);
       p1 /= p1.w;
       p1 = gbufferModelViewInverse * p1;
  vec4 p0 = gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex0, coord).x) * 2.0 - 1.0);
       p0 /= p0.w;
       p0 = gbufferModelViewInverse * p0;

  vec3 Ta = GetAbsorptionCoe(albedo2);
  vec3 Ts = GetScatteringCoe(albedo2);

  float solidPart = step(0.9, albedo.a);

  if(AnyGlass){
    Ts *= mix(0.1, 1.0, solidPart);
    Ta *= mix(1.0, 0.0, solidPart);
  }else{
  }

  if(!AnyGlass && !isWater) {
    Ta *= 0.01;
  }

  if(isEyeInWater == 1 || bool(albedo.a)){
    //color.rgb += CalculateRayMarchingScattering(p1, p0, dither, Ta, Ts, albedo2.rgb, sqrt(maxDepth)).rgb;
  }
  //color.rgb = mat3(gbufferModelViewInverse) * visibleNormal;
  //
  //vec4 p0 =
  //vec4 p1 =

  //color.rgb += (p1, p0, dither, Ta, Ts, albedo.rgb, sqrt(rayMax * 2.0)).rgb;

//color.rgb = eyesWaterColor.rgb;
  //color.rgb = skyLightingColorRaw;
  //color.rgb = CalculateInScattering(vec3(0.0), vec3(0.0, 1.0, 0.0), mat3(gbufferModelViewInverse) * normalize(sunPosition), 0.76, ivec2(5, 2), vec3(1.0, 1.0, 0.0));
  //color.rgb = G2Linear(color.rgb);
  //color.rgb = vec3(isParticels || emissiveParticels);

  //color.rgb

  //color.rgb

  //color.rgb *= exp(-vec3(0.12, 0.16, 0.2) * 0.03 * viewLength);
  ///color.rgb *= rayColor.a;
  //color.rgb = vec3(0.0, 0.0, 0.0);
  ///color.rgb += rayColor.rgb;
  //color.rgb = vec3(0.0);
  //color.rgb += rayColor.rgb;

  vec3 L = normalize(shadowLightPosition);
  vec3 D = normalize(viewPosition);

  //color.rgb = vec3(rayColor.a);

  float sunTemperature = dot(normalize(sunPosition), normalize(upPosition));
  float temperature = biomeTemperature * sunTemperature + biomeTemperature * 2.0;
  /*
  float height = (worldPosition.y + cameraPosition.y - 63.05) + 0.0;

  vec3 br = vec3(0.14, 0.18, 0.2);
  vec3 bm = vec3(1.1);

  float dr = exp(-height / 50.0);
  float dm = exp(-height / 20.0);

  vec3 T = exp(-(br) * height) / height * (1.0 - exp(-(br) * height * viewLength));// * (1.0-exp( -distance*rayDir.y*b ))/rayDir.y;

  color.rgb *= T;
  color.rgb += (1.0 - T) * sunLightingColorRaw;
  */

  //color.rgb = rayColor.rgb;

  /*
  //color.rgb += rayColor.rgb * rayColor.a * sunLightingColorRaw * HG(dot(L, D), 0.76);

  float s = viewLength;
  if(isSky) s = 1000.0;

  vec3 b = vec3(0.11, 0.14, 0.2) * 0.02;

  float h = worldPosition.y + (cameraPosition.y - 63.0);
  //h *= 2.0;

  float density = exp(-h / 20.0);
  //color.rgb *= exp(-s * b * density);

  float g = 0.2;
  float mu = dot(L, D);
  float phase = HG(mu, g);
  float phaseR = 0.0596831 * (1.0 + mu*mu);

  //b *= sunLightingColorRaw * raysColor.rgb * raysColor.a * phase + skyLightingColorRaw;

  vec3 scattering = (1.0 - exp(-(b + sunLightingColorRaw) * density * s));
  */
  //color.rgb = texture(noisetex, round(texcoord.xy * float(noiseTextureResolution) + vec2(0.0, 0.0)) / float(noiseTextureResolution)).rgb;
  //color.rgb = vec3(noise(worldPosition.xyz + cameraPosition + vec3(0.05)));

  //color.rgb = vec3(hash(normalize(worldPosition)));

  //color.rgb = (1.0 - exp(-b * viewLength));
  //color.rgb = rayColor.rgb;

  //color.rgb = scattering;

  //color.rgb *= exp(-s * (sunLightingColorRaw + skyLightingColorRaw) * density);
  //color.rgb += (1.0 - exp(-(s * (sunLightingColorRaw + skyLightingColorRaw) * density))) * (sunLightingColorRaw*phase+skyLightingColorRaw*phaseR)/(sunLightingColorRaw+skyLightingColorRaw);

  //color.rgb *= exp(-s * b);
  //color.rgb += (1.0 - exp(-s * b)) * phase;

  //color.rgb = (1.0 - exp(-length(L-D) * s * b)) * phase;

  //color.rgb *= exp(-s * b * density);
  //color.rgb += (1.0 - exp(-((length(L-D) + s) * b * density))) * sunLightingColorRaw * 200.0 * b * (phase + phaseR) * raysColor.a;

  //vec3 bm = sunLightingColorRaw * 0.01;
  //color.rgb *= exp(-(bm+b) * s * density);
  //color.rgb += (1.0 - exp(-(bm+b) * s * density)) * (phase*bm+1.0*b)/(bm+b) * raysColor.a * raysColor.rgb * sunLightingColorRaw;

  //color.rgb *= exp(-s * b * density);
  //color.rgb += (1.0 - exp(-s * b * density)) * (skyLightingColorRaw + sunLightingColorRaw * raysColor.rgb * raysColor.a) * phase;

  //if(texcoord.x > 0.7) color.rgb = vec3(0.0);
  //color.rgb += vec3(1.0) * max(0.0, 2.0 - temperature) * biomeRainFall;

  float maxtemperature = biomeTemperature * 2.0;
  float mintemperature = biomeTemperature;

  //color.rgb += texture2D(gaux1, texcoord * LightShaft_Quality).rgb * (1.0 - exp(-viewLength / 5.0));

  //float d = NormalizedDiffusion(length(normalize(shadowLightPosition) - normalize(viewPosition)), pow3(alpha)) * 1.0;
  //color.rgb = vec3(d);

  //CalculateSSS(color.rgb, viewPosition.xyz, normalize(shadowLightPosition), normal, vec4(albedo.rgb, alpha));


  //float translucentThickness = 1.0;

  //bool escape = length(halfPosition) > translucentThickness;

  //if(escape) color.rgb = vec3(1.0);


  /*
  if(bool(albedo.a)){
    vec3 testPoint = direction + viewPosition.xyz;

    //if(testPoint.z > 1.0)

    vec2 coord = nvec3(gbufferProjection * nvec4(testPoint)).xy * 0.5 + 0.5;
    color.rgb = texture2D(gaux2, coord).rgb * overRange;
  }
  */
/*
  vec3 rayOrigin = normalize(shadowLightPosition);
  vec3 rayDirection = normalize(-viewPosition);

  float ndotl = dot(normal, rayOrigin);
  float ndotv = 1.0 - dot(normal, rayDirection);

  float backFace = sign(ndotl);

  vec3 inPosition = normalize(reflect(rayOrigin, normal));
  vec3 refractPosition = normalize(refract(rayOrigin, normalize(-rayOrigin-normal*(1.0 - clamp01(ndotl))*0.1), 1.333));
  vec3 hitPosition = mix(refractPosition, inPosition, step(0.0, ndotl));

  vec3 Fi = F(F0, abs(pow5(ndotl)));
  vec3 Fo = F(F0, clamp01(pow5(ndotv)));

  //scattering += step * steps

  if(!isSky){
    //float radius = length(-rayPosition - rayDirection) * 0.001;
    vec3 muti;
    vec3 scattering;
    vec3 f = Fi * Fo;
    //vec3 s = vec3(NormalizedDiffusion(radius, 1.0 - alpha * 0.9));

    float dither = R2sq(texcoord * resolution - jittering);

    vec3 start = viewPosition;
    vec3 direction = rayOrigin * 0.0625;
    vec3 test = start + direction * dither;

    float rayDepth = 0.0;

    float count = 0.0;

    vec3 discStartPosition = -hitPosition - rayDirection;

    float c = 1.0 - (alpha * alpha);

    float s = min(1.0, NormalizedDiffusion(length(discStartPosition), c));
    float sumScattering = s;

    //color.rgb = min(vec3(1.0), Fi * Fo * s) * (1.0 / Pi) * 2.0;
    for(int i = 0; i < 16; i++){
      vec2 sampleCoord = nvec3(gbufferProjection * nvec4(test)).xy * 0.5 + 0.5;
      vec3 samplePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(sampleCoord, texture(depthtex0, sampleCoord).x) * 2.0 - 1.0));

      float l = (test.z * test.z - samplePosition.z * samplePosition.z);
            l = sqrt(l);
      if(l > sqrt(2.0)) continue;

      float uvclamp = step(0.0, sampleCoord.x) * step(sampleCoord.x, 1.0) * step(0.0, sampleCoord.y) * step(sampleCoord.y, 1.0);
      float layerAlpha = texture2D(composite, sampleCoord).x;
      float layerDepth = (float(i) + dither) / 16.0;
      float layerScattering = NormalizedDiffusion(length(discStartPosition - direction * layerDepth), c) * s;
            //layerScattering = min(1.0, layerScattering);

      sumScattering += layerScattering;
      s *= min(1.0, layerScattering);

      float l2 = (samplePosition.z - start.z);
      //      l2 = sqrt(pow2(l2));

      if(test.z < samplePosition.z && samplePosition.z < test.z + 0.35){
        rayDepth += uvclamp;
        count += uvclamp;
      }

      test += direction;
    }

    sumScattering /= 1.0 + count;
    rayDepth = 1.0 - rayDepth / max(1.0, count);

    scattering = vec3(sumScattering);

    vec3 shading = vec3(1.0);

    float bias = 0.0;
    vec3 shadowPosition = wP2sP(vec4(worldPosition, 1.0), bias);
    float shadowMapDepth = texture(shadowtex1, shadowPosition.xy).x;
    float directLighting = step(shadowPosition.z, shadowMapDepth + shadowPixel * 1.0) * rayDepth;
    shading = mix(vec3(directLighting), vec3(1.0), albedo.a * step(alpha, 0.999));

    color.rgb += min(vec3(10.0), scattering * Fo * Fi * 2500.0 / Pi) * sunLightingColorRaw * 2.0 * shading * abs(ndotl);
    //color.rgb = vec3(rayDepth);
    //color.rgb = vec3(abs(dot(refractPosition, -rayDirection)));
  }
*/

  color.a = max(albedo.a, float(!isSky));

  vec4 tex = vec4(texture2D(gcolor, texcoord).rgb, 1.0);
  if(bool(step(0.5, albedo.a))) tex.a = min(0.99, texture(gnormal, texcoord).x * 0.8 + 0.2);
  if(isSky) tex = vec4(vec3(0.0), 1.0);

  color.a = max(emissive * 100.0, step(14.0, torchLightMap * 15.0) * 0.0);

  //color.rgb = mat3(gbufferModelViewInverse) * texturedNormal;
  /*
  vec3 uv = mat3(gbufferModelViewInverse) * reflect(normalize(viewPosition), flatNormal);
			 uv = uv.xzy;
			 uv.xy /= 1.0 + sqrt(uv.z * uv.z);

	uv.xy = uv.xy * 0.5 + 0.5;
  //uv.xy = round(uv.xy * 512.0) / 512.0;
	uv.xy *= 0.2;
	uv.x += 0.8;

	if(uv.z < 0.0){
		//uv.y = 0.2 - uv.y;
	}
  uv.y += 0.2;

  float mapReflectionAlpha = 1.0;

  float sp = 1.0 / 2560.0;

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 coord = round(uv.xy * 2560.0 + vec2(i, j)) / 2560.0;

      if(texture(shadowtex0, coord).x > 0.999) {
        mapReflectionAlpha = 0.0; break;
      }
    }
  }

	vec3 mapReflectionAlbedo = decodeGamma(texture2D(shadowcolor1, uv.xy).rgb);
  vec3 mapReflectionNormal = mat3(gbufferModelView) * (texture2D(shadowcolor0, uv.xy).xyz * 2.0 - 1.0);

  float encodeLightMap = texture2D(shadowcolor1, uv.xy).a;
  vec2 lightMap = vec2(fract(encodeLightMap * 16.0), floor(encodeLightMap * 16.0) / 15.0);
       lightMap.x = saturate(1.0 - lightMap.x * 100.0);


  //uv.xy = texcoord;
  vec4 sphereViewPosition = vec4(texture2D(shadowcolor1, uv.xy).xyz * 2.0 - 1.0, 1.0);//gbufferProjectionInverse * vec4(vec3(uv.xy, texture(shadowcolor1, uv.xy).b) * 2.0 - 1.0, 1.0);
       //sphereViewPosition /= sphereViewPosition.w;

  float dist = 0.0;
  vec3 sphereShadowCoord = wP2sP(sphereViewPosition, dist);

  float shading = step(sphereShadowCoord.z, texture(shadowtex0, sphereShadowCoord.xy).x + 0.001);

  vec3 torchLight = mapReflectionAlbedo.rgb * lightMap.x * torchLightingColor;
  vec3 skyLight = mapReflectionAlbedo.rgb * lightMap.y * skyLightingColorRaw;
  vec3 sunLight = mapReflectionAlbedo.rgb * sunLightingColorRaw * saturate(dot(normalize(shadowLightPosition), mapReflectionNormal)) * fading * lightMap.y;

  vec3 mapReflection = mapReflectionAlbedo * 0.01 * vec3(invPi) * step(uv.z, 0.0) * step(texture(shadowtex0, uv.xy).x, texture(shadowtex0, uv.xy - vec2(0.0, 0.2)).x + 0.001);
  */
	//color.rgb = encodeGamma(mapReflection * encodeHDR);// * step(10.0, length(mapViewPosition.xyz));

  //color.rgb = encodeGamma(color.rgb);

  //if(texcoord.x > 0.7)

  //color.rgb = vec3(step(texture(shadowtex0, texcoord).x, texture(depthtex0, texcoord).x)) * 0.001;
  /*
  vec3 uv = mat3(gbufferModelViewInverse) * reflect(normalize(viewPosition), flatNormal); uv = uv.xzy;
  
  uv.z += near * near * 2.0 * length(uv.xy) * sign(uv.z);

  uv = normalize(uv);

  uv.xy = uv.xy / (abs(uv.z) + 1.0);

  uv.xy = uv.xy * 0.5 + 0.5;

  if(uv.z > 0.0){
    uv.x = 1.0 - uv.x;
  }

  uv.xy *= 0.2;
  uv.x += 0.8;

  if(uv.z > 0.0) {
    uv.y += 0.4;
  }
  
  float dist = 0.0;

  color.rgb = vec3(texture2D(shadowcolor1, uv.xy).rgb) * 0.01;
  */

  //vec3 sphereShadowCoord = texture2D(shadowcolor1, texcoord).rgb;
  //if(sphereShadowCoord.z < texture(shadowtex1, sphereShadowCoord.xy).x);

  //color.rgb = vec3(texture2D(shadowcolor0, texcoord).aaa * 0.01);

  //float z = length()
  //sP.xy /= length(uv.xyz * 2.0 - 1.0);

  //color.rgb = step(sP.z, texture(shadowtex0, sP.xy).x) * vec3(0.01);

  //color.rgb = vec3(0.01) * step(length(sphereViewPosition.xyz), 3.0);
  //color *= step(texture(shadowtex0, uv.xy).x * far, 3.0);

  //color.rgb *= texture2D(shadowcolor1, uv.xy).rgb;
  //color.rgb = texture2D(gcolor, texcoord * 2.0).rgb * 0.01;
  /*
  if(texcoord.x < 0.5){
    //color.rgb = texture2D(shadowcolor1, texcoord * 0.5).rgb * 0.01;

    vec3 uv3 = vec3(texcoord * 0.5, texture(shadowtex0, texcoord * 0.5).x) * 2.0 - 1.0;
    uv3 = uv3 / length(uv3);
    uv3 = uv3 / (uv3.z + 1.0);

  }else{
    color.rgb = step(texture(depthtex0, texcoord).x, 0.995) * vec3(0.01) * texture2D(gcolor, texcoord).rgb;
  }

  */
  //color.rgb = vec3(0.01) * texture2D(shadowcolor1, texcoord.xy).rgb;
  //color.rgb = vec3(pow(pow(texture2D(shadowtex0, texcoord * 0.4 + vec2(0.8, 0.0)).x, 2.2) * encodeHDR, 1.0 / 2.2));
  //color.rgb = vec3(texture(depthtex0, texcoord).x * 0.005);

  /* DRAWBUFFERS:05 */
  gl_FragData[0] = tex;
  //gl_FragData[1] = vec4(reflectionMap, 1.0);
  gl_FragData[1] = color;
}

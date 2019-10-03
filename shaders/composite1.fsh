#version 130

#extension GL_EXT_gpu_shader4 : require
#extension GL_EXT_gpu_shader5 : require

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

uniform vec3 vanillaWaterColor;
uniform vec3 weatherFogColor;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform float frameTimeCounter;
uniform float rainStrength;
uniform float biomeTemperature;
uniform float biomeRainFall;

uniform int frameCounter;
uniform int isEyeInWater;
uniform int heldBlockLightValue;

uniform ivec2 eyeBrightness;
uniform ivec2 eyeBrightnessSmooth;

in vec2 texcoord;

in float fading;

in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

in vec4 eyesWaterColor;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define CalculateHightLight 1
#define CalculateShadingColor 2

//#define Void_Sky

#include "libs/common.inc"
#include "libs/dither.glsl"
#include "libs/jittering.glsl"
#include "libs/atmospheric.glsl"
#include "libs/brdf.glsl"
#include "libs/light.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}
/*
vec3 CalculateRays(in vec4 wP){
  vec3 raysColor = vec3(0.0);

  int steps = 4;

  vec3 nwP = wP.xyz;
       nwP = normalize(nwP) * length(nwP) / steps;

  float dither = R2sq(texcoord * resolution + R2sq2[int(mod(frameCounter, 16))]);
        //dither = bayer_32x32(texcoord + R2sq2[int(mod(frameCounter, 16))] * pixel, resolution);
        //dither = hash(texcoord * resolution + R2sq2[int(mod(frameCounter, 16))]);
        //dither = 1.0;

  vec4 rP = wP + vec4(nwP, 0.0) * dither * 0.11;

  for(int i = 0; i < steps; i++){
    rP.xyz -= nwP.xyz;

    vec4 sP = rP + vec4(nwP, 0.0) * dither * (1.0 + float(i) / steps);
         sP.xyz = wP2sP(sP);

    float allBlockRays = step(sP.z, texture2D(shadowtex0, sP.xy).z + 0.001);
    float solidBlockRays = step(sP.z, texture2D(shadowtex1, sP.xy).z + 0.001);

    if(floor(sP.xyz) == vec3(0.0)){
      //float scattering = 1.0 - clamp01(exp(-(length(mat3(gbufferModelView) * rP.xyz) * 0.2)));
      float scattering = clamp01(length(mat3(gbufferModelView) * (rP.xyz)) * 0.02);
      raysColor += mix(vec3(1.0), texture2D(shadowcolor0, sP.xy).rgb, clamp01(solidBlockRays - allBlockRays)) * solidBlockRays * scattering;
      //raysColor += scattering;
    }else{
      //raysColor += mix(vec3(1.0, .0, .0), vec3(1.0), allBlockRays);
      raysColor += vec3(1.0);
    }

    //raysColor += mix(vec3(1.0), vec3(step(sP.z, texture2D(shadowtex1, sP.xy).z + 0.0005)), float(floor(sP.xyz) == vec3(0.0)));
  }

  raysColor /= steps;
  //sunDirectLighting /= steps;

  //raysColor *= pow(sunDirectLighting, 1.0);

  //wP.xyz = wP2sP(wP);

  //if(-wP.z > -0.01 || wP.z > sky) return vec3(1.0);
  //return wP.xyz;
  //return texture2D(shadowcolor0, wP.xy).rgb;
  return raysColor;
}
*/
void main(){
  vec4 albedo = texture2D(gcolor, texcoord);

  vec4 color = texture2D(gaux2, texcoord);
       //color = rgb2L(color);

  float torchLightMap = texture2D(gdepth, texcoord).x;
  float skyLightMap = texture2D(gdepth, texcoord).y;

  float alpha = texture2D(gnormal, texcoord).z;

  float smoothness = texture2D(composite, texcoord).r;
        //smoothness *= smoothness;
  float metallic   = texture2D(composite, texcoord).g;
  float emissive   = texture2D(composite, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;

  float IOR = 1.0 + texture2D(composite, texcoord).b;
  IOR += step(0.5, metallic) * 49.0;

  float ri = 1.000293;
  if(isEyeInWater == 1) ri = 1.333;

  float ro = IOR;

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);
  vec3 normalBackFace = -normal;

  float depth = texture2D(depthtex0, texcoord).x;

  int id = int(round(texture2D(gdepth, texcoord).z * 255));

  bool isSky = id == 255;
  bool isWater = id == 8;

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0); vP /= vP.w;
  vec4 wP = gbufferModelViewInverse * vP;

  vec4 svP = (gbufferProjectionInverse * nvec4(vec3(texcoord, texture2D(depthtex1, texcoord).x) * 2.0 - 1.0));
       svP /= svP.w;

  vec2 juv = texcoord;
  vec4 jvP = vP;
  vec4 jwP = wP;

  #ifdef Enabled_TAA
     juv += R2sq2[int(mod(frameCounter, 16))] * pixel;
     jvP = gbufferProjectionInverse * nvec4(vec3(juv, texture2D(depthtex0, juv).x) * 2.0 - 1.0);
     jvP /= jvP.w;
     jwP = gbufferModelViewInverse * jvP;
  #endif

  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(nvP, normal);
  vec3 nrP = normalize(reflect(nvP, normal));
  vec3 refractP = refract(nvP, normal, ri);

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  //float dither = bayer_32x32(uvJittering, resolution);
  //float dither2 = hash(uvJittering * resolution);

  color.a = float(!isSky) + albedo.a;

  vec3 torchLightingColor = rgb2L(vec3(1.022, 0.782, 0.344));
/*
  if(isWater){
    float bias = 1.0;
    vec3 shadowPosition = wP2sP(gbufferModelViewInverse * svP, bias);
    float blocker = texture2D(shadowcolor0, shadowPosition.xy).z;
    float receiver = shadowPosition.z - (1.0 / shadowMapResolution) * 2.0;

    color.rgb *= vec3(1.0 - clamp01((receiver - blocker) / blocker));
  }
*/
  if(albedo.a > 0.99){
    vec3 h = normalize(nrP - nvP);

    float ndoth = clamp01(dot(normal, h));

    float vdoth = pow5(1.0 - clamp01(dot(-nvP, h)));

    //float r = mix(ri, ro, step(0.0, vdoth));

    vec3 f = F(F0, vdoth);

    float ndotl = clamp01(dot(nrP, normal));
    float ndotv = 1.0 - clamp01(dot(-nvP, normal));

    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);

    //color.rgb = vec3(1.0 - clamp01(d));

    vec3 halfVector  = vP.xyz - svP.xyz;

    float transBlockDepthMax = 1.0 * mix(1.0, 0.125, float(id == 106) * (1.0 - alpha));
    float transBlockDepth = min(transBlockDepthMax, max(length(halfVector), vdoth));

    vec3 customP = mat3(gbufferModelView) * (wP.xyz + (cameraPosition - (vec3(38.5, 86.5, 175.5))));
    vec3 customN = normal;//normalize(vP.xyz * vec3(-1.0, 1.0, 1.0));

    //float(floor((length(halfVector) / transBlockDepthMax) + 0.05) == 0.0)
    vec3 iIn = refract(nvP, normal, 1.000293 / IOR);
    vec3 iOut = refract(normalize(iIn), normalize(normal + normalize(normal - nvP) * transBlockDepth), 1.000293 / IOR);
    vec3 iR = iIn - (iOut - iIn) * float(!isWater && length(halfVector) < 1.0);//clamp01(floor(length(halfVector) / transBlockDepthMax + ) - float(isWater) * 1000)
         //iOut *= 1.0 / pow5(1.0 - clamp01(dot(normalize(iOut), normalize(normal + normalize(normal - nvP) * transBlockDepthMax))));
         //iIn = vP.xyz + refract(normalize(iIn), normal, (1.000293 / IOR) / IOR) * transBlockDepthMax;
         //iOut *= length(vP.xyz - iOut);


    d = clamp01(d) * 0.999 + 0.001;

    float roughnGlass  = clamp01((dot(h, -nvP)));
          //roughnGlass *= clamp01((dot(h, -normalize(vP.xyz + iIn))));
          //roughnGlass = 1.0 / (roughnGlass + 0.001);
          //roughnGlass = 1.0 / (mix(0.001, 1.0 - alpha, roughnGlass) * ndoth * ndoth * (1.0 / (d * (1.0 + transBlockDepth))) * 2.0 * Pi) * 8.0;

          //roughnGlass = 1.0 / (roughnGlass + 0.001);
          //roughnGlass *= 8.0;
          //roughnGlass *= clamp01(pow5(1.0 - dot(h, -normalize(iIn + vP.xyz))));
          //roughnGlass *= (1.0 / (d * Pi));
        /*
          roughnGlass = F(vec3(alpha * 0.999 + 0.001), pow5(vdoth)).x;
          roughnGlass *= F(vec3(alpha * 0.996 + 0.004), pow5(1.0 - clamp01(dot(normalize(iIn - (iOut - iIn)), h)))).x;
          roughnGlass *= min(length(halfVector), roughnGlass);
          roughnGlass *= (1.0 / d) * Pi;
          roughnGlass = clamp01(roughnGlass) * 8.0;
          */
          //roughnGlass = 0.0;

    //refractP = refractP * max(min(length(halfVector), 1.0 - 0.875 * float(id == 106)), pow5(vdoth));

    vec2 rcoord = nvec3(gbufferProjection * nvec4(vP.xyz + iR)).xy * 0.5 + 0.5;
    //     rcoord += (vec2(dither2, dither) * 2.0 - 1.0) * pixel * roughnGlass;
    //if(floor(rcoord) != vec2(0.0) || (texture2D(composite, texcoord).b > 0.519 && texture2D(composite, texcoord).b < 0.521)) rcoord = clamp(rcoord + (vec2(dither2, 1.0 - dither) - 0.5) * pixel * (1.0 + alpha * 7.0), pixel, 1.0 - pixel);

    vec4 color2 = texture2D(gaux2, rcoord);
    //color2.rgb = vec3(pow5(clamp01(dot())))

    //color2.rgb = f * F(F0, pow5(1.0 - clamp01(dot(normalize(wi), normal))));

    //for(int i = 0; i < 8; i++){
    //  float r = (float(1 + i) * 2.0 * 3.14159 + dither) / 8.0;

    //  color2.rgb += texture2D(gaux2, rcoord + vec2(cos(r), sin(r)) * pixel * roughnGlass).rgb;
    //}

    //color2.rgb *= 1.0 / 9.0;

    float tranBlockR = texture2D(gcolor, rcoord).a;

    color.rgb = color2.rgb * overRange;

    vec3 transBlockAlbedo = albedo.rgb * (1.0 - metallic * alpha);
    if(isWater) transBlockAlbedo = vec3(0.02) + max(albedo.rgb - albedo.rgb * albedo.b, vec3(0.0)) * (albedo.r + albedo.g) * 0.5;
         transBlockAlbedo = rgb2L(transBlockAlbedo);

    vec4 sunDirctLighting = CalculateShading(shadowtex0, shadowtex1, wP, 1.0);
    vec3 shading = mix(vec3(1.0), sunDirctLighting.rgb, sunDirctLighting.a);

    vec3 sunLighting = BRDF(transBlockAlbedo * (0.001 + 0.999 * alpha), normalize(shadowLightPosition), -nvP, normal, normal, roughness, metallic, F0) * shading * sunLightingColorRaw * pow(fading, 5.0) * 3.77;

    vec3 heldTorchLighting = vec3(0.0);
    if(heldBlockLightValue > 1) heldTorchLighting += BRDF(transBlockAlbedo * (0.001 + 0.999 * alpha), -nvP, -nvP, normal, normal, roughness, metallic, F0) * pow2(clamp01((heldBlockLightValue - 5.0 - length(vP.xyz)) * 0.1)) * torchLightingColor;

    vec3 halfVectorR = nvec3(gbufferProjectionInverse * nvec4(vec3(rcoord, texture2D(depthtex0, rcoord).x) * 2.0 - 1.0)) - nvec3(gbufferProjectionInverse * nvec4(vec3(rcoord, texture2D(depthtex1, rcoord).x) * 2.0 - 1.0));
    float halfVectorLength = max(length(halfVector) * (1.0 - texture2D(gcolor, rcoord).a), length(halfVectorR));
          halfVectorLength = min(halfVectorLength + 1.0, transBlockDepthMax + 254.0 * float(isWater));
    if(isEyeInWater == 1) halfVectorLength = 0.0;

    float FD90 = 0.5 + 2.0 * roughness * ndoth * ndoth;

    //vec3 scattering = transBlockAlbedo * skyLightingColorRaw * skyLightMap * (clamp01(dot(normalize(upPosition), normal)) * 0.3 + 0.7);
    //     scattering += max(vec3(0.0), transBlockAlbedo * torchLightingColor * torchLightMap * 0.06 - sunLighting - scattering);
    //     scattering = L2rgb(scattering);

    //vec3 scattering = L2rgb(transBlockAlbedo * skyLightingColorRaw * skyLightMap);
    //vec3 scattering = skyLightingColorRaw * skyLightMap;
    //     scattering = max(vec3(0.0), transBlockAlbedo - scattering) * mix(vec3(1.0), skyLightingColorRaw, skyLightMap);
    //     scattering = L2rgb(scattering);
         //scattering = max(vec3(0.0), transBlockAlbedo * transBlockAlbedo - scattering) * mix(vec3(1.0), skyLightingColorRaw, skyLightMap);

    float scatteringFactor = alpha * (alpha + halfVectorLength) * pow2(alpha + 1.0);
          scatteringFactor = 1.0 - clamp01(exp(-scatteringFactor));
          scatteringFactor = max(vdoth, scatteringFactor);

    vec3 torchLighting = transBlockAlbedo;
    if(isWater) torchLighting = vec3(dot03(transBlockAlbedo));
    torchLighting = torchLightingColor * torchLightMap * torchLighting;

    vec3 albedoL = rgb2L(albedo.rgb);
    vec3 scattering = skyLightingColorRaw * skyLightMap;
    vec3 colorAbsorption = (albedoL - scattering);
         colorAbsorption *= colorAbsorption;

    if(alpha < 0.999) scattering = max(scattering - max(scattering - colorAbsorption, vec3(0.0)) * 1.07, vec3(0.0)) * scattering * 4.0;
    scattering = L2rgb(scattering);

    if(isWater && isEyeInWater == 1) {
      scatteringFactor = step(length(refractP), 0.9);
    }

    //scattering = torchLightMap * torchLightingColor * (1.0 - scatteringFactor);

    color.rgb = mix(color.rgb, scattering, scatteringFactor);
    //color.rgb = scattering;

    vec3 absorption = albedo.rgb;

    float absorptionFactor = 1.0 - minComponent(absorption);
          absorptionFactor = 1.0 - clamp01(exp(-(1.0 + halfVectorLength * absorptionFactor)));
          absorptionFactor = max(vdoth, scatteringFactor);

    if(isWater && isEyeInWater == 1) {
      absorptionFactor = step(length(refractP), 0.9);
    }

    color.rgb *= mix(vec3(1.0), absorption, absorptionFactor);

    //torchLighting *= scatteringFactor * (1.0 - scatteringFactor);
    //torchLighting += heldTorchLighting;
    //torchLighting = max(torchLighting * 0.5 - (scattering + sunLighting) * 0.33, vec3(0.0));
    //torchLighting *= mix(vec3(1.0), absorption, absorptionFactor * (1.0 - absorptionFactor));

    color.rgb *= (1.0 - f * min(1.0, CalculateBRDF(-nvP, nrP, normal, pow2(1.0 - smoothness * smoothness))));
    color.rgb = rgb2L(color.rgb);
    color.rgb += max(heldTorchLighting - sunLighting * 0.33, vec3(0.0)) + sunLighting;
    color.rgb = L2rgb(color.rgb);

    color.rgb /= overRange;

    //roughnGlass = (abs(dot(h, -nvP)) * abs(dot(h, nrP))) / (abs(dot(normal, -nvP)) * abs(dot(normal, nrP)));
    //roughnGlass = (pow2(ri) * (1.0 - vdoth) * g * d) / pow2((ri) * clamp01(pow2(dot(-nvP, h)) + (ro) * clamp01(dot(nrP, h))));
    //roughnGlass = clamp01(roughnGlass * (1.0 + halfVectorLength));

    //color.rgb = vec3(roughnGlass);

    //color.rgb = vec3(clamp01(roughnGlass));

    //color.rgb = vec3(clamp01(dot(nvP, normalize(nvP + normalBackFace))));
    /*
    vec4  particlesColor = texture2D(gaux1, texcoord);
    float particlesDepth = length(nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture2D(gaux3, texcoord).z) * 2.0 - 1.0)));

    if(length(vP.xyz) * 1.05 - particlesDepth - 0.05 > 0.0 && particlesColor.a > 0.0){
      //particlesColor.rgb /= mix(1.0, 0.04 + particlesColor.a, emissive);

      emissive = texture2D(gaux3, texcoord).r;

      color.rgb = mix(color.rgb, particlesColor.rgb, particlesColor.a * (1.0 - emissive));
      color.rgb += particlesColor.rgb * emissive;
    }
    */

    //if(length(refractP) < 0.99) color.rgb = scattering;

  }

  float fogScattering = 0.0;
  float fogAbsorption = 0.0;

  if(isEyeInWater == 1){
    //color.rgb = vanillaWaterColor * 0.25 * 0.5;
    vec4 waterColor = eyesWaterColor;
         waterColor.rgb = rgb2L(waterColor.rgb);

    float viewVectorLength = min(far, length(vP.xyz) + 1.0);

    vec3 scattering = skyLightingColorRaw;
         scattering = clamp01(waterColor.rgb - scattering) * scattering + scattering * waterColor.rgb;
         scattering = L2rgb(scattering * 0.1);

    //float d = DistributionTerm(0.9996, 0.995);

    //fogScattering = clamp01(d) * (waterColor.a * waterColor.a * 0.9999 + 0.0001) * (waterColor.a + viewVectorLength) / Pi * 25.0 * clamp01(1.0 - torchLightMap * 0.5);
    fogScattering = waterColor.a * (alpha + viewVectorLength) * pow2(waterColor.a + 1.0);
    fogScattering = 1.0 - clamp01(exp(-fogScattering));

    color.rgb *= overRange;

    color.rgb = mix(color.rgb, scattering, fogScattering);

    color.rgb = rgb2L(color.rgb);
    color.rgb += waterColor.rgb * waterColor.a * torchLightingColor * max(0.0, (heldBlockLightValue - 5.0) * 0.1) * clamp01(1.0 - 0.5 * length((texcoord * 2.0 - 1.0) * vec2(aspectRatio, 1.0)));
    color.rgb = L2rgb(color.rgb);

    vec3 absorption = L2rgb(waterColor.rgb);

    float fogAbsorption = 1.0 - minComponent(absorption);
          fogAbsorption = 1.0 - clamp01(exp(-(1.0 + fogAbsorption * viewVectorLength)));

    color.rgb *= mix(vec3(1.0), absorption, fogAbsorption);

    //color.rgb = vec3((1.0 - clamp01(1.0 - dot(-nvP, normalize(vP.xyz * vec3(1.0, 1.0, -1.0))))) * 0.1;

    color.rgb /= overRange;
  }

/* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color.rgb, (float(!isSky) + albedo.a) * (1.0 - fogScattering));
  //gl_FragData[1] = vec4(fogScattering, fogAbsorption, 0.0, 1.0);
}

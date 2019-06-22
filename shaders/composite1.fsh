#version 130

#define SHADOW_MAP_BIAS 0.9

#define SpecularityReflectionPower 2.0            //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

//uniform vec3 waterFogColor;

uniform float viewWidth;
uniform float viewHeight;

uniform int frameCounter;
uniform int isEyeInWater;
uniform int heldBlockLightValue;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define CalculateHightLight 1
#define CalculateShadingColor 2

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

  int steps = 8;

  vec3 nwP = wP.xyz - vec3(0.0, playerEyeLevel, 0.0);
       nwP = normalize(nwP) * length(nwP) / steps;

  float dither = hash((texcoord + haltonSequence_2n3[int(mod(frameCounter, 16))] * pixel) * resolution) * 0.9;
        //dither *= 0.5;

  vec4 rP = wP + vec4(nwP, 0.0) * dither * 0.11;

  //if(isSky){
    //raysColor = vec3(1.0);
  //}

  for(int i = 0; i < steps; i++){
    rP.xyz -= nwP.xyz;

    vec4 sP = rP + vec4(nwP, 0.0) * dither * (1.0 + float(i) / steps);
    float sunDirectLighting = clamp01(dot(normalize(mat3(gbufferModelView) * sP.xyz), normalize(shadowLightPosition)));
    //if(floor(sP.xy) == vec2(0.0)){
         sP.xyz = wP2sP(sP);
    //if(floor(sP.xyz) == vec3(0.0)){

    //if(-sP.z < 0.0)
    //if(i == 2)raysColor = -sP.xyz * steps;
    //if(i == 7){
    raysColor += mix(vec3(1.0), vec3(step(sP.z, texture2D(shadowtex1, sP.xy).z + 0.0005)), float(floor(sP.xyz) == vec3(0.0))) * sunDirectLighting;
    //}
    //}
    //dither *= 1.1;
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

  vec3 svP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture2D(depthtex1, texcoord).x) * 2.0 - 1.0));

  vec2 uvJittering = texcoord;

  #ifdef Enabled_TAA
     uvJittering += haltonSequence_2n3[int(mod(frameCounter, 16))] * pixel;
  #endif

  vec4 vPJittering = gbufferProjectionInverse * nvec4(vec3(uvJittering, texture2D(depthtex0, uvJittering)) * 2.0 - 1.0); vPJittering /= vPJittering.w;
  vec4 wPJittering = gbufferModelViewInverse * vPJittering;

  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(nvP, normal);
  vec3 nrP = normalize(reflect(nvP, normal));
  vec3 refractP = refract(nvP, normal, ri);

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  float dither = bayer_32x32(uvJittering, resolution);
  float dither2 = hash(uvJittering * resolution);

  color.a = float(!isSky) + albedo.a;

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
    float specularity = pow((1.0 - g) * clamp01(d), SpecularityReflectionPower);

    //color.rgb = vec3(1.0 - clamp01(d));

    vec3 halfVector  = vP.xyz - svP;

    float transBlockDepthMax = 1.0 * mix(1.0, 0.125, float(id == 106));
    float transBlockDepth = min(transBlockDepthMax, max(length(halfVector), vdoth));

    //vec3 iOut = refract(nvP, normal, 1.000293 / IOR);
    //vec3 iIn = refract(iOut, normalBackFace, IOR / 1.000293);

/*
    vec3 iIn = refract(nvP, normalize(nvP + normalBackFace), 1.000293 / IOR);
         iIn *= max(transBlockDepth, 1.0 - clamp01(dot(nvP, normalize(nvP + normalBackFace))));
         iIn *= 1.0 - float(isWater);
    vec3 iOut = refract(iIn, normalBackFace, IOR / 1.000293);
         iOut *= max(transBlockDepth, 1.0 - clamp01(dot(iIn, normalBackFace)));
         iOut = mix(iOut, -refract(nvP, normal, 1.000293 / IOR), float(isWater));
*/



         //iIn *= pow5(1.0 - clamp01(dot(iIn, normal)));

    vec3 customP = mat3(gbufferModelView) * (wP.xyz + (cameraPosition - (vec3(38.5, 86.5, 175.5))));
    vec3 customN = normal;//normalize(vP.xyz * vec3(-1.0, 1.0, 1.0));

    //float(floor((length(halfVector) / transBlockDepthMax) + 0.05) == 0.0)
    vec3 iIn = refract(nvP, normal, 1.000293 / IOR);
    vec3 iOut = refract(normalize(iIn), normalize(normal + normalize(normal - nvP) * transBlockDepthMax), 1.000293 / IOR);
    vec3 iR = iIn - (iOut - iIn) * clamp01(floor(length(halfVector) / transBlockDepthMax) - float(isWater) * 1000);
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
         rcoord += (vec2(dither2, dither) * 2.0 - 1.0) * pixel * roughnGlass;
    //if(floor(rcoord) != vec2(0.0) || (texture2D(composite, texcoord).b > 0.519 && texture2D(composite, texcoord).b < 0.521)) rcoord = clamp(rcoord + (vec2(dither2, 1.0 - dither) - 0.5) * pixel * (1.0 + alpha * 7.0), pixel, 1.0 - pixel);

    vec4 color2 = texture2D(gaux2, rcoord);
    //color2.rgb = vec3(pow5(clamp01(dot())))

    //color2.rgb = f * F(F0, pow5(1.0 - clamp01(dot(normalize(wi), normal))));

    for(int i = 0; i < 8; i++){
      float r = (float(1 + i) * 2.0 * 3.14159 + dither) / 8.0;

      color2.rgb += texture2D(gaux2, rcoord + vec2(cos(r), sin(r)) * pixel * roughnGlass).rgb;
    }

    color2.rgb *= 1.0 / 9.0;

    float tranBlockR = texture2D(gcolor, rcoord).a;
    vec4 solidBlockSpecular = texture2D(gaux3, rcoord);

    color.rgb = color2.rgb;

    vec3 transBlockAlbedo = albedo.rgb * (1.0 - metallic * alpha);
         transBlockAlbedo = rgb2L(transBlockAlbedo);

    vec4 sunDirctLighting = CalculateShading(shadowtex0, wP, true);
    vec3 shading = mix(vec3(1.0), sunDirctLighting.rgb, sunDirctLighting.a);

    vec3 sunLighting = BRDF(transBlockAlbedo * (0.001 + 0.999 * alpha), normalize(shadowLightPosition), -nvP, normal,  roughness, metallic, F0) * shading * sunLightingColorRaw * pow(fading, 5.0) * 3.0;

    vec3 torchLightingColor = rgb2L(vec3(1.022, 0.782, 0.344));
    vec3 heldTorchLighting = vec3(0.0);
    if(heldBlockLightValue > 1) heldTorchLighting += BRDF(transBlockAlbedo * (0.001 + 0.999 * alpha), -nvP, -nvP, normal, (roughness), metallic, F0) * pow2(clamp01((heldBlockLightValue - 5.0 - length(vP.xyz)) * 0.1)) * torchLightingColor;

    vec3 transBlockColor = transBlockAlbedo * skyLightingColorRaw;
         //transBlockColor += sunLighting;

    transBlockColor = L2rgb(transBlockColor);
    transBlockColor *= 1.0 - metallic;

    vec3 halfVectorR = nvec3(gbufferProjectionInverse * nvec4(vec3(rcoord, texture2D(depthtex0, rcoord).x) * 2.0 - 1.0)) - nvec3(gbufferProjectionInverse * nvec4(vec3(rcoord, texture2D(depthtex1, rcoord).x) * 2.0 - 1.0));
    float hVdistance = max(length(halfVector) * (1.0 - texture2D(gcolor, rcoord).a), length(halfVectorR));
          hVdistance = min(hVdistance, transBlockDepthMax + 254.0 * float(isWater));

    float absorptionFactor = (1.0 - minComponent(albedo.rgb)) * (maxComponent(albedo.rgb));
    vec3 absorption = mix(vec3(1.0), albedo.rgb, absorptionFactor);
    //

    float FD90 = 0.5 + 2.0 * roughness * ndoth * ndoth;

    float scatteringFactor = (clamp01(d) * hVdistance * alpha / Pi * 8.0);
          scatteringFactor = 1.0 - clamp01(exp(-scatteringFactor));
          scatteringFactor = max(clamp01((alpha - 0.95) * 20.0), scatteringFactor);

    vec3 scattering = transBlockAlbedo * skyLightingColorRaw * skyLightMap * (clamp01(dot(normalize(upPosition), normal)) * 0.3 + 0.7);
         //scattering += transBlockAlbedo * skyLightingColorRaw * sunLightingColorRaw * fading * 1.0 * skyLightMap;
         scattering += transBlockAlbedo * torchLightingColor * torchLightMap * 0.06;
         //scattering += (sunLighting + heldTorchLighting) * (1.0 - scatteringFactor);
         scattering = L2rgb(scattering);

    //scatteringFactor = max(scatteringFactor, 1.0 - clamp01(length(refractP)));
    //if(isEyeInWater > 0) scatteringFactor = 0.0;

    vec3 solidBlockColor = color.rgb;/*
    color.rgb = clamp01(
                         F(vec3(alpha), pow5(1.0 - clamp01(dot(-nvP, h))))
                       //* F(vec3(alpha), pow5(1.0 - clamp01(dot(normalize(wi), h))))
                       //* hVdistance
                       /// (0.004 + clamp01(d) * Pi * 1.0)
                       );
*/
    //color.rgb = mix(albedo.rgb, color.rgb, clamp01(length(refractP)));

    //color.rgb = mix(absorption, scattering, scatteringFactor);
    //scatteringcolor.rgb = absorption;

    color.rgb = mix(color.rgb, scattering, scatteringFactor);
    color.rgb *= absorption;

    //color.rgb = vec3(FD90) * 0.1;

    //color.rgb = mix(color.rgb * absorption, scattering, clamp01(F(vec3(alpha), pow5(1.0 - clamp01(dot(-nvP, h)))) * F(vec3(alpha), pow5(1.0 - clamp01(dot(normalize(wi), h)))) * hVdistance / (0.004 + clamp01(d) * Pi * 1.0)));

    //color.rgb = mix(color.rgb, transBlockColor, alpha);
    color.rgb *= (1.0 - f * clamp01(g * d));
    color.rgb = rgb2L(color.rgb);
    color.rgb += sunLighting + heldTorchLighting;
    color.rgb = L2rgb(color.rgb);

    //roughnGlass = (abs(dot(h, -nvP)) * abs(dot(h, nrP))) / (abs(dot(normal, -nvP)) * abs(dot(normal, nrP)));
    //roughnGlass = (pow2(ri) * (1.0 - vdoth) * g * d) / pow2((ri) * clamp01(pow2(dot(-nvP, h)) + (ro) * clamp01(dot(nrP, h))));
    //roughnGlass = clamp01(roughnGlass * (1.0 + hVdistance));

    //color.rgb = vec3(roughnGlass);

    //color.rgb = vec3(clamp01(roughnGlass));

    //color.rgb = vec3(clamp01(dot(nvP, normalize(nvP + normalBackFace))));

    vec4  particlesColor = texture2D(gaux1, texcoord);
    float particlesDepth = length(nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture2D(gaux3, texcoord).z) * 2.0 - 1.0)));

    if(length(vP.xyz) * 1.05 - particlesDepth - 0.05 > 0.0 && particlesColor.a > 0.0){
      //particlesColor.rgb /= mix(1.0, 0.04 + particlesColor.a, emissive);

      emissive = texture2D(gaux3, texcoord).r;

      color.rgb = mix(color.rgb, particlesColor.rgb, particlesColor.a * (1.0 - emissive));
      color.rgb += particlesColor.rgb * emissive;
      //color.rgb = texture2D(gaux3, texcoord).xxx;
      //color.a = texture2D(gaux3, texcoord).x * (1.0 - r.a) + particlesColor.a * 1.0;
    }

    //color.rgb = vec3(float(floor((length(halfVector) / transBlockDepthMax) + 0.05) == 0.0));

    //color.rgb = vec3(pow5(1.0 - clamp01(dot(-normalize(customP), normal))));

    //color.rgb = vec3((1.0 - clamp01(dot(-nvP, normalize(normal - nvP * (transBlockDepthMax))))));

    //color.rgb = vec3(transBlockDepth);
    //color.rgb = vec3(scatteringFactor) * 0.1;
    //color.rgb = albedo.rgb;

    //color.rgb = shading;

    //color.rgb = vec3(min(alpha, color.a));
    //color.rgb = skyColor;
    //color.rgb = texture2D(gaux3, vec2(0.5)).rgb * 0.3;
  }

  //if(texcoord.x * resolution.x < (128.0)){
    //vec4 waterFogColor = vec4(rgb2L(fogColor), 0.04);

    //waterFogColor.a = max(waterFogColor.r, waterFogColor.g) * pow2(1.0 - waterFogColor.b) * 2.0 + waterFogColor.a;
    //waterFogColor.rgb = mix(waterFogColor.rgb * vec3(1.0, 1.0, 0.82), vec3(1.0), pow(clamp01(waterFogColor.b - max(waterFogColor.g, waterFogColor.r)), 0.5)) * (skyLightingColorRaw + pow5(fading) * sunLightingColorRaw * waterFogColor.a);
    //waterFogColor.rgb = waterFogColor.rgb * (skyLightingColorRaw);
    //waterFogColor.rgb = L2rgb(waterFogColor.rgb);

    //color.rgb *= vec3(floor(biomeRainFall * 255) / 255 > 0.5);
  //}

/*
  if(isEyeInWater == 1){
    float hVdistance = length(vP.xyz);
          hVdistance = min(hVdistance, 255.0);
          hVdistance = 1.0 - pow5(clamp01(exp(-hVdistance * 0.02)));

    vec3 absorption = color.rgb * mix(vec3(1.0), albedo.rgb, hVdistance);

    //color.rgb = absorption;
  }
*/
  /*
  float scatteringFactor = -min(512.0, length(vP.xyz));

  float a = 1.0 - pow5(clamp01(exp(scatteringFactor * pow2(0.017))));

  float h1 = clamp01((-(wP.y * mix(1.0, 1.0 / length(vP.xyz) * max(far, 256.0), float(isSky)) + cameraPosition.y) + 90.0) / 24.0);
  float h2 = (clamp01((-(cameraPosition.y) + 90.0) / 24.0));

  vec3 sP2 = normalize(sunPosition + (mat3(gbufferModelView) * vec3(50.0, 0.0, 0.0)));
  float d = abs( dot(vec3(0.3333), L2rgb(sunLightingColorRaw))
          - dot(vec3(0.3333), L2rgb(CalculateSky(sP2, mat3(gbufferModelViewInverse) * sP2, cameraPosition.y, 0.375))) );
        d = (clamp01(50 * (d) * pow2(1.0 - clamp01(dot(normalize(upPosition), normalize(sunPosition))))));

  float w = (1.0 - pow5(clamp01(exp(scatteringFactor * pow2(0.05))))) * (h1 + h2) / 2.0;
  vec3 rays = CalculateRays(wPJittering).rgb;

  if(!isSky) color.rgb = mix(color.rgb, skyLightingColorRaw + sunLightingColorRaw * rays, a);
  else color.rgb += sunLightingColorRaw * rays * a * clamp01((-(wP.y * mix(1.0, 1.0 / length(vP.xyz) * max(far, 256.0), float(isSky)) + cameraPosition.y) + 256.0) / 128.0);

  color.rgb = mix(color.rgb, (skyLightingColorRaw + 1.0 + sunLightingColorRaw * rays) * 0.33, (w) * d);
  */
  //color.rgb = vec3(dot(sP2, normal));

  //color.rgb = vec3(dot(normalize(mat3(gbufferModelView) * vec3(0.5, 0.0, 0.0) + normalize(sunPosition)), normal));
  //color.rgb = vec3(d);
  //color.rgb = vec3(max(h1, h2));

  //color.rgb = vec3(dot(mat3(gbufferModelView) * vec3(1.0, 0.0, 0.0), normal));
  //color.rgb = vec3();

  //color.a = scatteringFactor;
  //color.rgb = CalculateRays(wPJittering).rgb * 0.1;

  //if(texture2D(gaux3, texcoord).x * 1024.0 - length(vP.xyz) < 1.5) color.rgb = vec3(1.0, 0.0, 0.0);

  //color = L2rgb(color);

  //color.rgb = texture2D(gaux3, texcoord).rgb;

/* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color.rgb, color.a);
}

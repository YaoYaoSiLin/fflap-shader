#version 130

#extension GL_EXT_gpu_shader4 : require
#extension GL_EXT_gpu_shader5 : require

//#define Enabled_SSAO
    #define SSAO_Scale 0.5  //[0.5 0.70710677 1.0]

#define Enabled_ScreenSpace_Shadow
//#define Fast_Normal

const int   noiseTextureResolution  = 64;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex5;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D depthtex2;

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
uniform int frameCounter;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowProjectionInverse;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / vec2(viewWidth, viewHeight);

#define gaux2 colortex5

#include "libs/common.inc"
#include "libs/dither.glsl"
#include "libs/jittering.glsl"

#define CalculateHightLight 1
#define CalculateShadingColor 1

#define Void_Sky

#include "libs/brdf.glsl"
#include "libs/light.glsl"
#include "libs/atmospheric.glsl"

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
/*
float noise(vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = smoothstep(0.0, 1.0, f);

    vec2 uv = (p.xy+vec2(37.0, 17.0)*p.z) + f.xy;
    float v1 = texture2D( noisetex, (uv) / noiseTextureResolution, -100.0 ).x;
    float v2 = texture2D( noisetex, (uv + vec2(37.0, 17.0)) / noiseTextureResolution, -100.0 ).x;
    return mix(v1, v2, f.z);
}
*/
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

/*
vec3 CalculateRays(in vec4 wP, in bool isSky){
  vec3 light = vec3(0.0);
  int steps = 8;

  float dither = bayer_32x32(texcoord + haltonSequence_2n3[int(mod(frameCounter, 8))], resolution);

  if(!isSky){
    vec4 rP = wP;

    //float dither = bayer_16x16(texcoord, vec2(viewWidth, viewHeight));

    vec3 nwP = rP.xyz - vec3(0.0, playerEyeLevel, 0.0);
         nwP = normalize(nwP) * length(nwP) / float(steps) * 0.62;

    float vlCount = 0.0;

    for(int i = 0; i < steps; i++){
      rP.xyz -= nwP.xyz;
      vec4 rP2 = rP + vec4(nwP, 0.0) * fract(dither);

      float l = length(rP2.xyz);

      rP2.xyz = wP2sP(rP2);
      float shadow = float(texture2D(shadowtex1, rP2.xy).z + 0.0001 > rP2.z);

      light += vec3(shadow) * clamp01(l / 128.0 + 0.03);//mix(vec3(shadow), shadowColor.rgb, shadowAlp * shadow);
      //vlCount += 1.0;
    }

    //if(vlCount > 0.0) light /= vlCount;
    light /= 8.0;

    //light = min(light, vec3(1.0));
  }else{
    light = vec3(1.0);
  }

  return light;
}
*/
void main() {
  float depth         = texture2D(depthtex0, texcoord).x;
  float particlesDepth = texture2D(colortex4, texcoord).a;
  if(linearizeDepth(depth) > linearizeDepth(particlesDepth) && particlesDepth > 0.0) depth = particlesDepth;
  vec4  vP            = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0); vP /= vP.w;

  vec4 albedo = texture2D(colortex0, texcoord);

  vec2 lightMap = texture2D(colortex1, texcoord).xy;
  float torchLightMap = lightMap.x * 15.0;
  float skyLightMap = max(pow2(lightMap.y) * 1.004 - 0.004, 0.0);

  int id = int(round(texture2D(colortex1, texcoord).z * 255));
  bool isSky = id == 255;
  bool isParticles = false;

  vec3 screenCenterVector = nvec3(gbufferProjectionInverse * nvec4(vec3(vec2(0.5), particlesDepth * 2.0 - 1.0)));

  vec2 normalData = texture2D(colortex2, texcoord).xy;
  if(texture2D(colortex2, texcoord).a < 0.2) normalData.xy = normalEncode(normalize(screenCenterVector * vec3(1.0, 0.0, -1.0)));
  vec3 normal = normalDecode(normalData);

  #ifndef Fast_Normal
    vec2 normalExct = texture2D(gaux2, texcoord).xy;
    vec3 sunVisibleNormal = normal;
    if(texture(colortex1, texcoord).z < 0.5) sunVisibleNormal = normalDecode(normalExct.xy);
  #endif

  float preShadow = texture2D(colortex2, texcoord).z;

  float smoothness = texture2D(colortex3, texcoord).r;
        //smoothness *= smoothness;
  float metallic   = texture2D(colortex3, texcoord).g;
  float emissive   = texture2D(colortex3, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec4 wP = gbufferModelViewInverse * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(normalize(vP.xyz), normal);
  vec3 nrP = normalize(rP.xyz);

  vec2 uvJ = texcoord + R2sq2[int(mod(frameCounter, 16))] * pixel;
  vec4 vPJ = GetViewPosition(uvJ, depthtex1);
  vec4 wPJ = (gbufferModelViewInverse) * vPJ;

  vec3 sP = normalize(mat3(gbufferModelViewInverse) * sunPosition);

  vec3 torchLightingColor = rgb2L(vec3(1.022, 0.768, 0.334) - 0.1);

  vec3 color = vec3(0.0);

  vec3 shading = vec3(1.0);
  float ao = 1.0;

  vec4 skySpecularReflection = vec4(0.0);

  //if(particlesDepth >= blockDistance - 0.05 || particlesDepth < 0.05){
  //  albedo.rgb = mix(albedo.rgb, particlesColor.rgb, particlesColor.a);
  //  albedo.a = albedo.a * (1.0 - particlesColor.a) + particlesColor.a;
  //}

  albedo.rgb = pow(albedo.rgb, vec3(2.2));
  albedo.a = 0.0;

  if(!isSky){
    albedo.a = 1.0;

    //vec3 sunLightingColor = CalculateFogLighting(sunLightingColorRaw, playerEyeLevel + cameraPosition.y, wP.y + cameraPosition.y, sunLightingColorRaw, skyLightingColorRaw, fading);
    //vec3 skyLightingColor = CalculateFogLighting(skyLightingColorRaw, playerEyeLevel + cameraPosition.y, wP.y + cameraPosition.y, sunLightingColorRaw, skyLightingColorRaw, fading);

    #ifdef Enabled_SSAO
    //vec4 aoSample = textureGather(colortex5, texcoord * SSAO_Scale);
    //ao = pow(dot(aoSample, vec4(0.25)), 2.0);
    #endif

    vec4 sunDirctLighting = CalculateShading(shadowtex1, shadowtex0, wP, clamp01(dot(nvP, normalize((shadowLightPosition) + vP.xyz))) + (1.0 - max(0.0, dot(normalize(shadowLightPosition), normal))));
/*
    vec3 o = mat3(gbufferModelView) * wP.xyz;

    vec3 rayPosition = o.xyz;
    vec3 rayDirection = normalize(shadowLightPosition) * 0.125 * length(shadowLightPosition) / 240;

    rayPosition -= rayDirection * 0.05;

    float lastDepth = (texture2D(depthtex0, texcoord).x);

    for(int i = 0; i < 8; i++){
      rayPosition += rayDirection * 1.0;

      vec2 newUV = nvec3(gbufferProjection * nvec4(rayPosition)).xy * 0.5 + 0.5;
      float newDepth = texture2D(depthtex0, newUV).x;

      //if(floor(newUV) == vec2(0.0)){

      //vec3 newP = nvec3(gbufferProjectionInverse * nvec4(vec3(newUV, texture2D(depthtex0, newUV).x) * 2.0 - 1.0));
      //color = texture2D(colortex0, newUV.xy).rgb;
      //color = vec3(length(newP)) * 0.01;
      //if(texcoord.x < 0.5) color = vec3(length(o)) * 0.01;

      //color = vec3(length(newP) - length(o) + 0.05 > 0.0 && length(newP) - length(o) + 0.05 < 1.0);
      //if(length(newP) - length(o) > 0.0 && length(newP) - length(o) < 0.05) {uv = newUV; lastDepth = newDepth;}
      lastDepth = min(lastDepth, newDepth);
    }

*/


    shading = mix(vec3(clamp01((pow5(skyLightMap) - 0.2) * 12.5)), sunDirctLighting.rgb, sunDirctLighting.a) * preShadow;

    //vec3 sunLighting = albedo.rgb * sunLightingColorRaw * pow(fading, 5.0) * shading * clamp01(dot(normalize(shadowLightPosition), normal));

    vec3 sunLighting = BRDF(albedo.rgb, normalize(shadowLightPosition), -nvP, normal, sunVisibleNormal, roughness, metallic, F0) * shading * sunLightingColorRaw * fading * SunLight;
    vec3 heldLighting = vec3(0.0);

    if(heldBlockLightValue > 1){
      heldLighting = BRDF(albedo.rgb, -nvP, -nvP, normal, normal, roughness, metallic, F0);
      //heldLighting *= pow2(clamp01((heldBlockLightValue - 1.0 - length(vP.xyz)) * 0.1));
      heldLighting *= pow2(max(0.0, heldBlockLightValue - 4.0 - length(vP.xyz)) * 0.0909);
      //vec3 lightPosition = mat3(gbufferModelView) * (wP.xyz + cameraPosition - vec3(-79.5, 79.25, 47.5));
      //     lightPosition = vP.xyz * vec3(-1.0, 1.0, 1.0);
      //heldLighting *= vec3(ScreenSpaceShadow(-lightPosition, vP.xyz));
    }

    //vec3 skyLighting = albedo.rgb * (clamp01(dot(normalize(upPosition), normal) * 0.5 + 0.5) * 0.85 + 0.15) * ao;
    vec3 skyLighting  = albedo.rgb * clamp01(dot(normalize(upPosition), normal));
         skyLighting += albedo.rgb * abs(dot(normalize(reflect(mat3(gbufferModelView) * vec3(1.0, 0.0, 0.0), normalize(upPosition))), normal)) * 0.48 * ao;
         skyLighting += albedo.rgb * abs(dot(normalize(reflect(mat3(gbufferModelView) * vec3(0.0, 0.0, 1.0), normalize(upPosition))), normal)) * 0.48 * ao;
         skyLighting += albedo.rgb * abs(dot(normalize(reflect(mat3(gbufferModelView) * vec3(0.0, 1.0, 0.0), normalize(upPosition))), normal)) * 0.21 * ao;
         skyLighting *= skyLightMap * skyLightingColorRaw * 0.92;

    vec3 nlP = normalize(shadowLightPosition);
    vec3 nuP = normalize(upPosition);
    vec3 zPlane = mat3(gbufferModelView) * vec3(0.0, 0.0, 1.0);

    vec3 fakeGI  = albedo.rgb * clamp01(dot(normalize(reflect(nlP, nuP)), normal)) * clamp01(pow5(dot(nlP, nuP)));
         fakeGI += albedo.rgb * clamp01(dot(reflect(nlP, zPlane), normal)) * clamp01(pow5(dot(nlP, zPlane)));
         fakeGI *= sunLightingColorRaw * skyLightMap * skyLightMap * ao * step(-0.5, dot(normal, nuP)) * 0.5;
         //fakeGI += albedo.rgb * clamp01(dot(normalize(vP.xyz - shadowLightPosition), normal)) * 0.09;
         //fakeGI *= (skyLightingColorRaw + sunLightingColorRaw * pow(fading * clamp01(dot(sP, vec3(0.0, 1.0, 0.0))), 3.0)) * skyLightMap * skyLightMap * ao;

    color = sunLighting;

    vec3 diffuse  = skyLighting;
         diffuse += fakeGI;
         //diffuse += torchLighting;

    vec3 h = normalize(nrP - nvP);

    float ndotv = 1.0 - clamp01(dot(-nvP, normal));
    float vdoth = 1.0 - clamp01(dot(-nvP, h));
    float ndoth = clamp01(dot(normal, h));
    float ndotl = clamp01(dot(nrP, normal));

    vec3  f = F(F0, pow5(vdoth));
    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);

    float brdf = clamp01(g * d);

    diffuse *= rgb2L(vec3(1.0 - metallic));
    diffuse *= rgb2L(1.0 - f * brdf);

    //torchLightMap = max(0.0, pow5(lightMap.x) * 15.0 - 0.005 + max(0.0, pow5(lightMap.x) * 15.0 - 5.0) * 0.3);
    //torchLightMap = (pow2(lightMap.x) * 15.0 - 0.007) * (1.0 - metallic) * (1.0 + (1.0 - pow2(dot(normalize(upPosition), normal))));
    //torchLightMap += max(0.0, pow5(lightMap.x - 0.14 * d) / 0.04 - 0.07);
    //torchLightMap = max(torchLightMap, 0.0) * 0.25;

    torchLightMap = lightMap.x - 0.0625;
    torchLightMap = pow5(torchLightMap) * 15.0 * (1.0 - metallic) * pow2((1.0 - abs(dot(normalize(upPosition), normal))) * 0.3 + 0.7) + pow5(torchLightMap * torchLightMap - 0.0625 * d) * 15.0;

    vec3 torchLighting = albedo.rgb * torchLightMap * 0.0549;
         torchLighting = (heldLighting * 0.7 + torchLighting) * torchLightingColor * 1.42;

    color += diffuse + max(vec3(0.0), torchLighting - color * 0.33);

    color += albedo.rgb * emissive;

    //color = vec3(sss) * 0.1;

    //color = vec3(step(length(vP.xyz), length(rePVector)) * 0.1);

//color = torchLighting;
    //color = vec3(preShadow) * 0.1;

    //color = vec3(texture2D(shadowcolor0, texcoord).zzz * 0.1);
    //color = vec3(pow5(texture2D(shadowcolor0, texcoord).z));
  }

  //color = particlesColor.rgb;
  //color = vec3(clamp01(length(vP.xyz) - length(particlesDepth)));

  if(albedo.a < 1.0){
  vec3 skyWorldVector = mat3(gbufferModelViewInverse) * nvP;

  vec3 atmosphericScattering = CalculateSky(nvP, sP, 500.0, 1.0);
       //atmosphericScattering = CalculateAtmosphericScattering(atmosphericScattering, -skyWorldVector.y + 0.1);
       atmosphericScattering = CalculateSun(atmosphericScattering, nvP, normalize(sunPosition), sunLightingColorRaw, skyWorldVector.y);
  color.rgb = mix(atmosphericScattering, color.rgb, albedo.a);
  }

  //color += emissive * mix(albedo.rgb, particlesColor.rgb, float(isParticles));
  //color = mix(color, particlesColor.rgb / (0.04 + particlesColor.a), emissive * float(isParticles));
  //particlesColor.a = max(particlesColor, emissive);

  //color.rgb = albedo.aaa;

  //color += particlesColor.rgb * emissive * 0.7 / (0.04 + particlesColor.a) * float(isParticles);

  color = L2rgb(color);

  if(!isSky){
    color += albedo.rgb * vec3(0.2874, 0.2848, 0.4278) / (albedo.rgb + 0.001) * getLum(clamp01(albedo.rgb - color)) * 0.5 * nightVision;
  }

  //color = vec3(fading);
  //if(fading > 0.99) color = vec3(1.0, 0.0, 0.0);

  //color = vec3(dot(nvP, normalize((shadowLightPosition) + vP.xyz)));

  color *= 1.0 / overRange;



/*
  vec3 o = mat3(gbufferModelView) * wP.xyz;

  vec3 rayPosition = o.xyz;
  vec3 rayDirection = normalize(shadowLightPosition) * 0.125 * length(shadowLightPosition) / 240;

  rayPosition -= rayDirection * 0.05;

  float lastDepth = (texture2D(depthtex0, texcoord).x);

  for(int i = 0; i < 8; i++){
    rayPosition += rayDirection * 1.0;

    vec2 newUV = nvec3(gbufferProjection * nvec4(rayPosition)).xy * 0.5 + 0.5;
    float newDepth = texture2D(depthtex0, newUV).x;

    //if(floor(newUV) == vec2(0.0)){

    //vec3 newP = nvec3(gbufferProjectionInverse * nvec4(vec3(newUV, texture2D(depthtex0, newUV).x) * 2.0 - 1.0));
    //color = texture2D(colortex0, newUV.xy).rgb;
    //color = vec3(length(newP)) * 0.01;
    //if(texcoord.x < 0.5) color = vec3(length(o)) * 0.01;

    //color = vec3(length(newP) - length(o) + 0.05 > 0.0 && length(newP) - length(o) + 0.05 < 1.0);
    //if(length(newP) - length(o) > 0.0 && length(newP) - length(o) < 0.05) {uv = newUV; lastDepth = newDepth;}
    lastDepth = max(lastDepth, newDepth);
  }

  vec3 shadowPosition = wP2sP(wP);
  color = vec3(lastDepth > depth);
*/
  /*
  float dither = bayer_32x32(texcoord + haltonSequence_2n3[int(mod(frameCounter, 16))], resolution);

  vec3 o = vP.xyz;

  vec3 rayPosition = vP.xyz;
  vec3 rayDirection = normalize(shadowLightPosition) * 0.125 * length(shadowLightPosition) / 240;

  rayPosition -= rayDirection * 0.05;
  */
       //rayDirection *= 0.001;

  //vec2 uv = nvec3(gbufferProjection * nvec4(rayPosition + rayDirection)).xy * 0.5 + 0.5;


  //color = vec3(length(rayPosition - rayDirection)) * 0.006;
/*
  float ssShading = 0.0;

  float lastDepth = (depth);
  vec2 uv = texcoord;

  for(int i = 0; i < 8; i++){
    rayPosition += rayDirection * 1.0;

    vec2 newUV = nvec3(gbufferProjection * nvec4(rayPosition)).xy * 0.5 + 0.5;
    float newDepth = texture2D(depthtex0, newUV).x;

    //if(floor(newUV) == vec2(0.0)){

    vec3 newP = nvec3(gbufferProjectionInverse * nvec4(vec3(newUV, texture2D(depthtex0, newUV).x) * 2.0 - 1.0));
    //color = texture2D(colortex0, newUV.xy).rgb;
    //color = vec3(length(newP)) * 0.01;
    //if(texcoord.x < 0.5) color = vec3(length(o)) * 0.01;

    //color = vec3(length(newP) - length(o) + 0.05 > 0.0 && length(newP) - length(o) + 0.05 < 1.0);
    //if(length(newP) - length(o) > 0.0 && length(newP) - length(o) < 0.05) {uv = newUV; lastDepth = newDepth;}
    lastDepth = min(lastDepth, newDepth);
  }
*/
    //color = vec3(length(nvec3(gbufferProjectionInverse * nvec4(vec3(uv, lastDepth) * 2.0 - 1.0)))) * 0.01;
    //color = vec3(lastDepth);

    //if(length(newP) > length(o) && length(newP) - length(o) < 3) ssShading = 1.0;
    //if(texture2D(depthtex0, newUV).x)

    //if(lastDepth < LinearlizeDepth(texture2D(depthtex0, newUV).x)){
    //ssShading = 1.0;
    //lastDepth = LinearlizeDepth(texture2D(depthtex0, newUV).x);
    //}
    //}

    //ssShading = float(newUV.z > depth);
    //vec2 screenUV =
    //float newDepth = nvec3(gbufferProjection)

    //ssShading += float(length(rayPosition - o) / length(o));
  //}

  //color = vec3(ssShading);

  //color = texture2D(shadowcolor0, shadowUV.xy).rgb * 0.33;
  //color = vP.xyz + wP.xyz;

  //if(texcoord.x > 0.5) color = vec3(blockDistance * 1.01 - particlesDepth + 0.05 >= 0.0 && particlesColor.a > 0.001);

  //color = texture2D(colortex5, texcoord).rgb;

  //if(1.0 - min(1.0, abs(id - 1.0)) == 1.0)  color = vec3(1.0, 0.0, 0.0);
  //if(texture2D(colortex1, texcoord).a > 0.5) color = vec3(1.0, 0.0, 0.0);
  //color = vec3(max(0.0, (1.0 - texture2D(colortex1, texcoord).a) - 0.7)) * 3.33;
  //if(texcoord.x < 0.5)color = vec3(texture2D(colortex1, texcoord).a < 0.1);
  //color = texture2D(colortex1, texcoord).aaa;

  //color = texture2D(shadowcolor0, texcoord).rgb;
  //color = albedo.rgb * vec3(texture2D(colortex2, texcoord).b);

  //if(particlesDepth >= blockDistance - 0.05) color = vec3(0.0);
  //color = vec3(clamp01(blockDistance - particlesDepth));

/* DRAWBUFFERS:0125 */
  gl_FragData[0] = vec4(albedo.rgb, 0.0);
  gl_FragData[1] = vec4(torchLightMap / 15.0, skyLightMap, texture2D(colortex1, texcoord).ba);
  gl_FragData[2] = vec4(normalData.xy, preShadow, 1.0);
  gl_FragData[3] = vec4(color, 1.0);
}
                                                                                           

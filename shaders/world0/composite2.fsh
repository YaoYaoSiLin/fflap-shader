#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform mat4 shadowProjectionInverse;

uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

uniform float viewWidth;
uniform float viewHeight;

uniform int frameCounter;
uniform int isEyeInWater;

in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

in vec2 texcoord;

/* DRAWBUFFERS:25 */

vec2 resolution = vec2(viewWidth, viewHeight);

#define Gaussian_Blur
#define CalculateHightLight 1

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/brdf.glsl"
#include "../libs/light.glsl"

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

vec4 CalculateSunDirectLighting(in vec3 viewPosition, in vec3 normal, inout float sss){
    float viewLength = length(viewPosition);
    if(viewLength > shadowDistance) return vec4(1.0);

    vec4 shadowPosition = gbufferModelViewInverse * vec4(viewPosition, 1.0);
         shadowPosition = shadowModelView * shadowPosition;
         shadowPosition = shadowProjection * shadowPosition;
         shadowPosition /= shadowPosition.w;
         shadowPosition = shadowPosition;

    float distort = 1.0 / (mix(1.0, length(shadowPosition.xy), SHADOW_MAP_BIAS) / 0.95);
    vec3 shadowCoord = shadowPosition.xyz * vec3(vec2(distort), 1.0) * 0.5 + 0.5;

    float d = texture(shadowtex1, shadowCoord.xy).x;
    float d0 = texture(shadowtex0, shadowCoord.xy).x;

    float shading = step(shadowCoord.z, d + shadowPixel * 2.0);

    sss = 1.0 - clamp01(exp(d0 * 128.0) * exp(-shadowCoord.z * 128.0) - 0.0);

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

vec3 CalculateTranslucent(in vec3 rayOrigin, in vec3 normal, in float rayMax, in vec4 albedo, in float IOR){
  float mask = round(texture2D(gdepth, texcoord).z * 255.0);
  bool isSky = bool(step(254.5, mask));
  bool isWater = CalculateMaskID(8, mask);
  bool isGlass = CalculateMaskID(20, mask);
  bool isGlassPane = CalculateMaskID(106, mask);

  vec3 rayDirection = normalize(-rayOrigin);

  vec3 opaquePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex1, texcoord).x) * 2.0 - 1.0));
  vec3 translucentPosition = opaquePosition - rayOrigin;

  float blockDepth = min(rayMax, length(translucentPosition));

  float eta1 = 1.000293;
  float eta2 = IOR;

  float eta = eta1 / eta2;
  if(bool(isEyeInWater)) eta = eta2 / eta1;

  //float eta = 1.333;
  //if(isEyeInWater == 0) eta = 1.000293 / IOR;

  vec3 refractDirection = vec3(0.0);
  bool TIR = totalInternalReflection(rayDirection, refractDirection, normal, eta);
  refractDirection *= min(1.0, blockDepth);

  vec2 refracted = nvec3(gbufferProjection * nvec4(rayOrigin.xyz + refractDirection)).xy * 0.5 + 0.5;
  vec3 rP = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(depthtex0, refracted).x) * 2.0 - 1.0));
  vec3 rPO = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(depthtex1, refracted).x) * 2.0 - 1.0));

  blockDepth = max(min(rayMax, length(rPO - rP)), blockDepth);

  vec3 color = texture2D(gaux2, refracted).rgb * overRange;
  if(TIR) color = vec3(0.0);
  //color = vec3(1.0);

  float scatteringFactor = 1.0 - exp(-blockDepth * sqrt(albedo.a) * 0.5);
        scatteringFactor *= float(!isGlass && !isGlassPane);
  color = mix(color, albedo.rgb * G2Linear(skyLightingColorRaw), scatteringFactor);

  color -= color * (1.0 - exp(-(albedo.a * blockDepth + albedo.a * 0.5) * Pi)) * exp(-albedo.rgb);

  float smoothness = texture(gnormal, texcoord).b;
  float metallic = texture(composite, texcoord).b;
  float roughness = pow2(1.0 - smoothness);
  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float g,d;
  vec3 f;

  vec3 reflectDirection = reflect(normalize(rayOrigin), normal);
  vec3 m = normalize(reflectDirection - normalize(rayOrigin));

  FDG(f, d, g, normalize(-rayOrigin), reflectDirection, normal, m, F0, roughness);
  color.rgb *= 1.0 - f;

  if(isWater) albedo.rgb = vec3(0.02);

  color = L2Gamma(color);

  vec3 sunLighting = BRDF(albedo.rgb, normalize(shadowLightPosition), rayDirection, normal, normal, roughness, max(float(isGlass || isGlassPane), metallic), F0);
       sunLighting = L2Gamma(sunLighting);

  float sss = (1.0);
  vec4 shading = CalculateSunDirectLighting(rayOrigin, normal, sss);

  color += sunLighting * sunLightingColorRaw * 4.0 * shading.rgb;

  color = G2Linear(color);

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

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main(){
  vec4 color = texture2D(gaux2, texcoord);
       color.rgb *= overRange;

  float alpha = texture2D(composite, texcoord).x;
  vec4 albedo = texture2D(gcolor, texcoord);

  float depth = texture(depthtex0, texcoord).x;
  vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0));
  vec3 worldPosition = mat3(gbufferModelViewInverse) * viewPosition;

  float viewLength = length(viewPosition);

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);
  vec3 normalSurface = normalDecode(texture2D(composite, texcoord).xy);

  vec3 normalVisible = normal;
  if(!bool(albedo.a) && dot(normalSurface, normalize(viewPosition)) < 0.1) normalVisible = normalSurface;

  float mask = round(texture2D(gdepth, texcoord).z * 255.0);
  bool isSky = bool(step(254.5, mask));
  bool isWater = CalculateMaskID(8, mask);
  bool isGlass = CalculateMaskID(20, mask);
  bool isGlassPane = CalculateMaskID(106, mask);

  float smoothness = texture(gnormal, texcoord).b;
  float metallic = texture(composite, texcoord).b;

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  if(bool(albedo.a) && !isSky){
    float blockDepth = 1.0;
    if(isWater) blockDepth = 255.0;
    if(isGlassPane) blockDepth = 0.125;

    float IOR = 1.333;
    if(isGlass || isGlassPane) IOR = 1.5;

    color.rgb = CalculateTranslucent(viewPosition, normal, blockDepth, vec4(albedo.rgb, alpha), IOR);
  }

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
  color.rgb /= overRange;

  gl_FragData[0] = vec4(smoothness, metallic, normalEncode(normalVisible));
  gl_FragData[1] = color;
}

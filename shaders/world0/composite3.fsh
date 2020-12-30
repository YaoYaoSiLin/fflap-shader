#version 130

#define SSR_Rendering_Scale 0.5

//remove setting: Low
#define Surface_Quality High        //[Medium High]

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex2;

uniform sampler2D noisetex;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform vec3 upPosition;
uniform vec3 shadowLightPosition;
uniform vec3 sunPosition;
uniform vec3 cameraPosition;

uniform vec2 jitter;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform float frameTimeCounter;

uniform int frameCounter;
uniform int moonPhase;
uniform int isEyeInWater;

in vec2 texcoord;
in vec3 normalSample;
in float fading;
in vec3 skyLightingColorRaw;
in vec3 sunLightingColorRaw;
in vec4 eyesWaterColor;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/atmospheric.glsl"
#include "../libs/brdf.glsl"
#include "../libs/specular.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 KarisToneMapping(in vec3 color){
	float a = 0.002;
	float b = float(0x2fff) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-b*luma)/(2.0*a-b-luma));
	return color;
}

void CalculateBRDF(out vec3 f, out float g, out float d, in float roughness, in float metallic, in vec3 F0, in vec3 h, in vec3 n, in vec3 o, in vec3 L){
	//vec3 h = normalize(L + o);

	float ndoth = clamp01(dot(n, h));
	float vdoth = pow5(1.0 - clamp01(dot(o, h)));
	float ndotl = clamp01(dot(L, n));
	float ndotv = 1.0 - clamp01(dot(o, n));

	f = F(F0, vdoth);
	d = DistributionTerm(roughness, ndoth);
	g = VisibilityTerm(d, ndotv, ndotl);
  //c = 4.0 * abs(dot(o, n)) * abs(dot(L, n));
}

void CalculateBRDF(out vec3 f, out float g, out float d, in float roughness, in float metallic, in vec3 F0, in vec3 n, in vec3 o, in vec3 L){
	vec3 h = normalize(L + o);

	float ndoth = clamp01(dot(n, h));
	float vdoth = pow5(1.0 - clamp01(dot(o, h)));
	float ndotl = clamp01(dot(L, n));
	float ndotv = 1.0 - clamp01(dot(o, n));

	f = F(F0, vdoth);
	d = DistributionTerm(roughness, ndoth);
	g = VisibilityTerm(d, ndotv, ndotl);
}

/*
float halton1(in float base){
  float r = 0.0;
  float f = 1.0;

  float i = float(frameCounter);
  int count;

  while(i > 0){
    f /= base;
    r += f * mod(i, base);
    i /= base;

    count++;
    if(count > 10) continue;
  }

  return r;
}
*/

vec4 ImportanceSampleGGX(in vec2 E, in float roughness){
  roughness *= roughness;
  roughness = clamp(roughness, 0.0001, 0.9999);

  float Phi = E.x * 2.0 * Pi;
  float CosTheta = sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  vec3 H = vec3(cos(Phi) * SinTheta, sin(Phi) * SinTheta, CosTheta);
  float D = DistributionTerm(roughness, CosTheta) * CosTheta;

  return vec4(H, 1.0 / D);
}

vec4 CalculateNormal(in vec3 n, in mat3 tbn, in float roughness){
  vec2 offset = -jitter;

  float blueNoise = GetBlueNoise(depthtex2, texcoord, resolution.y * SSR_Rendering_Scale, offset);
  float blueNoise2 = GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y * SSR_Rendering_Scale, offset);

  float steps = 8.0;
  float invsteps = 1.0 / (steps);
  float index = mod(float(frameCounter), steps);

  vec2 E = vec2(blueNoise2, blueNoise);
       E.y = ApplyBRDFBias(E.y);
  vec4 H = ImportanceSampleGGX(E, roughness);

  //PDF = H.w;

  return vec4(normalize(tbn * H.xyz), H.w);
}

vec2 GetClosest(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel / SSR_Rendering_Scale;
      //float neighbor = texture(depthtex0, texcoord).x;
      float neighbor = texture(depthtex0, coord + neighborhood).x;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;

  return closest.xy;
}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

uniform sampler2D shadowtex0;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

#define SHADOW_MAP_BIAS 0.9

void CalculateMappingReflection(inout vec3 color, in vec3 viewPosition, in vec3 normal){
  vec3 rayDirection = reflect(normalize(viewPosition), normal);

  vec3 uv = mat3(gbufferModelViewInverse) * rayDirection; uv = uv.xzy;

  uv.z += near * near * 2.0 * length(uv.xy) * sign(uv.z);
  uv.xy /= 1.0 + length(uv.z);
  uv.xy = (uv.xy * 0.5 + 0.5);

  bool downSide = bool(step(uv.z, 0.0));

  if(downSide) uv.y = 1.0 - uv.y;

  uv.xy *= 0.2;
  uv.x += 0.8;
  uv.y += 0.05;

  if(downSide) uv.y += 0.25;

  //if(uv.x < 0.8 || uv.y > 0.4) return;
  //if(uv.x > 0.8 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 0.4 && texture(shadowtex0, uv.xy).x < 0.999 && length(viewPosition) < 32.0) {
  if(linearizeDepth(texture(shadowtex0, uv.xy).x, far) > 0.15 / far) return;
  vec3 mapReflectionAlbedo = (texture2D(shadowcolor0, uv.xy).rgb);
  //if(dot(mapReflectionAlbedo, vec3(1.0 / 3.0)) >= 0.5) return;
  mapReflectionAlbedo = decodeGamma(mapReflectionAlbedo);

  //vec3 sphereWorldPosition = texture2D(shadowcolor1, uv.xy).xyz * 2.0 - 1.0;//nvec3(gbufferProjectionInverse * nvec4(vec3(uv.xy, texture(shadowtex0, uv.xy).x * far) * 2.0 - 1.0));
  //vec4 sphereShadowCoord = shadowProjection * shadowModelView * nvec4(sphereWorldPosition);
  //     sphereShadowCoord /= sphereShadowCoord.w; 
  //     sphereShadowCoord.xy /= mix(1.0, length(sphereShadowCoord.xy), SHADOW_MAP_BIAS) / 0.95;
  //     sphereShadowCoord = sphereShadowCoord * 0.5 + 0.5;
  //     sphereShadowCoord.xy *= 0.8;
  //if(length(sphereViewPosition) < 8.0) return;

  //color = vec3(step(sphereViewPosition.z, texture(shadowtex0, sphereViewPosition.xy).x + (8.0 / 2048.0)));
  //return;
  vec3 sphereShadowCoord = texture2D(shadowcolor1, uv.xy).xyz;
  float shading = step(sphereShadowCoord.z, texture(shadowtex0, sphereShadowCoord.xy).x + (1.5 / 512.0));
        shading *= saturate((texture2D(shadowcolor0, uv.xy).a * 2.0 - 1.0) * 2.0);

  //vec4 sphereWorldPosition = vec4(sphereViewPosition, texture2D(shadowcolor1, uv.xy).w * 2.0 - 1.0);
  //vec4 sphereShadowCoord = shadowProjection * shadowModelView * sphereWorldPosition; 
  //sphereShadowCoord.z = length(sphereShadowCoord.xyz) * sign(sphereShadowCoord.z) / (far * 2.0) * 2.0 - 1.0;

  //sphereShadowCoord /= sphereShadowCoord.w; sphereShadowCoord = sphereShadowCoord * 0.5 + 0.5;
  


  //if(sphereShadowCoord.z < texture(shadowtex0, sphereShadowCoord.xy * 0.8).x) return;
  //if(length(sphereWorldPosition.xyz / sphereWorldPosition.w) * far * 0.5 * 0.04 < 4.0) return;

  //if(length(sphereViewPosition) > 4.0) return;

  vec3 mapReflectionNormal = mat3(gbufferModelView) * (texture2D(shadowcolor0, uv.xy).xyz * 2.0 - 1.0);

  float encodeLightMap = texture2D(shadowcolor1, uv.xy).a;
  vec2 lightMap = vec2(fract(encodeLightMap * 16.0), floor(encodeLightMap * 16.0) / 15.0);
       lightMap.x = pow5(lightMap.x);
       lightMap.y = min(1.0, pow5(lightMap.y) * 1.4);

  vec3 torchLightingColor = vec3(1.049, 0.5821, 0.0955);

  vec3 skyLight = mapReflectionAlbedo.rgb * lightMap.y * skyLightingColorRaw;
  vec3 sunLight = mapReflectionAlbedo.rgb * sunLightingColorRaw * fading * shading;
  vec3 torchLight = mapReflectionAlbedo.rgb * lightMap.x * torchLightingColor * 0.1;

	color.rgb = (sunLight + skyLight + torchLight) * invPi;
  //}
}

void main() {
  vec2 coord = texcoord;

  //coord -= jittering * pixel;

  float neighborhoodSky = 0.0;

  for(float i = -2.0; i <= 2.0; i += 1.0){
    for(float j = -2.0; j <= 2.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      neighborhoodSky = max(neighborhoodSky, texture(gnormal, coord + neighborhood).b);
    }
  }

  if(bool(step(0.999, neighborhoodSky)))
  coord = GetClosest(coord);

  //coord -= jittering * pixel;

	vec4 albedo = texture2D(gcolor, coord);

  vec2 lightmapPackge = unpack2x8(texture(gdepth, coord).x);
  float torchLightMap = lightmapPackge.x;
  float skyLightMap = lightmapPackge.y;

  int mask = int(round(texture(gnormal, coord).z * 255.0));
  bool isPlants = mask == 18 || mask == 31 || mask == 83;
  bool isParticles = bool(step(249.5, float(mask)) * step(float(mask), 252.5));
  bool isHand = CalculateMaskID(248.0, float(mask));

  vec3 flatNormal = normalDecode(texture2D(gnormal, coord).xy);
	vec3 texturedNormal = normalDecode(texture2D(composite, coord).xy);
  vec3 visibleNormal = texturedNormal;
  if(bool(step(texture2D(gcolor, coord).a, 0.9999))) flatNormal = texturedNormal;

  vec2 specularPackge = unpack2x8(texture(composite, coord).b);
	float smoothness = specularPackge.x;
	float metallic   = specularPackge.y;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, decodeGamma(albedo.rgb), step(0.5, metallic));

	vec3 color = texture2D(gaux2, coord).rgb;

	float depth = texture2D(depthtex0, coord).x;

	vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0));
	vec3 nvP = normalize(vP);
  float viewLength = length(vP);

  vec3 facetoPlayerNormal = normalize(-nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.7) * 2.0 - 1.0)));

  if(isParticles) {
    flatNormal = facetoPlayerNormal;
    texturedNormal = facetoPlayerNormal;
  }

  float ndotv = dot(-nvP, texturedNormal);
  if(bool(step(ndotv, 0.2))) visibleNormal = flatNormal;

  vec3 normal = visibleNormal;
  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

  vec3 upVector = abs(worldNormal.z) < 0.4999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  upVector = mat3(gbufferModelView) * upVector;

	vec3 t = normalize(cross(upVector, normal));
	vec3 b = cross(normal, t);
	mat3 tbn = mat3(t, b, normal);

	vec2 motionDither = jittering;
	//float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale);
  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y * SSR_Rendering_Scale, jitter);

  float rayBRDF;

  vec4 rayPDF = CalculateNormal(normal, tbn, roughness);

  float visibility = dot(-nvP, rayPDF.xyz);
  rayPDF.xyz *= sign(visibility);
  //normal = rayPDF.xyz;

  //if(abs(visibility) > 1e-5) 
  normal = rayPDF.xyz;
  //normal = flatNormal;
  //if(visibility > 0.0) normal = rayPDF.xyz;
  //normal = flatNormal;

	vec3 rayDirection = reflect(nvP, normal);
	vec3 normalizedRayDirection = normalize(rayDirection);

	vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

	vec3 reflection = vec3(0.0);

  vec3 eyePosition = vec3(0.0, cameraPosition.y - 63.0, 0.0);
  vec3 rayDirectionWorld = mat3(gbufferModelViewInverse) * (normalizedRayDirection);

	vec3 skySpecularReflection = CalculateInScattering(eyePosition, rayDirectionWorld, sP, 0.76, ivec2(16, 2), vec3(1.0, 1.0, 0.0));
       skySpecularReflection = ApplyEarthSurface(skySpecularReflection, eyePosition, rayDirectionWorld, sP);
       //skySpecularReflection = encodeGamma(skySpecularReflection);

  if(isEyeInWater == 1) {
    skySpecularReflection = eyesWaterColor.rgb * encodeGamma(skyLightingColorRaw);
  }

  torchLightMap = max(0.0, torchLightMap - 0.0667) * 1.071;
  torchLightMap = pow2(torchLightMap * torchLightMap);

  vec3 torchLightingColor = vec3(1.049, 0.5821, 0.0955);

  vec3 fakeLightingReflection = albedo.rgb;
       fakeLightingReflection *= 1.0 - exp(-(torchLightingColor) * 0.125 * torchLightMap);

  skySpecularReflection *= smoothstep(0.466, 0.866, skyLightMap);
  //skySpecularReflection = mix(color, skySpecularReflection, smoothstep(0.466, 0.866, skyLightMap)) + fakeLightingReflection * 0.0;
/*
  vec2 moonAtlasScale = vec2(textureSize(depthtex2, 0));
  float moonAspectRatio = 1.0 / (moonAtlasScale.x / moonAtlasScale.y);

  vec3 worldSkyPosition = mat3(gbufferModelViewInverse) * nreflectVector;

  vec2 moonPhases = vec2(0.25, 0.25);
       moonPhases.x += 0.5 * mod(moonPhase, 4);
       if(moonPhase > 3) moonPhases.y += 0.5;
       moonPhases.x *= moonAspectRatio;

  vec2 moonDrawPosition = worldSkyPosition.xy;
       moonDrawPosition /= 1.0 + worldSkyPosition.z;
       moonDrawPosition = -(-sP.xy / (1.0 - sP.z) - moonDrawPosition);
       moonDrawPosition *= 1.0 + worldSkyPosition.z;
       moonDrawPosition.x *= moonAspectRatio;

  vec3 moonTexture = texture2D(depthtex2, moonDrawPosition + moonPhases).rgb;
       moonTexture = rgb2L(moonTexture) * 0.4;

  if(floor(moonDrawPosition * vec2(4.0, 2.0) + 0.5) != vec2(0.0) || worldSkyPosition.z < -0.5) moonTexture = vec3(0.0);
  float top = 1.0;
  skySpecularReflection += L2rgb(vec3(1.022, 0.782, 0.344) * dot03(moonTexture) * top);
*/
  float sphereMapRenderDistance = 32.0;
  
  reflection = skySpecularReflection;
  if(bool(step(viewLength, sphereMapRenderDistance))) CalculateMappingReflection(reflection, vP.xyz, normal);

  float rayDepth = 0.999;

	vec4 ssR = vec4(0.0);
	vec3 hitPosition = vec3(0.0);

  if(bool(step(0.5, smoothness)) /* || bool(step(0.2, smoothness) * step(sphereMapRenderDistance, viewLength)) */){
    vec3 rayOrigin = vP + flatNormal * clamp(dot(-nvP, flatNormal) * 4.0, 0.0, 0.1);

    vec3 ssrDirection = rayDirection;
    
    //float thickness = clamp(viewLength * roughness, 1.0, 20.0);
    //ssrDirection *= thickness;

	  ssR = ScreenSpaceReflection(rayOrigin, ssrDirection, hitPosition, dither);
    //if(viewLength * roughness > 20.0) ssR = vec4(1.0, 0.0, 0.0, 1.0);
  }

  //if(!bool(ssR.a) && bool(step(viewLength, sphereMapRenderDistance))) CalculateMappingReflection(reflection, vP.xyz, normal);

  reflection = mix(reflection, ssR.rgb, ssR.a);

  reflection = encodeGamma(reflection);

  vec2 T = RaySphereIntersection(vec3(0.0, rE + max(1.0, eyePosition.y), 0.0), rayDirectionWorld, vec3(0.0), rA);
  if(bool(step(ssR.a, 0.5))) hitPosition = vP + normalizedRayDirection * max(T.x, T.y);

  rayDepth = P2UV(hitPosition).z;

  //if(bool(ssR.a)) rayDepth = P2UV(hitPosition).z;
  //else hitPosition = rayDirection + vP;
/*
  rayPDF = GetRayPDF(normalize(hitPosition), -nvP, normal, roughness);
  reflection *= 1.0 / rayPDF;

  vec3 hitPosition2 = hitPosition + 2.0 * normal * dot(nvP, normal);

  for(int i = 0; i < 8; i++){
    vec2 E = haltonSequence_2n3[i];
         E.y = lerq(E.y, 0.0, BRDF_Bias);

    vec3 N = (smoothNormal, tbn, roughness);
    vec3
  }
*/

  //vec3 hitScreenCoord = nvec3(gbufferProjection * nvec4(hitPosition)) * 0.5 + 0.5;
  hitPosition = normalize((hitPosition) - (vP));

  float d, g;
  vec3 f;

  CalculateBRDF(f, g, d, roughness, metallic, F0, flatNormal, -nvP, hitPosition);

  //vec3 h = normalize(normalizedRayDirection-nvP);
  //f = F(F0, pow5(1.0 - saturate(dot(-nvP, h))));

  //vec3 h = normalize(normalize(hitPosition) - nvP);
  //float vdoth = pow5(1.0 - clamp01(dot(-nvP, normal)));

  //rayPDF = GetRayPDF(normalize(hitPosition), normal, smoothNormal, roughness);
  //rayPDF /= 4.0 * vdoth;
  rayPDF.w = max(1e-5, rayPDF.w);

  rayBRDF = max(d * g, 1e-5);

  float specular = min(1.0, rayBRDF / rayPDF.w);

  //reflection *= specular;

  rayDepth = mix(texture(depthtex0, coord).x, rayDepth, saturate(rayBRDF / rayPDF.w * 0.01));
  //rayDepth = mix(texture(depthtex0, coord).x, rayDepth, pow2(saturate(rayPDF * 0.05)));
  //rayDepth = texture(depthtex0, coord).x;

  //reflection = mat3(gbufferModelViewInverse) * normalDecode(texture2D(composite, coord).xy);

/* DRAWBUFFERS:14 */
  //gl_FragData[0] = vec4(color, 1.0);
  //gl_FragData[0] = vec4(specular, normalize(hitPosition) * 0.5 + 0.5);
  //gl_FragData[0] = vec4(rayPDF, rayBRDF / 40.0, vdoth, 1.0);
  gl_FragData[0] = vec4(hitPosition.xyz * 0.5 + 0.5, rayPDF.w / 128.0);
  gl_FragData[1] = vec4(reflection, rayDepth);
}

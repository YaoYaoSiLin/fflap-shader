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

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform int moonPhase;
uniform int isEyeInWater;

in vec2 texcoord;
in vec3 normalSample;
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

float hitPDF(in vec3 n, in vec3 h, in float roughness){
  float CosTheta = clamp01(dot(n, h));
  return DistributionTerm(roughness, CosTheta) * CosTheta;
}

float hitPDF(in float CosTheta, in float roughness){
  return DistributionTerm(roughness, CosTheta) * CosTheta;
}

vec4 ImportanceSampleGGX(in vec2 E, in float roughness){
  roughness *= roughness;
  //roughness = clamp(roughness, 0.01, 0.99);

  float Phi = E.x * 2.0 * Pi;
  float CosTheta = sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  float D = DistributionTerm(roughness, CosTheta) * CosTheta;
  if(CosTheta < 0.0) return vec4(0.0, 0.0, 1.0, D);

  vec3 H = vec3(cos(Phi) * SinTheta, sin(Phi) * SinTheta, CosTheta);
       //H.xy *= 0.1;

  return vec4(H, D);
}

float GetRayPDF(in vec3 o, in vec3 h, in vec3 n, in float roughness){
  roughness *= roughness;

  //vec3 h = normalize(L + o);

  float ndoth = clamp01(dot(n, h));
  float vdoth = clamp01(dot(o, h));

  float pdf = DistributionTerm(roughness, ndoth) * ndoth;

  return max(pdf, 1e-5);
}

vec3 TBN(in vec3 vec, in vec3 n, in vec3 upVector){
  vec3 t = normalize(cross(upVector, n));
  vec3 b = cross(n, t);
  mat3 tbn = mat3(t, b, n);

  return normalize(tbn * vec);
}

vec3 CalculateNormal(in vec3 n, in mat3 tbn, in float roughness, out float PDF){
  float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale - jittering * 1.0);
  float blueNoise = GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jittering * 1.0);
  float blueNoise2 = GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y * 0.5, jittering * 1.0);

  float steps = 8.0;
  float invsteps = 1.0 / (steps);
  float index = mod(float(frameCounter), steps);

  vec2 E = vec2(blueNoise2, blueNoise);
       E.y = ApplyBRDFBias(E.y);
  vec4 H = ImportanceSampleGGX(E, roughness);

  //H.xy = rotate * H.xy;

  PDF = H.w;

  return normalize(tbn * H.xyz);
}

void main() {
  vec2 j = texcoord - pixel * jittering * SSR_Rendering_Scale;
  vec2 coord = texcoord;

  coord -= jittering * pixel;

	vec4 albedo = texture2D(gcolor, coord);

  float skyLightMap = texture(gdepth, coord).y;
  float torchLightMap = texture(gdepth, coord).x;

  int mask = int(round(texture(gdepth, coord).z * 255.0));
  bool isPlants = mask == 18 || mask == 31 || mask == 83;
  bool isParticles = bool(step(249.5, float(mask)) * step(float(mask), 252.5));

	vec3 normalSurface = normalDecode(texture2D(composite, coord).xy);
  vec3 normalVisible = normalDecode(texture2D(gnormal, coord).xy);

	float smoothness = texture(gnormal, coord).x;
	float metallic   = texture(gnormal, coord).y;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, albedo.rgb, step(0.5, metallic));

	vec3 color = texture2D(gaux2, coord).rgb;

	float depth = texture2D(depthtex0, coord).x;

	vec3 vP = vec3(coord, depth) * 2.0 - 1.0;
			 vP = nvec3(gbufferProjectionInverse * nvec4(vP));
	vec3 nvP = normalize(vP);
  float viewLength = length(vP);

  vec3 facetoPlayerNormal = normalize(-nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.7) * 2.0 - 1.0)));

  if(isParticles) {
    normalVisible = facetoPlayerNormal;
    normalSurface = facetoPlayerNormal;
  }

  float ndotv = dot(nvP, normalSurface);
  if(-0.15 > ndotv) normalVisible = normalSurface;

  vec3 normal = normalDecode(texture2D(gnormal, coord).zw);
  if(isParticles) normal = facetoPlayerNormal;
  vec3 smoothNormal = normal;

  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

  vec3 upVector = abs(worldNormal.z) < 0.4999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  upVector = mat3(gbufferModelView) * upVector;

	vec3 t = normalize(cross(upVector, normal));
	vec3 b = cross(normal, t);
	mat3 tbn = mat3(t, b, normal);

	vec2 motionDither = jittering;
	//float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale);
  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jittering * 0.5);
  float bayer32 = bayer_32x32(texcoord, resolution * SSR_Rendering_Scale);

  float rayPDF;
  float rayBRDF;

  vec3 H = CalculateNormal(normal, tbn, roughness, rayPDF);
  normal = H.xyz;

	vec3 rayDirection = reflect(nvP, normal);
	vec3 normalizedRayDirection = normalize(rayDirection);

	vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

	vec3 reflection = vec3(0.0);

  vec3 eyePosition = vec3(0.0, cameraPosition.y - 63.0, 0.0);
  vec3 rayDirectionWorld = mat3(gbufferModelViewInverse) * (normalizedRayDirection);

	vec3 skySpecularReflection = CalculateInScattering(eyePosition, rayDirectionWorld, sP, 0.76, ivec2(16, 2), vec3(1.0, 1.0, 0.0));
       skySpecularReflection = ApplyEarthSurface(skySpecularReflection, eyePosition, rayDirectionWorld, sP);
       skySpecularReflection = G2Linear(skySpecularReflection);

  if(isEyeInWater == 1) {
    skySpecularReflection = eyesWaterColor.rgb * L2rgb(skyLightingColorRaw);
  }

  torchLightMap = max(0.0, torchLightMap - 0.0667) * 1.071;
  torchLightMap = pow2(torchLightMap * torchLightMap);

  vec3 torchLightingColor = vec3(1.049, 0.5821, 0.0955);

  vec3 fakeLightingReflection = albedo.rgb;
       fakeLightingReflection *= 1.0 - exp(-(torchLightingColor) * 0.125 * torchLightMap);

  skySpecularReflection = mix(color, skySpecularReflection, smoothstep(0.466, 0.866, skyLightMap)) + fakeLightingReflection;

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

  float rayDepth = 0.999;

	vec4 ssR = vec4(0.0);
	vec3 hitPosition = vec3(0.0);

  if(smoothness > 0.1){
	   ssR = raytrace(vP + normal * 0.1, rayDirection, hitPosition, dither);
     ssR.rgb *= overRange;
  }

  reflection = mix(skySpecularReflection, ssR.rgb, ssR.a);

  if(bool(ssR.a)) rayDepth = P2UV(hitPosition).z;
  else hitPosition = rayDirection + vP;
/*
  rayPDF = GetRayPDF(normalize(hitPosition), -nvP, normal, roughness);
  reflection *= 1.0 / rayPDF;

  vec3 hitPosition2 = hitPosition + 2.0 * normal * dot(nvP, normal);

  for(int i = 0; i < 8; i++){
    vec2 E = haltonSequence_2n3[i];
         E.y = lerq(E.y, 0.0, BRDF_Bias);

    vec3 N = CalculateNormal(smoothNormal, tbn, roughness);
    vec3
  }
*/

  hitPosition = normalize(hitPosition - vP);

  vec3 f;
  float d, g;
  roughness = ApplyBRDFBias(roughness);
  CalculateBRDF(f, g, d, roughness, metallic, F0, smoothNormal, -nvP, hitPosition);

  //vec3 h = normalize(normalize(hitPosition) - nvP);
  //float vdoth = pow5(1.0 - clamp01(dot(-nvP, normal)));

  //rayPDF = GetRayPDF(normalize(hitPosition), normal, smoothNormal, roughness);
  //rayPDF /= 4.0 * vdoth;
  rayPDF = max(1e-5, rayPDF);

  rayBRDF = max(d * g, 1e-5);

  float specular = min(1.0, rayBRDF * rayPDF);

  reflection *= specular;

  rayDepth = mix(texture(depthtex0, coord).x, rayDepth, pow2(specular));
  //rayDepth = texture(depthtex0, coord).x;

/* DRAWBUFFERS:4 */
  //gl_FragData[0] = vec4(color, 1.0);
  //gl_FragData[0] = vec4(specular, normalize(hitPosition) * 0.5 + 0.5);
  //gl_FragData[0] = vec4(rayPDF, rayBRDF / 40.0, vdoth, 1.0);
  gl_FragData[0] = vec4(reflection, rayDepth);
}

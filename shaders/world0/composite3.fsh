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

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;

uniform vec3 upPosition;
uniform vec3 shadowLightPosition;
uniform vec3 sunPosition;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform int moonPhase;
uniform int isEyeInWater;

in vec2 texcoord;
in vec3 normalSample;
in vec3 skyLightingColorRaw;
in vec4 eyesWaterColor;

const bool gaux2MipmapEnabled = true;
//const bool gaux1Clear = true;

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

vec2 GetMotionVector(in vec3 coord){
  vec4 view = gbufferProjectionInverse * nvec4(coord * 2.0 - 1.0);
       view /= view.w;
       view = gbufferModelViewInverse * view;
       view.xyz += cameraPosition - previousCameraPosition;
       view = gbufferPreviousModelView * view;
       view = gbufferPreviousProjection * view;
       view /= view.w;
       view.xy = view.xy * 0.5 + 0.5;

  vec2 velocity = coord.xy - view.xy;

  if(coord.z < 0.7) velocity *= 0.0001;

  return velocity;
}

vec2 normalMuti[8] = vec2[8](vec2(-0.7, -0.7),
                             vec2(-1.0, 0.0),
                             vec2(-0.7, 0.7),
														 vec2(0.0, 1.0),
														 vec2(0.7, 0.7),
														 vec2(0.0, 1.0),
														 vec2(0.7, -0.7),
														 vec2(0.0, 1.0)
														 );
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

vec2 RotateDirection(vec2 V, vec2 RotationCosSin) {
    return vec2(V.x*RotationCosSin.x - V.y*RotationCosSin.y,
                V.x*RotationCosSin.y + V.y*RotationCosSin.x);
}

float hitPDF(in vec3 n, in vec3 h, in float roughness){
  float CosTheta = clamp01(dot(n, h));
  return DistributionTerm(roughness, CosTheta) * CosTheta;
}

float hitPDF(in float CosTheta, in float roughness){
  return DistributionTerm(roughness, CosTheta) * CosTheta;
}

#define BRDF_Bias 0.7

vec4 ImportanceSampleGGX(in vec2 E, in float roughness){
  roughness *= roughness;
  //roughness = clamp(roughness, 0.01, 0.99);

  float Phi = E.x * 2.0 * Pi;
  float CosTheta = clamp(sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y)), 0.998, 1.0);
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  float D = DistributionTerm(roughness, CosTheta) * CosTheta;
  if(CosTheta < 0.0) return vec4(0.0, 0.0, 1.0, 1.0);

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

void CalculateRayPDF(inout vec3 lcolor, in vec3 L, in vec3 o, in vec3 n, in mat3 tbn, in float roughness){
  float rayPDF = GetRayPDF(L, o, n, roughness);

  float specular = 1.0;

  lcolor *= specular / rayPDF;

  float totalWeight = 0.0;

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  for(int i = 0; i < steps; i++){
    vec2 E = haltonSequence_2n3[i];
         E.y = lerq(E.y, 0.0, BRDF_Bias);
    vec4 H = ImportanceSampleGGX(E, roughness);

    vec3 N = normalize(tbn * H.xyz);
         N = mix(N, n, 0.7);

    float stepPDF = GetRayPDF(L, o, N, roughness);
    float stepSpecular = 1.0;

    totalWeight += stepSpecular / stepPDF;
  }

  lcolor /= totalWeight;
  lcolor *= float(steps);
}

vec3 CalculateNormal(in vec3 n, in mat3 tbn, in float roughness, out float PDF){
  float r = R2sq(texcoord * resolution * SSR_Rendering_Scale - jittering * 1.0) * 2.0 * Pi;
  mat2 rotate = mat2(cos(r), -sin(r), sin(r), cos(r));

  float steps = 8.0;
  float invsteps = 1.0 / (steps);
  float index = 0.0;//mod(float(frameCounter), steps);

  vec2 E = haltonSequence_2n3[int(index)];
       E.y = mix(E.y, 0.0, BRDF_Bias);
  vec4 H = ImportanceSampleGGX(E, roughness);

  H.xy = rotate * H.xy;
  PDF = H.w;

  return normalize(tbn * H.xyz);
}

void main() {
  vec2 j = texcoord - pixel * jittering * SSR_Rendering_Scale;
  vec2 coord = texcoord;

	vec4 albedo = texture2D(gcolor, coord);

  float skyLightMap = texture(gdepth, coord).y;

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

	vec3 color = texture2DLod(gaux2, coord, 0).rgb;

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
	float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale);
  float bayer32 = bayer_32x32(texcoord, resolution * SSR_Rendering_Scale);

  float rayPDF;
  float rayBRDF;

  vec3 H = CalculateNormal(normal, tbn, roughness, rayPDF);
  normal = H.xyz;

	vec3 rayDirection = reflect(nvP, normal);
	vec3 normalizedRayDirection = normalize(rayDirection);

	vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

	vec3 reflection = vec3(0.0);

	vec3 skySpecularReflection = L2rgb(CalculateSky(normalizedRayDirection, sP, 0.0, 1.0));
  if(isEyeInWater == 1) {
    skySpecularReflection = eyesWaterColor.rgb * L2rgb(skyLightingColorRaw);
  }
  skySpecularReflection = mix(color, skySpecularReflection, smoothstep(0.466, 0.866, skyLightMap));

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

  if(!isPlants && !isParticles && smoothness > 0.1){
	   ssR = raytrace(vP + normal * 0.1, rayDirection, hitPosition, 0.0, 0.0);
     ssR.rgb *= overRange;
  }

  reflection = mix(skySpecularReflection, ssR.rgb, ssR.a);

  if(bool(ssR.a)) rayDepth = P2UV(hitPosition).z;
  else hitPosition = rayDirection;
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

  vec3 f;
  float d, g;
  CalculateBRDF(f, g, d, roughness, metallic, F0, normal, -nvP, normalize(hitPosition));

  vec3 h = normalize(normalize(hitPosition) - nvP);
  float vdoth = pow5(1.0 - clamp01(dot(-nvP, normal)));

  //rayPDF = GetRayPDF(normalize(hitPosition), normal, smoothNormal, roughness);
  //rayPDF /= 4.0 * vdoth;
  rayPDF = max(1e-5, rayPDF);

  rayBRDF = max(d * g, 1e-5);

  float specular = min(1.0, rayBRDF * rayPDF);

  reflection *= min(1.0, rayBRDF * rayPDF);

  //color.rgb = reflection / overRange;

  //if(texcoord.x > 0.5)
  //CalculateRayPDF(reflection, normalize(hitPosition), -nvP, normal, tbn, roughness);

  //vec2 E = haltonSequence_2n3[int(mod(frameCounter, 8))];
  //     E.y = lerq(E.y, 0.0, 0.7);
  //reflection *= 1.0 / ImportanceSampleGGX(E, roughness).w;

  /*
  h = normalize(nreflectVector -nvP);
  ndoth = clamp01(dot(h, normal));
  vdoth = clamp01(dot(-nvP, h));
  float rayPDF = hitPDF(ndoth, roughness) / (4.0 * vdoth);

  reflection *= rayPDF;
  */

  /*
  H.w = max(H.w, 1e-5);
  reflection /= min(20.0, H.w);

  float weightSum = 0.0;

  for(int i = 0; i < int(steps); i++){
    vec2 E = vec2(float(i + 1) * invsteps * 2.0 * Pi, haltonSequence_2n3[i].y);
    E.y = lerq(E.y, 0.0, 0.95);
    float D = ImportanceSampleGGX(E, roughness).w;

    weightSum += 1.0 / max(1e-5, D);
  }

  reflection /= max(weightSum, 0.05);
  */

  //reflection = min(reflection, reflection / H.w / weightSum * steps);

  //reflection *= 0.00001;
  //reflection *= steps;

  //reflection = KarisToneMapping(reflection);


  //vec2 reflectionCoord = (hitPosition).xy;

  //rayDepth = texture(depthtex0, texcoord).x;
  //rayDepth = 1.0 - rayDepth;

	//color = reflection / overRange;
/*
  vec2 velocity = GetMotionVector(vec3(texcoord, depth));
  vec2 previous = texcoord - velocity;

  float mixRate = 1.0 / 8.0;

  float error = length(previous * resolution - texcoord * resolution);

  if(error > resolution.y * 0.014) mixRate = 1.0 - mixRate;
  if(floor(previous) != vec2(0.0)) mixRate = 1.0;
  mixRate *= step(0.0, texture2D(gaux1, previous * SSR_Rendering_Scale).a);

  vec3 mixed = texture2D(gaux1, previous * SSR_Rendering_Scale).rgb;
  mixed = mix(mixed, reflection, mixRate);
*/
  //if(texcoord.x > 0.5)

  /*
  float frameHitPDF = hitPDF(normal, normalize(nreflectVector - nvP), roughness);

  float weightSum = 0.0;

  for(int i = 0; i < int(steps); i++){
    float r = (float(i) + dither * 8.0) * 2.0 * Pi * invsteps;
    vec2 offset = vec2(cos(r), sin(r));
         offset *= 0.05 * radius;

    vec3 n = normalize(tbn * vec3(offset, ndoth));
    vec3 dir = normalize(reflect(nvP, n));

    float pdf = hitPDF(n, normalize(dir - nvP), roughness);
    weightSum += 1.0 / pdf;
    //break;
  }

  if(texcoord.x > 0.5) reflection = reflection / frameHitPDF / weightSum * steps;
  */
  //reflection = reflection / max(1e-5, pdfa) / pdfaSum * steps;

  //vec4 color = texture2D(gaux2, texcoord);


/* DRAWBUFFERS:4 */
  //gl_FragData[0] = vec4(color, 1.0);
  //gl_FragData[0] = vec4(specular, normalize(hitPosition) * 0.5 + 0.5);
  //gl_FragData[0] = vec4(rayPDF, rayBRDF / 40.0, vdoth, 1.0);
  gl_FragData[0] = vec4(reflection, rayDepth);
}

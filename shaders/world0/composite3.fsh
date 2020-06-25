#version 130

#define SSR_Rendering_Scale 0.5

//remove setting: High
#define Surface_Quality Medium        //[Low Medium]

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
in vec3 skyLightingColorRaw;
in vec4 eyesWaterColor;

const bool gaux2MipmapEnabled = true;
const bool gaux1Clear = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);;

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

void CalculateBRDF(out vec3 f, out float g, out float d, in float roughness, in float metallic, in vec3 F0, in vec3 normal, in vec3 view, in vec3 L){
	vec3 h = normalize(L + view);

	float ndoth = clamp01(dot(normal, h));
	float vdoth = pow5(1.0 - clamp01(dot(view, h)));
	float ndotl = clamp01(dot(L, normal));
	float ndotv = 1.0 - clamp01(dot(view, normal));

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

void main() {
  vec2 j = texcoord - pixel * jittering * SSR_Rendering_Scale;

	vec4 albedo = texture2D(gcolor, texcoord);

  float skyLightMap = texture(gdepth, texcoord).y;

  int mask = int(round(texture(gdepth, texcoord).z * 255.0));
  bool isPlants = mask == 18 || mask == 31 || mask == 83;

	vec3 normalSurface = normalDecode(texture2D(composite, texcoord).xy);
  vec3 normalVisible = normalDecode(texture2D(gnormal, texcoord).xy);

	float smoothness = texture2D(gnormal, texcoord).z;//smoothness = floor((0.5 - abs(texcoord.x - 0.5)) * 64.0 * 2.0) * (1.0 / 64.0);
	float metallic   = texture2D(composite, texcoord).z;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, albedo.rgb, step(0.5, metallic));

	vec3 color = texture2DLod(gaux2, texcoord, 0).rgb;

	float depth = texture2D(depthtex0, texcoord).x;

	vec3 vP = vec3(texcoord, depth) * 2.0 - 1.0;
			 vP = nvec3(gbufferProjectionInverse * nvec4(vP));
	vec3 nvP = normalize(vP);
  float viewLength = length(vP);

  if(mask == 250) {
    vec3 facetoPlayerNormal = normalize(-nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.7) * 2.0 - 1.0)));

    normalVisible = facetoPlayerNormal;
    normalSurface = facetoPlayerNormal;
  }

  float ndotv = dot(nvP, normalSurface);
  if(-0.15 > ndotv) normalVisible = normalSurface;
  vec3 normal = normalVisible;
  vec3 smoothNormal = normal;

	vec3 t = normalize(cross(nvP, normal));
	vec3 b = cross(normal, t);
	mat3 tbn = mat3(t, b, normal);

	vec2 motionDither = jittering;
	float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale);
  float bayer32 = bayer_32x32(texcoord, resolution * SSR_Rendering_Scale);

  float NdotV = clamp01(dot(-nvP, normal));
  float coneTangent = clamp(NdotV * (1.0 - smoothness), 0.0, roughness);

  vec3 refVector = vP + normalize(reflect(nvP, normal));
       refVector = P2UV(refVector);
  float ruvLength = length(refVector.xy - texcoord);

  float radius = clamp(log2(coneTangent * resolution.x * ruvLength), 0.0, 6.2831);

  float steps = 8.0 / SSR_Rendering_Scale;
  float invsteps = 1.0 / steps;
  float frameIndex = mod(float(frameCounter), steps);

  vec2 offset = vec2(0.0);

  #if Surface_Quality == High
    dither *= 2.0 * Pi;
  	mat2 rotate = mat2(cos(dither), -sin(dither), sin(dither), cos(dither));

    float r = (1.0 + frameIndex) * invsteps * Pi * 2.0;

    offset = vec2(cos(r), sin(r)) * 0.01;
    offset = offset * rotate;
  #elif Surface_Quality == Medium
    float r = (frameIndex * invsteps + dither) * Pi * 2.0;

    offset = vec2(cos(r), sin(r)) * 0.01;
  #endif

  offset = offset * radius;

	vec3 normalMap = vec3(offset, 1.0);
       //normalMap.xy *= 1.0 / clamp(d, 1.0, 40.0);
			 normalMap = normalize(tbn * normalMap);
       #if Surface_Quality > Low
			 normal = normalMap;
       #endif

	vec3 reflectVector = reflect(nvP, normal);
	vec3 nreflectVector = normalize(reflectVector);

	vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

	vec3 reflection = vec3(0.0);

	vec3 skySpecularReflection = L2rgb(CalculateSky(normalize(reflect(nvP, smoothNormal)), sP, 0.0, 1.0));
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

  float rayDepth = P2UV(vP + nreflectVector).z;

  NdotV = clamp01(dot(-nvP, normal));
  float contTangent = clamp(NdotV * (1.0 - smoothness), 0.0, roughness*1.0);

	vec4 ssR = vec4(0.0);
	vec3 hitPosition = vec3(0.0);

  if(!isPlants && smoothness > 0.1){
	   ssR = raytrace(vP + normal * 0.1, reflectVector, hitPosition, 0.0, contTangent);
     ssR.rgb *= overRange;
  }

	reflection = mix(skySpecularReflection, ssR.rgb, ssR.a);

  if(hitPosition.z > 0.0) rayDepth = hitPosition.z;
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
/* DRAWBUFFERS:4 */
  gl_FragData[0] = vec4(reflection, rayDepth);
}

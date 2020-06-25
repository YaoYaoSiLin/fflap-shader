#version 130

uniform sampler2D depthtex0;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor1;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

in float night;
in float remap;

in vec2 texcoord;
in vec2 jitter;

in vec3 lightVector;
in vec3 worldLightVector;
in vec3 nworldLightVector;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);;

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/atmospheric.glsl"

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;
const bool  shadowHardwareFiltering = false;
float shadowPixel = 1.0 / float(shadowMapResolution);

vec4 GetViewPosition(in vec2 coord, in sampler2D depth){
  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, texture(depth, coord).x) * 2.0 - 1.0);
       vP /= vP.w;

  return vP;
}

vec3 wP2sP(in vec4 wP, out float bias){
	vec4 sP = shadowModelView * wP;
       sP = shadowProjection * sP;
       sP /= sP.w;

  bias = 1.0 / (mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95);

  sP.xy *= bias;
  sP.z /= max(far / shadowDistance, 1.0);
  sP = sP * 0.5 + 0.5;

	return sP.xyz;
}

vec4 CalculateRays(in vec4 wP, in bool isSky){
  vec4 raysColor = vec4(0.0);

  if(isSky) return vec4(1.0);

  int steps = 12;
	float invsteps = 1.0 / float(steps);

  float dither = R2sq((texcoord * resolution + jitter) * 0.5);

	vec3 rayDirection = normalize(wP.xyz) * length(wP.xyz) * invsteps;
  vec4 rayStart = wP + vec4(rayDirection, 0.0) * dither;

  float bias = 0.0;
  float diffthresh = shadowPixel * 1.0;

  for(int i = 0; i < steps; i++){
    rayStart.xyz -= rayDirection.xyz;

    vec3 shadowMap = wP2sP(rayStart, bias);

		float d = texture2D(shadowtex1, shadowMap.xy).x + diffthresh;
    float d2 = texture2D(shadowtex0, shadowMap.xy).x + diffthresh;

    float sampleShading = step(shadowMap.z, d);
    float sampleShading2 = step(shadowMap.z, d2);

		raysColor.a += sampleShading * (1.0 - exp(-length(rayStart.xyz) * 0.1));

    vec4 sampleTexture = texture2D(shadowcolor1, shadowMap.xy);
    sampleTexture.rgb *= exp(-sampleTexture.a * Pi * (1.0 - sampleTexture.rgb));
    raysColor.rgb += mix(sampleTexture.rgb, vec3(1.0), step(sampleShading, sampleShading2));
  }

  raysColor *= invsteps;

  return raysColor;
}

void main() {
	vec4 vP = GetViewPosition(texcoord, depthtex0);
	vec4 wP = gbufferModelViewInverse * vP;

  vec4 atmospheric = vec4(0.0);

	float dither = R2sq(texcoord * resolution) * 1.0;

	vec3 samplePosition = vec3(0.0);
	float fog = 0.0;

	vec3 skyP = mat3(gbufferModelViewInverse) * nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, 1.0) * 2.0 - 1.0));

  bool hited = false;

	for(int i = 0; i < 16; i++){
		float depth = 1.0 - (i + dither * 0.99) * 0.0625;
					depth = pow(depth, 0.001);

		vec3 p = mat3(gbufferModelViewInverse) * nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0));

		float hit = step(length(p), length(wP.xyz));

		if(bool(hit)){
			samplePosition = p;
			fog = 1.0 - exp(-length(samplePosition) * remap * 0.00008);
      hited = true;
		}
	}
	vec3 lP = mat3(gbufferModelViewInverse) * lightVector;

  bool isSky = bool(step(0.9999, texture(depthtex0, texcoord).x));
  float isLand = step(texture(depthtex0, texcoord).x, 0.9999);

  float g = -0.56;
  float g2 = g * g;
  float m = -dot(normalize(wP.xyz), nworldLightVector);
  float HG = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * m, 1.5));

  vec4 rays = CalculateRays(wP, isSky);

  m *= rays.a;
  rays.a *= HG * isLand;

  samplePosition = normalize(worldLightVector + samplePosition);

  atmospheric.rgb = G2Linear(AtmosphericScattering(vec3(0.0, 0.0, 0.0), samplePosition, nworldLightVector, m, g, bool(night)) / overRange);
  atmospheric.rgb *= rays.rgb * rays.a;
  atmospheric.a = rays.a;
  //atmospheric.a = rays.a *;
  //atmospheric.rgb = (rays.rgb * rays.a) * atmospheric.rgb + atmospheric.rgb * fog;
  //atmospheric.a = rays.a + fog;

/* DRAWBUFFERS:4 */
  gl_FragData[0] = atmospheric;
}

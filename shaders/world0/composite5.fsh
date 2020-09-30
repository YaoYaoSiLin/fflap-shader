#version 130

#define SSR_Rendering_Scale 0.5

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform int isEyeInWater;

in float fading;

in vec2 texcoord;

in vec3 sunLightingColorRaw;

in vec4 eyesWaterColor;


vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/brdf.glsl"

vec3 KarisToneMapping(in vec3 color){
	float a = 0.00002;
	float b = float(0xfff) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-b*luma)/(2.0*a-b-luma));
	return color;
}

vec3 invKarisToneMapping(in vec3 color){
	float a = 0.002;
	float b = float(0x2fff) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-(2.0*a-b)*luma)/(b-luma));
	return color;
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
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

  if(texture(depthtex0, texcoord).x < 0.7) velocity *= 0.001;

  return velocity;
}

// https://software.intel.com/en-us/node/503873
vec3 RGB_YCoCg(vec3 c)
{
  // Y = R/4 + G/2 + B/4
  // Co = R/2 - B/2
  // Cg = -R/4 + G/2 - B/4
  //return c;
  //if(texcoord.x > 0.5) return c;
  return vec3(
     c.x/4.0 + c.y/2.0 + c.z/4.0,
     c.x/2.0 - c.z/2.0,
    -c.x/4.0 + c.y/2.0 - c.z/4.0
  );
}

vec3 YCoCg_RGB(vec3 c)
{
  // R = Y + Co - Cg
  // G = Y + Cg
  // B = Y - Co - Cg
  //return c;
  //if(texcoord.x > 0.5) return c;
  return saturate(vec3(
    c.x + c.y - c.z,
    c.x + c.z,
    c.x - c.y - c.z
  ));
}

vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    // note: only clips towards aabb center (but fast!)
    vec3 center  = 0.5 * (maximum + minimum);
    vec3 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    vec3 offset = color - center;

    vec3 ts = abs(extents / (offset + 0.0001));
    float t = clamp(minComponent(ts), 0.0, 1.0);
    return center + offset * t;
}

float GetRayDepth(in vec2 coord){
	return texture2D(gaux1, coord * SSR_Rendering_Scale).a;
}

vec3 GetClosestRayDepth(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      //float neighbor = texture(depthtex0, texcoord).x;
      float neighbor = GetRayDepth(coord + neighborhood);

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;

  return closest;
}

vec3 GetClosest(in vec2 coord, in float scale){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      //float neighbor = texture(depthtex0, texcoord).x;
      float neighbor = texture(depthtex0, coord + neighborhood).x;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;
	closest.xy *= scale;

  return closest;
}

vec3 ReprojectSampler(in sampler2D tex, in vec2 coord){
	#if 0
	return texture2D(tex, coord).rgb;
	#else
	vec3 color = vec3(0.0);
	float totalWeight = 0.0;

	//if(texcoord.x > 0.5)
	coord = round(coord * resolution) * pixel;

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			vec2 samplePosition = vec2(i, j) - 1.5;
			float weight = gaussianBlurWeights(samplePosition + 0.001);

			samplePosition = samplePosition * pixel * SSR_Rendering_Scale + coord;
			vec3 sampleColor = texture2D(tex, samplePosition).rgb;

			color += sampleColor * weight;
			totalWeight += weight;
		}
	}

	color /= totalWeight;

	return color;
	#endif
}

void main() {
	vec2 coord = texcoord;
			 //coord = GetClosest(coord, SSR_Rendering_Scale).st;
			 coord *= SSR_Rendering_Scale;
	vec2 unjitterUV = texcoord + jittering * pixel;

	vec3 albedo = texture2D(gcolor, texcoord).rgb;
	vec3 albedoG = L2Gamma(albedo);

  vec3 normalSurface = normalDecode(texture2D(composite, texcoord).xy);
  vec3 normalVisible = normalDecode(texture2D(gnormal, texcoord).xy);

	float smoothness = texture2D(gnormal, texcoord).r;
	float metallic   = texture2D(gnormal, texcoord).g;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, albedo.rgb, step(0.5, metallic));

	bool isSky = bool(step(texture(gdepth, texcoord).z, 0.999));

	float depth = texture2D(depthtex0, texcoord).x;
	vec3 vP = vec3(texcoord, depth) * 2.0 - 1.0;
			 vP = nvec3(gbufferProjectionInverse * nvec4(vP));
	vec3 nvP = normalize(vP);

  float viewLength = length(vP.xyz);

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).zw);

	vec3 rayDirection = normalize(reflect(nvP, normal));

	vec3 color = texture2D(gaux2, texcoord).rgb;

	float g = 0.0;
	float d = 0.0;
	vec3 f = vec3(0.0);
	//roughness = ApplyBRDFBias(roughness);
	FDG(f, g, d, -nvP, rayDirection, normal, F0, roughness);
	float brdf = max(0.0, g * d);
	//f = F(F0, pow5(1.0 - max(0.0, dot(normalize(nreflectVector-nvP), -nvP))));

	vec3 specular = ReprojectSampler(gaux1, coord);
	vec3 gi = ReprojectSampler(composite, coord);

	//		 reflection = CalculateReflection(gaux1, reflection, coord);
	specular *= f;

	gi = L2Gamma(gi);
	gi *= albedoG * sunLightingColorRaw * fading * invPi;
	gi *= (1.0 - max(step(0.5, texture2D(gcolor, texcoord).a), metallic));
	gi = max(vec3(0.0), gi);
	gi = G2Linear(gi);

  vec3 indirect = specular + gi;

/* DRAWBUFFERS:4 */
  gl_FragData[0] = vec4(indirect, texture2D(gaux1, coord).a);
  //gl_FragData[1] = vec4(antialiased, 1.0);
}

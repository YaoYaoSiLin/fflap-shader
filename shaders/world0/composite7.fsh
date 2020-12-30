#version 130

#define SSR_Rendering_Scale 1.0

#define Enabled_Global_Illumination

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform usampler2D gaux3;

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

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;

in vec4 eyesWaterColor;

const bool gaux3Clear = false;

uniform vec2 jitter;
uniform vec2 pixel;
uniform vec2 resolution;

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/brdf.glsl"
#include "../lib/packing.glsl"

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

  if(texture(depthtex0, texcoord).x < 0.7) velocity *= 0.0001;

  return velocity;
}

// https://software.intel.com/en-us/node/503873
vec3 RGB_YCoCg(vec3 c)
{
  // Y = R/4 + G/2 + B/4
  // Co = R/2 - B/2
  // Cg = -R/4 + G/2 - B/4
  return decodeGamma(c);
  //if(texcoord.x > 0.5) return c;

	c = decodeGamma(c);

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
  return c;
  //if(texcoord.x > 0.5) return c;
  return (saturate(vec3(
    c.x + c.y - c.z,
    c.x + c.z,
    c.x - c.y - c.z
  )));
}

vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    // note: only clips towards aabb center (but fast!)
    vec3 center  = 0.5 * (maximum + minimum);
    vec3 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    vec3 offset = color - center;

    vec3 ts = abs(extents / (offset + 0.0001));

    float t = minComponent((ts));
		t = clamp(t, 0.0, 1.0);

    return center + offset * t;
}

vec4 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
  vec4 result = vec4(0.0);
  float weights = 0.0;

	vec2 position = resolution * pixelPos;
	vec2 centerPosition = floor(position - 0.5) + 0.5;

	vec2 f = position - centerPosition;
	vec2 f2 = f * f;
	vec2 f3 = f * f2;

	float sharpen = 1.0;

	float c = sharpen  * 0.01;
	vec2 w0 =         -c  *  f3 + 2.0 * c          *  f2 - c  *  f;
	vec2 w1 =  (2.0 - c)  *  f3 - (3.0 - c)        *  f2            + 1.0;
	vec2 w2 = -(2.0 - c)  *  f3 + (3.0 - 2.0 * c)  *  f2 + c  *  f;
	vec2 w3 =          c  *  f3 - c                *  f2;
	vec2 w12 = w1 + w2;

	vec2 tc12 = pixel * (centerPosition + w2 / w12);
	vec3 centerColor = texture2D(tex, vec2(tc12.x, tc12.y)).rgb;
	vec2 tc0 = pixel * (centerPosition - 1.0);
	vec2 tc3 = pixel * (centerPosition + 2.0);

	result = vec4(texture2D(tex, vec2(tc12.x, tc0.y)).rgb, 1.0) * (w12.x * w0.y) +
								vec4(texture2D(tex, vec2(tc0.x, tc12.y)).rgb, 1.0) * (w0.x * w12.y) +
								vec4(centerColor, 1.0) * (w12.x * w12.y) +
								vec4(texture2D(tex, vec2(tc3.x, tc12.y)).rgb, 1.0) * (w3.x * w12.y) +
								vec4(texture2D(tex, vec2(tc12.x, tc3.y)).rgb, 1.0) * (w12.x * w3.y);

	result /= result.a;
	result.rgb = encodeGamma(saturate(result.rgb));

	/*
  int steps = 8;
  float invsteps = 1.0 / float(steps);

  for(int i = 0; i < steps; i++){
    float r = float(i) * invsteps * 2.0 * Pi;
    vec2 samplePos = vec2(cos(r), sin(r));
    float weight = gaussianBlurWeights(samplePos + 0.0001);
    samplePos = samplePos * pixel + pixelPos;

    vec4 sampler = texture2D(tex, samplePos);
    result += sampler * weight;
    weights += weight;
  }

  result /= weights;
	*/
  //result.rgb = RGB_YCoCg(result.rgb);

  return result;
}

vec3 GetClosestRayDepth(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      //float neighbor = texture2D(gaux1, texcoord).a;
      float neighbor = texture2D(gaux1, min(0.5 - pixel, coord * 0.5 + neighborhood)).a;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;

  return closest;
}

vec3 GetClosest(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      //float neighbor = texture(depthtex0, texcoord).x;
      //float neighbor = texture(depthtex0, coord + neighborhood).x;
      float neighbor = texture2D(gdepth, min(0.5 - pixel, coord * 0.5 + neighborhood)).a;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;

  return closest;
}

void ResolverAABB(in sampler2D colorSampler, in vec2 coord, inout vec3 minColor, inout vec3 maxColor, in float AABBScale){
	vec3 sampleColor = vec3(0.0);
	float totalWeight = 0.0;

	vec3 m1 = vec3(0.0);
	vec3 m2 = vec3(0.0);

	float radius = 1.0;

	for(float i = -radius; i <= radius; i += 1.0){
		for(float j = -radius; j <= radius; j += 1.0){
			if(i == 0.0 && j == 0.0) continue;

			float weight = 1.0;
			totalWeight += weight;

			vec3 sampler = RGB_YCoCg(texture2D(colorSampler, coord + vec2(i, j) * pixel).rgb) * weight;
			float l = dot03(sampler);

			sampleColor += sampler * weight;
			m1 += sampler * weight;
			m2 += (sampler * sampler) * weight;

			//minColor = min(minColor, sampler);
			//maxColor = min(maxColor, sampler);
		}
	}

	sampleColor /= totalWeight;
	m1 /= totalWeight;
	m2 /= totalWeight;

	vec3 mean = m1;
	vec3 stddev = sqrt((m2) - (mean * mean));

	minColor = sampleColor - stddev * AABBScale;
	maxColor = sampleColor + stddev * AABBScale;

	vec3 centerColor = RGB_YCoCg(texture2D(colorSampler, coord).rgb);
	minColor = min(minColor, centerColor);
	maxColor = max(maxColor, centerColor);
}

vec3 TemporalAntialiasingSpecular(in sampler2D sampler, in vec2 coord){
	vec2 unjitter = min(coord * 0.5 + jitter * pixel, 0.5 - pixel);

	vec3 closest = GetClosestRayDepth(texcoord);

  vec2 velocity = GetMotionVector(closest);
	float velocityLength = length(velocity * resolution);

	vec2 previousCoord = texcoord - velocity;

	vec3 currentColor = RGB_YCoCg(texture2D(sampler, unjitter).rgb);

	vec3 previousColor = RGB_YCoCg(vec3(unpackUnorm2x16(texture2D(gaux3, previousCoord).r).x,
																			unpackUnorm2x16(texture2D(gaux3, previousCoord).g).x,
																			unpackUnorm2x16(texture2D(gaux3, previousCoord).b).x));
//apoidea
	vec3 weightA = vec3(0.95) * float(floor(previousCoord) == vec2(0.0));
  vec3 weightB = vec3(1.0 - weightA);

	vec3 maxColor = vec3(0.0);
	vec3 minColor = vec3(1.0);
	ResolverAABB(sampler, unjitter, minColor, maxColor, 4.0);

	vec3 antialiased = previousColor;
			 if(bool(velocityLength))antialiased = clamp(antialiased, minColor, maxColor);
			 antialiased = weightA * antialiased + currentColor * weightB;

	return YCoCg_RGB(antialiased);
}

#ifdef Enabled_Global_Illumination
vec3 TemporalAntialiasingGI(in sampler2D sampler, in vec2 coord){
	vec2 unjitter = min(coord * 0.5, 0.5 - pixel * 2.0);

	vec3 closest = GetClosest(texcoord);

  	vec2 velocity = GetMotionVector(closest);
	float velocityLength = length(velocity * resolution);

	vec2 previousCoord = texcoord - velocity;

	vec3 currentColor = RGB_YCoCg(texture2D(sampler, unjitter).rgb);
	//vec3 previousColor = RGB_YCoCg(encodeGamma(texture2D(gaux3, previousCoord).rgb));

	vec3 previousColor = RGB_YCoCg(vec3(unpackUnorm2x16(texture2D(gaux3, previousCoord).r).y,
																			unpackUnorm2x16(texture2D(gaux3, previousCoord).g).y,
																			unpackUnorm2x16(texture2D(gaux3, previousCoord).b).y));
//apoidea
	vec3 weightA = vec3(0.95) * float(floor(previousCoord) == vec2(0.0));
  vec3 weightB = vec3(1.0 - weightA);

	vec3 maxColor = vec3(0.0);
	vec3 minColor = vec3(1.0);
	ResolverAABB(sampler, unjitter, minColor, maxColor, 2.0);

	vec3 antialiased = previousColor;
			 //antialiased = mix(antialiased, clamp(antialiased, minColor, maxColor), saturate(velocityLength * 0.125));
		 if(bool(velocityLength)) antialiased = clamp(antialiased, minColor, maxColor);
			 antialiased = weightA * antialiased + currentColor * weightB;

	return YCoCg_RGB(antialiased);
}
#endif

vec3 invKarisToneMapping(in vec3 color){
	float a = 0.0027;
	float b = float(0x9fff) / 65535.0;

	float lum = maxComponent(color);

	if(bool(step(lum, a))) color;

	return color/lum*((a*a-(2.0*a-b)*lum)/(b-lum));
}

void main() {
	vec2 coord = texcoord;
			 //coord = GetClosest(coord, SSR_Rendering_Scale).st;
			 //coord *= SSR_Rendering_Scale;

	vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
	vec3 viewDirection = normalize(viewPosition);

	vec3 flatNormal = normalDecode(texture2D(gnormal, coord).xy);
	vec3 texturedNormal = normalDecode(texture2D(composite, coord).xy);
  vec3 visibleNormal = texturedNormal;
  if(bool(step(texture2D(gcolor, coord).a, 0.9999))) flatNormal = texturedNormal;
  float ndotv = dot(-viewDirection, texturedNormal);
  if(bool(step(ndotv, 0.2))) visibleNormal = flatNormal;

	vec3 rayDirection = normalize(reflect(viewDirection, visibleNormal));
	vec3 halfNormal = normalize(rayDirection - viewDirection);

	vec3 albedo = texture2D(gcolor, texcoord).rgb;
			 albedo = decodeGamma(albedo);

	float materialID = round(texture(gnormal, texcoord).z * 255.0);
	float isOutLine = step(246.5, materialID) * step(materialID, 247.5);
	float isSky = step(249.5, materialID);

	vec2 specularPackge = unpack2x8(texture(composite, texcoord).b);
	float smoothness = specularPackge.x;
	float metallic   = specularPackge.y;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = mix(vec3(max(0.02, metallic)), (albedo.rgb), step(0.5, metallic));

	//if(texcoord.x > 0.5)
	//rayDirection = texture2D(gdepth, texcoord * 0.5).rgb * 2.0 - 1.0;

	//vec3 f = F(F0, pow5(1.0 - saturate(dot(-viewDirection, halfNormal))));
	vec3 f = vec3(0.0);
	float g, d = 0.0;
	float c = 4.0 * abs(dot(visibleNormal, rayDirection)) * abs(dot(visibleNormal, -viewDirection)) + 1e-5;
	FDG(f, g, d, -viewDirection, rayDirection, visibleNormal, F0, roughness);

	float ndotl = dot(rayDirection, visibleNormal);

	vec3 fr = f * saturate(g * d * abs(ndotl));
	//vec3 fr = f * saturate(g * d / c * abs(ndotl));
	//if(maxComponent(fr) > 1.0) fr = vec3(1.0, 0.0, 0.0);

	vec3 color = texture2D(gaux2, texcoord).rgb;
		 color = decodeGamma(color) * decodeHDR;

	vec3 indirect = vec3(0.0);
	if(bool(step(texture2D(gcolor, texcoord).a, 0.001))) indirect = vec3(0.0);

	vec3 specularRaw = (TemporalAntialiasingSpecular(gaux1, coord));
	vec3 diffuseRaw = vec3(0.0);

	vec3 specular = (specularRaw) * (1.0 - isOutLine - isSky);
	color += specular * fr;

	//diffuseRaw = encodeGamma(TemporalAntialiasingGI(gdepth, coord));
	//vec3 diffuse = diffuseRaw * (1.0 - metallic) * (1.0 - f * min(1.0, g * d));

	#ifdef Enabled_Global_Illumination
		diffuseRaw = (TemporalAntialiasingGI(gdepth, coord));

		vec3 diffuse = diffuseRaw * (1.0 - isOutLine - isSky);
				 //diffuse *= pow(texture2D(gnormal, texcoord).a, 0.5);//Ambient Occlusion
				 diffuse *= 1.0 - (metallic);
				 diffuse *= step(0.99, texture2D(gcolor, texcoord).a);
				 //diffuse *= 1.0 - f;

		color += invPi * (diffuse) * albedo.rgb * sunLightingColorRaw * fading * 3.0;

		diffuseRaw = encodeGamma(diffuseRaw);
	#endif

	color = encodeGamma(color * encodeHDR);

	specularRaw = encodeGamma(specularRaw);

	uvec4 uiantialiased = uvec4(packUnorm2x16(vec2(specularRaw.r, diffuseRaw.r)),
															packUnorm2x16(vec2(specularRaw.g, diffuseRaw.g)),
															packUnorm2x16(vec2(specularRaw.b, diffuseRaw.b)),
															0x0);

/* DRAWBUFFERS:56 */
  gl_FragData[0] = vec4(color, 1.0);
  //gl_FragData[1] = vec4(diffuse, 1.0);
  gl_FragData[1] = uiantialiased;
}

#version 130

//#define RawOut
#define Enabled_TAA
	#define TAA_ToneMapping 0x4fff //[OFF 0xfff 0x1fff 0x3fff 0x4fff 0x5fff 0x6fff 0x7fff 0xffff]

uniform sampler2D gaux2;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

in vec2 texcoord;

const bool gaux2MipmapEnabled = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "../libs/common.inc"

#define Stage Bloom
#define bloomSampler gaux2

#include "../libs/dither.glsl"
#include "../libs/PostProcessing.glsl"

vec3 Uncharted2Tonemap(in vec3 color, in float x) {
	float A = 2.51;
	float B = 0.59;
	float C = 0.1;
	float D = 0.21;
	float E = 0.0001;
	float F = 0.09;

	vec3 color2 = ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))/E/F;
			 color2 /= ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))/E/F;

	return (color2);
}

vec3 Uncharted2Tonemap(in vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;

  return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

vec3 ACESToneMapping(in vec3 color, in float adapted_lum) {
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

//https://graphicrants.blogspot.com/2013/12/tone-mapping.html
vec3 KarisToneMapping(in vec3 color){
	float a = 0.01953;
	float b = float(TAA_ToneMapping) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-b*luma)/(2.0*a-b-luma));
	return color;
}

vec4 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
	vec4 result = vec4(0.0);

	float weights = 0.0;

	vec3 centerColor = texture2D(tex, pixelPos).rgb * overRange;

	vec3 minColor = vec3(100.0);
	vec3 maxColor = vec3(0.0);

	for(float i = -1.0; i <= 1.0; i += 1.0){
		for(float j = -1.0; j <= 1.0; j += 1.0){
			vec2 samplePos = vec2(i, j) * 0.5;
			float weight = gaussianBlurWeights(samplePos + 0.0001);

			vec4 sampler = texture2D(tex, pixelPos + samplePos * pixel);

			vec3 sampler2 = texture2D(tex, pixelPos + vec2(i, j) * pixel).rgb * overRange;
			minColor = min(minColor, sampler2);
			maxColor = max(maxColor, sampler2);

			weights += weight;
			result += sampler * weight;
		}
	}

	result /= weights;
	result *= overRange;

	vec3 blend = saturate((maxColor - minColor) / centerColor);
			 blend = mix(vec3(0.25), vec3(0.75), blend);

	result.rgb = mix(centerColor, result.rgb, blend);


	return result;
}

vec4 opElongate( in vec3 p, in vec3 h )
{
    //return vec4( p-clamp(p,-h,h), 0.0 ); // faster, but produces zero in the interior elongated box

    vec3 q = abs(p)-h;
    return vec4(max(q,vec3(0.0)), min(maxComponent(q), 0.0));
}

void main() {
  vec4 bloom = vec4(0.0);

  #ifdef Enabled_Bloom
	bloom = CalculateBloomSampler(texcoord);
  #endif

  vec3 color = texture2D(gaux2, texcoord).rgb;
	color *= overRange;

	//if(floor(coord) != vec2(0.0)) color = vec3(0.0);

	#if defined(Enabled_TAA) && TAA_ToneMapping > OFF
	//if(texcoord.x > 0.5)
	color = ReprojectSampler(gaux2, texcoord).rgb;
	//color = texture2D(gaux2, round(texcoord * resolution) * pixel).rgb * overRange;

	//color = getLum(color);
	//color = color * vec3(0.25, 0.3, 0.45) / 0.5;

	//float luma = getLum(color);

	//color = mix(color * vec3(0.2, 0.99, 0.5) * 2.0, color, saturate((luma * overRange - 0.5)));
	//color = mix(color * vec3(0.2, 0.3, 0.5) * 2.0, color, saturate(vec3(luma) * overRange * 127.0 - 127.0));

	color = KarisToneMapping(color);
	#endif

	#ifndef RawOut

	#endif

/* DRAWBUFFERS:235 */
  gl_FragData[0] = bloom;
  gl_FragData[1] = vec4(texture2D(gaux2, texcoord).rgb, 1.0);
  gl_FragData[2] = vec4(color, 1.0);
}

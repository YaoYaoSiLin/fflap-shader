#version 130

//#define RawOut
#define Enabled_TAA
	#define TAA_ToneMapping 0x7ff 		//[OFF 0xff 0x1ff 0x2ff 0x3ff 0x7ff 0xfff 0x1ffff 0x3fff 0x7fff 0xffff]

#define Enabled_Bloom

uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;

in vec2 texcoord;

const bool gaux2MipmapEnabled = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"
#include "../lib/packing.glsl"

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
	float a = 0.0027;
	float b = float(0x9fff) / 65535.0;

	float lum = maxComponent(color);

	if(bool(step(lum, a))) return color;

	return color/lum*((a*a-b*lum)/(2.0*a-b-lum));

	//return color;

	//if(luma > a) 
	//color = color / luma*((a*a-b*luma)/(2.0*a-b-luma));
	//return color;
}

const float sigma = 0.83;
const float phi = 2.0 * sigma * sigma;

vec3 GetBloomSample(in vec2 coord, in vec2 offset, in float mipmap){
	float radius = 1.0;

	coord -= offset;

	coord = round(coord * resolution + 2.0) * mipmap * pixel;
	if(floor(coord) != vec2(0.0)) return vec3(0.0);

	vec4 blur = vec4(0.0);

	for(float i = -radius; i <= radius; i += 1.0){
		for(float j = -radius; j <= radius; j += 1.0){
			//if(i == 0.0 && j == 0.0) continue;

			vec2 direction = vec2(i, j);
			//direction += saturate(abs(direction) - 1.0) * vec2(sign(direction.x), sign(direction.y));

			vec2 bloomCoord = coord + direction * pixel * mipmap;
			if(floor(bloomCoord) != vec2(0.0)) continue;

			float l2 = (direction.x * direction.x + direction.y * direction.y + 1e-5);
			float weight = exp(-l2 / phi);

			vec3 bloomSample = decodeGamma(texture2D(gaux2, bloomCoord).rgb);
			//vec3 bloomSample = decodeGamma(texture2DLod(gaux2, bloomCoord, (mipmap - 2.0) * 0.25).rgb);
					 bloomSample = sqrt(bloomSample * luminance(bloomSample));
					 bloomSample = saturation(bloomSample, 2.0);

			blur.rgb += bloomSample * weight;
			blur.a += weight;
		}
	}

	return blur.rgb / blur.a;
}

void main() {
  vec3 bloom = vec3(0.0);

  vec3 color = texture2D(gaux2, texcoord).rgb;
	   color = decodeGamma(color) * decodeHDR;

	vec2 bloomOffset = pixel * 5.0;

	//color = vec3(0.0);

	#ifdef Enabled_Bloom

	for(int level = 2; level < 6; level++){
		float mipmap = exp2(float(level));

		// + (pixel * mipmap * 0.0625 * vec2(1.0, aspectRatio))
		bloom += GetBloomSample(texcoord, bloomOffset, mipmap);
		bloomOffset.x += (1.0 / mipmap) + pixel.x * mipmap + pixel.x * 5.0;
	}

	bloom = encodeGamma(bloom);
	#endif

	#if defined(Enabled_TAA) && TAA_ToneMapping > OFF
	color = KarisToneMapping(color * 30.0);
	#endif

	color = encodeGamma(color);

	#ifndef RawOut

	#endif

/* DRAWBUFFERS:235 */
  gl_FragData[0] = vec4(bloom, 1.0);
  gl_FragData[1] = vec4(texture2D(gaux2, texcoord).rgb, 1.0);
  gl_FragData[2] = vec4(color, 1.0);
}

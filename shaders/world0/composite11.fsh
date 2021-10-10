#version 130

//#define RawOut
#define Enabled_TAA
	#define TAA_ToneMapping 0x7ff 		//[OFF 0xff 0x1ff 0x2ff 0x3ff 0x7ff 0xfff 0x1ffff 0x3fff 0x7fff 0xffff]

#define Enabled_Bloom

uniform sampler2D gaux2;

uniform vec2 resolution;
uniform vec2 pixel;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;

in vec2 texcoord;

const bool gaux2MipmapEnabled = true;

#include "/libs/common.inc"
#include "/libs/dither.glsl"
#include "/lib/packing.glsl"

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
	return color = color / (color + 1.0);
/*
	color = pow(color, vec3(0.5));

	float a = 0.1;
	float b = 1.0;

	float lum = maxComponent(color);
	
	if(lum <= a) return color;

	return color/lum*((a*a-b*lum)/(2.0*a-b-lum));
	*/
}

const float sigma = 0.83;
const float phi = 2.0 * sigma * sigma;
const float filter_radius = 2.0;
#define BloomGaussBlurRadius 2.0

vec3 GetBloomSample(in vec2 coord, in vec2 offset, in float mipmap){
	coord -= offset;
	coord += 0.5 * pixel * vec2(1.0, 1.0 / aspectRatio);

	coord = floor(coord * resolution * vec2(mipmap)) * pixel;

	coord += jitter;
	if(floor(coord) != vec2(0.0)) return vec3(0.0);

	vec4 blur = vec4(0.0);

	#if 1
	for(float i = -filter_radius; i <= filter_radius; i += 1.0){
		for(float j = -filter_radius; j <= filter_radius; j += 1.0){
			//if(i == 0.0 && j == 0.0) continue;

			vec2 direction = vec2(i, j);
			//direction += saturate(abs(direction) - 1.0) * vec2(sign(direction.x), sign(direction.y));

			vec2 bloomCoord = coord + direction * pixel * mipmap;
			//if(floor(bloomCoord) != vec2(0.0)) continue;

			float l2 = pow2(length(direction)) + 1e-5;
			float weight = exp(-l2 / phi);

			vec3 bloomSample = decodeGamma(texture2D(gaux2, bloomCoord).rgb);

			blur += vec4(bloomSample, 1.0) * weight;
		}
	}

	#else
		blur = vec4(decodeGamma(texture2D(gaux2, coord).rgb), 1.0);
	#endif

	return blur.rgb / blur.a;
}

	vec3 RGB_YCoCg(vec3 c)
	{
		// Y = R/4 + G/2 + B/4
		// Co = R/2 - B/2
		// Cg = -R/4 + G/2 - B/4
    //return c;

    c = decodeGamma(c);

		return vec3(
			 c.x/4.0 + c.y/2.0 + c.z/4.0,
			 c.x/2.0 - c.z/2.0,
			-c.x/4.0 + c.y/2.0 - c.z/4.0
		);
	}

void main() {
  vec3 bloom = vec3(0.0);

  vec3 color = texture2DLod(gaux2, texcoord, 0).rgb;
	   color = decodeGamma(color);

	   //color = decodeGamma(texture2D(gaux1, texcoord * 0.5).rgb);

	vec2 filter_radius_offset = pixel * (1.0 + filter_radius * 2.0);

	vec2 bloomOffset = filter_radius_offset;

	//color = vec3(0.0);

	#ifdef Enabled_Bloom

	for(int level = 1; level < 6; level++){
		float mipmap = exp2(float(level));

		// + (pixel * mipmap * 0.0625 * vec2(1.0, aspectRatio))
		bloom += GetBloomSample(texcoord, bloomOffset, mipmap);
		bloomOffset.x += (1.0 / mipmap) + filter_radius_offset.x;
	}

	bloom /= 255.0;
	bloom = encodeGamma(bloom);
	#endif

	#if defined(Enabled_TAA) && TAA_ToneMapping > OFF
	color = KarisToneMapping(color);
	color = encodeGamma(color);
	#else
	color = encodeGamma(color);
	#endif

	#ifndef RawOut

	#endif

  	gl_FragData[0] = vec4(encodeGamma(decodeGamma(texture2DLod(gaux2, texcoord, 0).rgb / 31.0)), 1.0);
  	gl_FragData[1] = vec4(bloom, 1.0);
  	gl_FragData[2] = vec4(color, 1.0);
}
/* DRAWBUFFERS:135 */

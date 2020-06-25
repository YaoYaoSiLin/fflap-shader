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
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);;

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

void main() {
  vec4 bloom = vec4(0.0);

  #ifdef Enabled_Bloom
	bloom = CalculateBloomSampler(texcoord);
  #endif

  vec3 color = texture2D(gaux2, texcoord).rgb;
	/*
	vec3 sharpen = vec3(0.0);

	for(float i = -1.0; i <= 1.0; i += 1.0){
		for(float j = -1.0; j <= 1.0; j += 1.0){
			sharpen += texture2D(gaux2, texcoord + vec2(i, j) / MC_RENDER_QUALITY * pixel).rgb;
		}
	}

	sharpen -= color;
	sharpen *= 0.125;
	sharpen -= color;

	color += sharpen * 0.1;
	*/
	//color = clamp01(color);

/*
	vec2 fragCoord = round(texcoord * 0.5 * resolution);
	if(mod(fragCoord.x + fragCoord.y, 2) > 0.5 && texcoord.x < 0.5) {
		//color = vec3(0.0);
		vec3 s1 = texture2D(gaux2, texcoord + vec2(pixel.x, 0.0)).rgb;
		vec3 s2 = texture2D(gaux2, texcoord - vec2(pixel.x, 0.0)).rgb;
		vec3 s3 = texture2D(gaux2, texcoord + vec2(0.0, pixel.y)).rgb;
		vec3 s4 = texture2D(gaux2, texcoord - vec2(0.0, pixel.y)).rgb;
		vec3 minColor = min(s1, min(s2, min(s3, s4)));
		vec3 maxColor = max(s1, max(s2, max(s3, s4)));
		vec3 weight = clamp(clamp01((maxColor - minColor) / (color)), vec3(0.1), vec3(0.5));
		vec3 weight2 = 1.0 - weight;

		color = (s1 + s2 + s3 + s4) * 0.25 * weight2 + weight * color;
	}
*/
	color *= overRange;

	#if defined(Enabled_TAA) && TAA_ToneMapping > OFF
	//if(texcoord.x > 0.5)
	color = KarisToneMapping(color);
	#endif

	#ifndef RawOut

	#endif

/* DRAWBUFFERS:235 */
  gl_FragData[0] = bloom;
  gl_FragData[1] = vec4(texture2D(gaux2, texcoord).rgb, 1.0);
  gl_FragData[2] = vec4(color, 1.0);
}

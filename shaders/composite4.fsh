#version 120

uniform sampler2D gaux2;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

varying vec2 texcoord;

const bool gaux2MipmapEnabled = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);;

#include "libs/common.inc"

#define Stage Bloom
#define bloomSampler gaux2

#include "libs/dither.glsl"
#include "libs/PostProcessing.glsl"

vec3 Uncharted2Tonemap(in vec3 color, in float x) {
	float A = 2.51;
	float B = 0.59;
	float C = 0.1;
	float D = 0.31;
	float E = 0.0001;
	float F = 0.09;

	//color = L2rgb(color);

	vec3 color2 = ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))/E/F;
			 color2 /= ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))/E/F;
			 //color2 = color2 * 0.249;

	//float l = getLum(color2) * 0.999 + 0.001;

	//color = color / (color * 0.999 + 0.001) * color2;
	//color = color2 / ;

	//l = color2 / (1.0 + l) * 0.5;

	//vec3 l = color2 / (color * 0.996 + 0.004) * color2;
	//color = l / (color2 * 0.996 + 0.004) * color;

	return (color2);
}

void main() {
  vec4 bloom = vec4(0.0);

  #ifdef Enabled_Bloom
	bloom = CalculateBloomSampler(texcoord);
  #endif

  vec3 color = texture2D(gaux2, texcoord).rgb;

  //color *= overRange;

  color = rgb2L(color);

	color *= overRange;
  //color = color / (color + 0.08);
	color = (color / (color + 0.0032)) / (1.0 / (1.0 + 0.0032));
	//color = Uncharted2Tonemap(color, 1.0);

  color = L2rgb(color);

/* DRAWBUFFERS:235 */
  gl_FragData[0] = bloom;
  gl_FragData[1] = vec4(texture2D(gaux2, texcoord).rgb, 1.0);
  gl_FragData[2] = vec4(color, 1.0);
}

#version 130

#define Brightness 0.3

#define Enabled_TAA
	#define TAA_Post_Sharpen 50 			//[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
	#define TAA_ToneMapping 0x4fff 		//[OFF 0xfff 0x1fff 0x3fff 0x4fff 0x5fff 0x6fff 0x7fff 0xffff]

#define Enabled_Bloom								//Bloom, used on screen Effect and hurt Effects
	#define Bloom_Strength 1.0				//[0.08 0.12 0.16 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.3 2.6]
	#define Bloom_Sample_Lod 0.7			//[0.7 0.75 0.8 0.85 0.9 0.95 1.0]
	#define Only_Bloom		 0					//[0 1 2]

//#define RawOut

#define Screen_Bias 0.03

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

#define colortex gaux2

uniform float frameTimeCounter;
uniform float viewWidth;
uniform float viewHeight;
uniform float rainStrength;
uniform float aspectRatio;

uniform int isEyeInWater;

uniform ivec2 eyeBrightnessSmooth;

#define Setting 1
#define Advanced Setting

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "../libs/common.inc"

#define Stage Final
#define bloomSampler gnormal

#include "../libs/PostProcessing.glsl"

vec2 uv = gl_TexCoord[0].st;
/*
float w0(float a)
{
    return (1.0/6.0)*(a*(a*(-a + 3.0) - 3.0) + 1.0);
}

float w1(float a)
{
    return (1.0/6.0)*(a*a*(3.0*a - 6.0) + 4.0);
}

float w2(float a)
{
    return (1.0/6.0)*(a*(a*(-3.0*a + 3.0) + 3.0) + 1.0);
}

float w3(float a)
{
    return (1.0/6.0)*(a*a*a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a)
{
    return 1.0 + w3(a) / (w2(a) + w3(a));
}

vec4 texture2D_bicubic(sampler2D tex, vec2 uv)
{
	vec2 resolution = vec2(viewWidth, viewHeight);

  uv = uv * 2.0 - 1.0;
	uv = uv * resolution;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

    float g0x = g0(fuv.x);
    float g1x = g1(fuv.x);
    float h0x = h0(fuv.x);
    float h1x = h1(fuv.x);
    float h0y = h0(fuv.y);
    float h1y = h1(fuv.y);

	vec2 p0 = (iuv + vec2(h0x, h0y) * 0.33) / resolution * 0.5 + 0.5;
	vec2 p1 = (iuv + vec2(h1x, h0y) * 0.33) / resolution * 0.5 + 0.5;
	vec2 p2 = (iuv + vec2(h0x, h1y) * 0.33) / resolution * 0.5 + 0.5;
	vec2 p3 = (iuv + vec2(h1x, h1y) * 0.33) / resolution * 0.5 + 0.5;

  //return texture2D(tex, OffsetCoord(uv, vec2(h0x, h0y) - 0.5));


  vec4 r = g0(fuv.y) * (g0x * texture2D(tex, p0)  +
                        g1x * texture2D(tex, p1)) +
           g1(fuv.y) * (g0x * texture2D(tex, p2)  +
                        g1x * texture2D(tex, p3));

  //if(r.a < 0.999) return vec4(r, )
  //if(r.a > 0.0) r.a = 1.0;
  return r;
}
*/

vec3 Uncharted2Tonemap(vec3 x) {
	const float A = 0.22f;
	const float B = 0.30f;
	const float C = 0.10f;
	const float D = 0.20f;
	const float E = 0.01f;
	const float F = 0.30f;

	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 ACESToneMapping(in vec3 color) {
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

vec3 ToneMapping(in vec3 color){
	float a = 0.03125;
	//color = color / (0.24 + color*color);
	//color = L2rgb(color);

	//color = pow(color, vec3(1.2));
	//color *= 2.3;
	//color = (color / (color + a)) / (1.0 / (1.0 + a));

	//color = pow(color, vec3(0.8333));

	//const float white = 0.99;

	//color = color / (a + color);
	//float curr = white / (a + white);
	//color *= curr;

	const float white = 26.8;

	//color *= overRange;
	//if(maxComponent(color) > white) color = vec3(1.0, 0.0, 0.0);

	color = Uncharted2Tonemap(color*38.0);
	color /= Uncharted2Tonemap(vec3(white));
	color = saturation(color, 1.09);

	//color = 1.0 - exp(-color * (1.0 - dot03(color)) * 38.0 * 38.0);
	//color *= 0.1;

	//color *= 38.0;
	//color = color / ()

	//color = ACESToneMapping(color * overRange);
	/*
	color = saturation(color, 1.1);
	*/

	#if Tone_Mapping == Uncharted2
	#endif

	#if Tone_Mapping == ACES
	#endif

	#if Tone_Mapping == Uncharted2_ACES
	#endif

	//color = 1.0 - exp(-color / dot03(color) * 1.0);
	//color *= 2.0;

	//color = pow(color, vec3(1.1));
	//color = saturation(color, 1.1414);

	/*
	color = color / (a+color);
	float curr = W / (a+W);
	color *= curr;
	color = ACESToneMapping(color);
*/
	/*
	color*=overRange;
	color = color / (1.0+color);
	color *= 1.0/(overRange*0.7+1.0);
	color = ACESToneMapping(color);*/


	//color = pow(color, vec3(0.95));

	//color = max(vec3(0.0), color);

	//color = color / (color+1.0);
	//color*= 2.0;
	//color = ACESToneMapping(color*8.0);


	/*
	const float W = 11.2;

	color *= overRange;

	float ExposureBias = 1.0;
	color = Uncharted2Tonemap(color*ExposureBias);

	vec3 curr = 1.0/Uncharted2Tonemap(vec3(W));
	color *= curr;

	color = pow(color, vec3(1.1));
	color = saturation(color, 1.1414);
	*/

	//color = ACESToneMapping(color);

	//color = pow(color,vec3(1.1));

	//color = saturation(color, 1.13);

	//color *= overRange/16.0;

	//color = Uncharted2Tonemap(color*32.0);
	//color /= Uncharted2Tonemap(vec3(11.2));

	//color *= 3.0;
	//if(color.r > 1.0 || color.g>1.0||color.b>1.0) color = vec3(0.0);
	//color = rgb2L(color);

	//color *= 16.0;
	//color *= 16.0;
	//color = ACESToneMapping(color);

	return color;
}

//https://graphicrants.blogspot.com/2013/12/tone-mapping.html
vec3 invKarisToneMapping(in vec3 color){
	float a = 0.01953;
	float b = float(TAA_ToneMapping) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-(2.0*a-b)*luma)/(b-luma));
	return color;
}

void main(){
	vec2 texcoord = uv;

  vec3 color = texture2D(gaux2, texcoord).rgb;

	//if(maxComponent(color) < 0.01953) color = vec3(1.0, 0.0, 0.0);

	#ifdef Enabled_TAA
		#if TAA_Post_Sharpen > 0
			vec3 colorIndex = vec3(0.0);

			for(float i = -1.0; i <= 1.0; i += 1.0){
				for(float j = -1.0; j <= 1.0; j += 1.0){
					if(i != 0.0 || j != 0.0)
					colorIndex += texture2D(colortex, texcoord + pixel * vec2(i, j)).rgb;
				}
			}

			colorIndex *= 0.125;

			color += (color - colorIndex) * 0.0025 * TAA_Post_Sharpen;
			color = clamp01(color);
		#endif

		#if TAA_ToneMapping > OFF
		color = invKarisToneMapping(color);
		#endif
		color = clamp01(color);
	#endif

	color = rgb2L(color);

	float vignette = distance(texcoord, vec2(0.5));

	vec3 bloom = vec3(0.0);

	#ifdef Enabled_Bloom
	CalculateBloom(bloom, texcoord);
	color += bloom;
	#endif


	float exposure = texture2D(colortex, vec2(0.5)).a;
				exposure = exposure / (0.45 + exposure) * (1.0 / (1.0 + 0.45));
				exposure *= Pi * 2.0;

	//color /= exposure + 0.0001;

	color = ToneMapping(color);
	//color = ACESToneMapping(color);

	color = L2rgb(color);
	//color = texture2D(gaux2, texcoord).rgb * 2.0 - 1.0;
	//if(texcoord.x < 0.1) color = vec3(exposure);

	#ifdef RawOut
	color = texture2D(gaux2, texcoord).rgb;
	#endif

  gl_FragColor = vec4(color, 1.0);
}

#version 130

#define Brightness 0.3

#define Enabled_Exposure

#define Enabled_TAA
	#define TAA_Post_Sharpen 0 				//[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
	#define TAA_ToneMapping 0x7ff 		//[OFF 0xff 0x1ff 0x2ff 0x3ff 0x7ff 0xfff 0x1ffff 0x3fff 0x7fff 0xffff]

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

uniform sampler2D noisetex;

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

//#include "../libs/PostProcessing.glsl"

vec4 opElongate( in vec3 p, in vec3 h )
{
    //return vec4( p-clamp(p,-h,h), 0.0 ); // faster, but produces zero in the interior elongated box

    vec3 q = abs(p)-h;
    return vec4( max(q,0.0), min(max(q.x,max(q.y,q.z)),0.0) );
}

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

	float white = 13.4;

	//color *= overRange;
	//if(maxComponent(color) > white) color = vec3(1.0, 0.0, 0.0);

	//#if TAA_ToneMapping > OFF && defined(Enabled_TAA)
	//color = Uncharted2Tonemap(color*1.0);
	//#else
	color = Uncharted2Tonemap(color*30.0);
	//#endif

	color /= Uncharted2Tonemap(vec3(20.0));
	color = saturation(color, 1.02);

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
	float a = 0.0027;
	float b = float(0x9fff) / 65535.0;

	float lum = maxComponent(color);

	if(bool(step(lum, a))) color;

	return color/lum*((a*a-(2.0*a-b)*lum)/(b-lum));
}

vec4 GetColor(in sampler2D tex, in vec2 coord){
	return vec4(decodeGamma(texture2D(tex, coord).rgb), 1.0);
}

//from : https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
vec4 SampleTextureCatmullRom(sampler2D tex, vec2 uv, vec2 texSize )
{
		// We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
		// down the sample location to get the exact center of our "starting" texel. The starting texel will be at
		// location [1, 1] in the grid, where [0, 0] is the top left corner.
		vec2 invtexSize = 1.0 / texSize;

		vec2 samplePos = uv * texSize;
		vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

		// Compute the fractional offset from our starting texel to our original sample location, which we'll
		// feed into the Catmull-Rom spline function to get our filter weights.
		vec2 f = samplePos - texPos1;
		vec2 f2 = f * f;

		// Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
		// These equations are pre-expanded based on our knowledge of where the texels will be located,
		// which lets us avoid having to evaluate a piece-wise function.
		vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
		vec2 w1 = 1.0 + f2 * (-2.5 + 1.5*f);
		vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
		vec2 w3 = f2 * (-0.5 + 0.5 * f);

		// Work out weighting factors and sampling offsets that will let us use bilinear filtering to
		// simultaneously evaluate the middle 2 samples from the 4x4 grid.
		vec2 w12 = w1 + w2;
		vec2 offset12 = w2 / (w1 + w2);

		// Compute the final UV coordinates we'll use for sampling the texture
		vec2 texPos0 = texPos1 - vec2(1.0);
		vec2 texPos3 = texPos1 + vec2(2.0);
		vec2 texPos12 = texPos1 + offset12;

		texPos0 *= invtexSize;
		texPos3 *= invtexSize;
		texPos12 *= invtexSize;

		vec4 result = vec4(0.0);
		result += GetColor(tex, vec2(texPos0.x,  texPos0.y)) * saturate(w0.x * w0.y);
		result += GetColor(tex, vec2(texPos12.x, texPos0.y)) * saturate(w12.x * w0.y);
		result += GetColor(tex, vec2(texPos3.x,  texPos0.y)) * saturate(w3.x * w0.y);

		result += GetColor(tex, vec2(texPos0.x,  texPos12.y)) * saturate(w0.x * w12.y);
		result += GetColor(tex, vec2(texPos12.x, texPos12.y)) * saturate(w12.x * w12.y);
		result += GetColor(tex, vec2(texPos3.x,  texPos12.y)) * saturate(w3.x * w12.y);

		result += GetColor(tex, vec2(texPos0.x,  texPos3.y)) * saturate(w0.x * w3.y);
		result += GetColor(tex, vec2(texPos12.x, texPos3.y)) * saturate(w12.x * w3.y);
		result += GetColor(tex, vec2(texPos3.x,  texPos3.y)) * saturate(w3.x * w3.y);

		result.rgb /= result.a;

		return result;
}

vec4 cubic(float x) {
		float x2 = x * x;
		float x3 = x2 * x;
		vec4 w;
		w.x =   -x3 + 3.0*x2 - 3.0*x + 1.0;
		w.y =  3.0*x3 - 6.0*x2       + 4.0;
		w.z = -3.0*x3 + 3.0*x2 + 3.0*x + 1.0;
		w.w =  x3;
		return w / 6.0;
}

vec3 BicubicTexture(in sampler2D tex, in vec2 coord, in vec2 texSize) {
	coord *= texSize;
	//coord = floor(coord + 0.5);

	float fx = fract(coord.x);
	float fy = fract(coord.y);
	coord.x -= fx;
	coord.y -= fy;

	fx -= 0.5;
	fy -= 0.5;

	vec4 xcubic = cubic(fx);
	vec4 ycubic = cubic(fy);

	vec4 c = vec4(coord.x - 0.5, coord.x + 1.5, coord.y - 0.5, coord.y + 1.5);
	vec4 s = vec4(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
	vec4 offset = c + vec4(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;

	vec3 sample0 = decodeGamma(texture2D(tex, vec2(offset.x, offset.z) / texSize).rgb);
	vec3 sample1 = decodeGamma(texture2D(tex, vec2(offset.y, offset.z) / texSize).rgb);
	vec3 sample2 = decodeGamma(texture2D(tex, vec2(offset.x, offset.w) / texSize).rgb);
	vec3 sample3 = decodeGamma(texture2D(tex, vec2(offset.y, offset.w) / texSize).rgb);

	float sx = s.x / (s.x + s.y);
	float sy = s.z / (s.z + s.w);

	return mix( mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

void main(){
	vec2 texcoord = uv;

  vec3 color = texture2D(gaux2, texcoord).rgb;
	   color = decodeGamma(color);

	//if(texcoord.x > 0.5) color = decodeGamma(texture2D(composite, texcoord).rgb) * decodeHDR;

	//if(maxComponent(color) < 0.01953) color = vec3(1.0, 0.0, 0.0);

	#ifdef Enabled_TAA
		#if TAA_Post_Sharpen > 0
			vec3 colorIndex = vec3(0.0);

			for(float i = -1.0; i <= 1.0; i += 1.0){
				for(float j = -1.0; j <= 1.0; j += 1.0){
					if(i == 0.0 && j == 0.0) continue;
					colorIndex += decodeGamma(texture2D(colortex, texcoord + pixel * vec2(i, j)).rgb);
				}
			}

			colorIndex *= 0.125;

			color += (color - colorIndex) * 0.0025 * TAA_Post_Sharpen;
			color = clamp01(color);
		#endif

		//#if TAA_ToneMapping > OFF
		//color = invKarisToneMapping(color);
		//#endif
		//color = clamp01(color);
	#endif

	#if TAA_ToneMapping > OFF && defined(Enabled_TAA)
		color = invKarisToneMapping(color) / 30.0;
	#endif

	float vignette = distance(texcoord, vec2(0.5));

	#ifdef Enabled_Bloom
		vec3 bloom = vec3(0.0);

		float bloomWeight = 0.0;

		vec2 bloomCoord = texcoord;
		vec2 bloomOffset = pixel * 5.0;

		const float sigma = 16.0;

		for(int level = 2; level < 6; level++){
			float mipmap = exp2(float(level));

			float weight = exp(-pow2(mipmap) / (2.0 * sigma * sigma));
			bloomWeight += weight;

			bloom += decodeGamma(texture2D(gnormal, bloomCoord / mipmap + bloomOffset)) * weight;

			bloomOffset.x += (1.0 / mipmap) + pixel.x * mipmap + pixel.x * 5.0;
		}

		bloom /= bloomWeight;
		bloom *= decodeHDR;
		//bloom *= 100.0;

		color = mix(color, bloom * 100.0, 0.00048);
		//color = bloom;
		//color = decodeGamma(texture2D(gnormal, texcoord).rgb) * decodeHDR;
	#endif

	#ifdef Enabled_Exposure
		float exposure = texture2D(gaux2, vec2(0.5)).a;
			  //exposure = (1.0 - exp(-exposure)) * 30.0;

		color /= 0.125 + exposure * 2.0;
	#endif

	color = ToneMapping(color);
	//color = ACESToneMapping(color);

	if(texcoord.x > 0.5){
		//color = decodeGamma(texture2D(gaux2, texcoord).a) * decodeHDR;
		//color = 1.0 - exp(-color * 16.0);
		//color = decodeGamma(texture2D(gaux2, texcoord).a) * decodeHDR;
	}

	color = encodeGamma(color);

	//if(maxComponent(texture2D(composite, texcoord).rgb * decodeHDR) > 120.0) color = vec3(1.0, 0.0, 0.0);
	//if(maxComponent(decodeGamma(texture2D(composite, texcoord).rgb) * decodeHDR) > 1.0) color = vec3(1.0, 0.0, 0.0);
	//if(maxComponent(decodeGamma(texture2D(gaux2, texcoord).rgb) * 30.0) > 20.0) color = vec3(1.0, 0.0, 0.0);

	#ifdef RawOut
	color = texture2D(gaux2, texcoord).rgb;
	#endif

	//color = texture2D(shadowcolor1, (texcoord * 2.0 - 1.0) * 0.5 + 0.5).rgb;

  gl_FragColor = vec4(color, 1.0);
}

#version 130

#define Brightness 0.3

#define Enabled_Exposure

#define Enabled_Average_Exposure
#define Exposure_Vault 1.0			//[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]

#define Enabled_TAA
	#define TAA_Post_Sharpen 50 				//[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
	#define TAA_ToneMapping 

#define Enabled_Bloom								//Bloom, used on screen Effect and hurt Effects
	#define Bloom_Strength 1.0				//[0.08 0.12 0.16 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.3 2.6]
	#define Bloom_Blend 0.0417				//[0.0156 0.0208 0.0313 0.0417 0.0625 0.0833 0.125]
	#define Bloom_Sample_Lod 0.7			//[0.7 0.75 0.8 0.85 0.9 0.95 1.0]
	#define Only_Bloom		 0					//[0 1 2]

//#define RawOut

#define Screen_Bias 0.03

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D noisetex;

uniform sampler2D depthtex0;

#define colortex gaux2

uniform float frameTimeCounter;
uniform float viewWidth;
uniform float viewHeight;
uniform float rainStrength;
uniform float aspectRatio;

uniform int isEyeInWater;

uniform ivec2 eyeBrightnessSmooth;

uniform vec3 sunPosition;
uniform mat4 gbufferModelViewInverse;

#define Setting 1
#define Advanced Setting

uniform vec2 resolution;
uniform vec2 pixel;

in vec2 texcoord;

#include "/libs/common.inc"

#define Stage Final
#define bloomSampler gnormal

//#include "../libs/PostProcessing.glsl"

//vec2 uv = gl_TexCoord[0].st;

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

vec4 texture2D_bicubic(sampler2D tex, vec2 uv, in vec2 resolution) {
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

	vec2 p0 = (iuv + vec2(h0x, h0y)) / resolution * 0.5 + 0.5;
	vec2 p1 = (iuv + vec2(h1x, h0y)) / resolution * 0.5 + 0.5;
	vec2 p2 = (iuv + vec2(h0x, h1y)) / resolution * 0.5 + 0.5;
	vec2 p3 = (iuv + vec2(h1x, h1y)) / resolution * 0.5 + 0.5;

  vec4 r = g0(fuv.y) * (g0x * texture2D(tex, p0)  +
                        g1x * texture2D(tex, p1)) +
           g1(fuv.y) * (g0x * texture2D(tex, p2)  +
                        g1x * texture2D(tex, p3));

  return r;
}

vec4 cubic(float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec3 textureBicubic(sampler2D sampler, vec2 texCoords, vec2 texSize){
    vec2 invTexSize = 1.0 / texSize;

    texCoords = texCoords * texSize - 0.5;

    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec3 sample0 = texture2D(sampler, offset.xz).rgb;
    vec3 sample1 = texture2D(sampler, offset.yz).rgb;
    vec3 sample2 = texture2D(sampler, offset.xw).rgb;
    vec3 sample3 = texture2D(sampler, offset.yw).rgb;

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

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
	float white = 6.0;
	vec3 whiteMul = vec3(0.2725, 0.5242, 0.2033);

	color *= white;

	color = Uncharted2Tonemap(color);
	color /= Uncharted2Tonemap(vec3(white));
	color = mix(color, ACESToneMapping(color), 0.625);

	return color;
}

//https://graphicrants.blogspot.com/2013/12/tone-mapping.html
vec3 invKarisToneMapping(in vec3 color){
	return color = -color / (color - 1.0);
	/*
	float a = 0.1;
	float b = 1.0;

	float lum = maxComponent(color);

	if(lum <= a) return pow(color, vec3(2.0));

	color = color/lum*((a*a-(2.0*a-b)*lum)/(b-lum));
	color = max(vec3(0.0), color);

	return pow(color, vec3(2.0));
	*/
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

vec2 hash2(in vec2 p){
  return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

float voronoi(in vec2 x){
	//x = fract(x * 8.0) * 8.0;
	//x = vec2(cos(x.x * 2.0 * Pi), sin(x.y * 2.0 * Pi));
	//x = x * 8.0;

	float grid = 8.0;
	x *= grid;
	//x = fract(x) * 1.0;

  vec2 n = floor(x);
  vec2 f = fract(x);

  float md = 8.0;

  for(int i = -1; i <= 1; i++) {
    for(int j = -1; j <= 1; j++) {
      vec2 g = vec2(i, j);
	  vec2 c = n + g;

    if(c.x == -1.0)c.x = float(grid-1);
    else if(c.x == float(grid))c.x = 0.0;
                
    if(c.y == -1.0)c.y =float(grid-1);
    else if(c.y == float(grid))c.y = 0.0;

      vec2 o = hash2(c);

      vec2 r = g + o - f;
      float d = dot(r, r);

      md = min(d, md);
    }
  }

  return md;
}

void main(){
	//vec2 texcoord = uv;

  vec3 color = texture2D(gaux2, texcoord).rgb;
	   color = decodeGamma(color);

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

			vec3 sharpen = clamp(color - colorIndex, vec3(-0.25), vec3(0.25));

			color = max(vec3(0.0), color + sharpen * TAA_Post_Sharpen * 0.001);
		#endif
	#endif

	float vignette = distance(texcoord, vec2(0.5));

	#define BloomGaussBlurRadius 2.0

	#ifdef Enabled_TAA
	color = invKarisToneMapping(color);
	#endif

	#ifdef Enabled_Bloom
		const float filter_radius = 2.0;
		vec2 filter_radius_offset = pixel * (filter_radius * 2.0 + 1.0);

		vec3 bloom = vec3(0.0);
		float bloomWeight = 0.0;

		vec2 bloomCoord = texcoord;

		vec2 bloomOffset = filter_radius_offset;

		const float sigma = 16.0;
		float phi = 1.0 / (2.0 * sigma * sigma);

		for(int level = 1; level < 6; level++){
			float mipmap = exp2(float(level));

			float weight = 1.0;//exp(-pow2(mipmap) * phi);

			bloomWeight += weight;
			bloom += decodeGamma(textureBicubic(composite, bloomCoord / mipmap + bloomOffset, resolution)) * weight;

			bloomOffset.x += (1.0 / mipmap) + filter_radius_offset.x;
		}

		bloom /= bloomWeight;
		bloom *= 255.0;

		color = mix(color, bloom, Bloom_Blend);

		//color = mix(color, bloom * 100.0, 0.00048);
		//color = bloom;
		//color = decodeGamma(texture2D(gaux1, texcoord).rgb) * decodeHDR;
	#endif

	#ifdef Enabled_Average_Exposure
		float exposure = pow(texture2D(gaux2, vec2(0.5)).a, 2.2);
			  //exposure = (1.0 - exp(-exposure)) * 30.0;

		color /= max(0.02, exposure);

		//color /= max(0.02, 1.0 - exp(-maxComponent(color) * 1.0));

		//if(texcoord.x > 0.5)
		color *= mix(mix(vec3(1.0), normalize(vec3(0.782, 0.344, 1.0)), 0.25), vec3(1.0), saturate(exposure * 16.0));//normalize(1.0 - vec3(0.3029, 0.4287, 0.2684))
	#endif

	color *= Exposure_Vault;

	color = ToneMapping(color);
	//color = color / (color + 0.33);
	//color = ACESToneMapping(color);

	//if(texcoord.x > 0.5){
		//color = decodeGamma(texture2D(gaux2, texcoord).a) * decodeHDR;
		//color = 1.0 - exp(-color * 16.0);
		//color = decodeGamma(texture2D(gaux2, texcoord).a) * decodeHDR;

		//vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);
		//if(abs(sP.y) < 0.001) color = vec3(1.0, 0.0, 0.0);
	//}

	//color = texture2D(gaux2, vec2(0.5)).rgb;
	//color = decodeGamma(color);

	color = encodeGamma(color);

	//if(maxComponent(texture2D(composite, texcoord).rgb * decodeHDR) > 120.0) color = vec3(1.0, 0.0, 0.0);
	//if(maxComponent(decodeGamma(texture2D(composite, texcoord).rgb) * decodeHDR) > 1.0) color = vec3(1.0, 0.0, 0.0);
	//if(maxComponent(decodeGamma(texture2D(gaux2, texcoord).rgb) * 30.0) > 20.0) color = vec3(1.0, 0.0, 0.0);

	#ifdef RawOut
	color = texture2D(gaux2, texcoord).rgb;
	#endif

	//if(texcoord.x > 0.8)
	//color = texture2D(shadowcolor0, texcoord).rgb;

	//color = vec3(texture(noisetex, texcoord).x);
	//color = vec3(voronoi(texcoord / vec2(1.0, aspectRatio) * 12.0));

	gl_FragColor = vec4(color, 1.0);
}

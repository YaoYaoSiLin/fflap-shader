#version 130

#define Enabled_TAA
	#define TAA_Sharpen_Factor 0			//[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
	#define TAA_Color_Sampler_Size 0.55	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]

#define Enabled_Bloom								//Bloom, used on screen Effect and hurt Effects
	#define Bloom_Strength 1.0				//[0.08 0.12 0.16 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.3 2.6]
	#define Bloom_Sample_Lod 0.7			//[0.7 0.75 0.8 0.85 0.9 0.95 1.0]
	#define Only_Bloom		 0					//[0 1 2]

//#define RawOut

#define Screen_Bias 0.03

uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D gaux2;
uniform sampler2D gaux3;
uniform sampler2D gaux4;

uniform sampler2D depthtex0;

uniform sampler2D noisetex;

uniform float frameTimeCounter;
uniform float viewWidth;
uniform float viewHeight;
uniform float rainStrength;
uniform float aspectRatio;

//uniform int frameCounter;

uniform ivec2 eyeBrightnessSmooth;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform float getHereMedic;
//uniform int RUA;
uniform float waterFall;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#include "libs/common.inc"

#define Stage Final
#define bloomSampler gaux3

//#include "libs/PostProcessing.glsl"

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

vec3 ACESToneMapping(in vec3 color, in float adapted_lum) {
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	//color = pow(color, vec3(2.2));
	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
	//return pow((color * (A * color + B)) / (color * (C * color + D) + E), vec3(1.0 / 2.2));
}

vec3 Uncharted2Tonemap(in vec3 color, in float x) {
	float A = 2.51;
	float B = 0.59;
	float C = 0.01;
	float D = 0.14;
	float E = 0.0001;
	float F = 0.15;

	//color = L2rgb(color);

	vec3 color2 = ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))/E/F;
			 color2 /= ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))/E/F;
			 //color2 = color2 * 0.249;

	float l = getLum(color2) * 0.999 + 0.001;

	color = color / (color * 0.999 + 0.001) * color2;

	return (color);
}

float noiseTexture5(in vec2 noiseCoord, float t){
	noiseCoord = noiseCoord * vec2(1.0, 0.56) * 0.33 + vec2(0.0, t);


	//noiseCoord.x *= distance(noiseCoord, vec2(0.5));

	//noiseCoord /= 64.0;

	float n  = (texture2D(noisetex, noiseCoord / 4.0).x) * 4.0; noiseCoord.y += t * 0.03;
			  n += (texture2D(noisetex, noiseCoord / 2.0).x) * 2.0; noiseCoord.y += t * 0.02;
				n += (texture2D(noisetex, noiseCoord).x);						 noiseCoord.x += t * 0.01;
				n += (texture2D(noisetex, noiseCoord * 2.0).x) * 0.5; noiseCoord.x -= t * 0.02;
				n += (texture2D(noisetex, noiseCoord * 4.0).x) * 0.25;
			  n = n / 7.75;
				n = fract(n) * 2.0 - 1.0;
				n = sin(n) * cos(n);

	//n *= mix(sin((uv.x * 2.0 - 1.0) * viewWidth * 0.003), 1.0, max(0.0, uv.y - 0.5) * 2.0);

	return n;
}

void main(){
	vec2 res = vec2(viewWidth, viewHeight);

	vec2 texcoord = uv;

	vec4 waterColor = vec4(0.0);

//////
	vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord * 2.0 - 1.0, 1.0));
			 vP /= vP.w;

	if(waterFall > 0.001){
		float t = frameTimeCounter * 0.15;

		vec2 noiseCoord = texcoord;

		float noiseTexture = noiseTexture5(texcoord * vec2(1.0, 0.5), t);
		float noiseTextureA = noiseTexture5((texcoord + vec2(0.2, 0.0)) * vec2(1.0, 0.5), t);
		float noiseTextureB = noiseTexture5((texcoord + vec2(0.0, 0.2)) * vec2(1.0, 0.5), t);

		vec3 n = vec3(0.0);
	/*
				 n = cross(
					   vec3(0.1, 4 * (noiseTexture.x - noiseTextureB.x) / 0.1, 0.0),
						 vec3(0.0, 4 * (noiseTexture.x - noiseTextureA.x) / 0.1, 0.1)
					   );
				n.z = 0.0;
	*/

		float waterFallAlive = max(0.0, pow(waterFall, 0.9) * 1.05 - 0.05);

		n.y += dot(vec3(0.25 * (noiseTexture - noiseTextureB) / 0.2), vec3(0.0, 1.0, 0.0));
		n.x += dot(vec3(0.25 * (noiseTexture - noiseTextureA) / 0.2), vec3(1.0, 0.0, 0.0));
		n.xy *= 1.0 - dot(n.xy, n.xy);
		n.z = 0.0;
		waterColor = vec4(vec3(0.02), waterFallAlive * dot(n.xy, vec2(0.0, 1.0)));
		n = mat3(gbufferModelView) * n;

		vec3 rVPN = refract((vP.xyz), (n), 1.333 / 1.000293);

		vec3 vPc = nvec3(gbufferProjectionInverse * nvec4(vec3(vec2(0.5), 1.0) * 2.0 - 1.0));
	       vPc = normalize(vPc);
	       vPc = mat3(gbufferModelViewInverse) * vPc;

		//if(rVPN == vec3(0.0)) waterColor = vec4(vec3(0.02), 1.0);

		vP.xyz = vP.xyz + rVPN * pow(1.0 - min(abs(vPc.y), 1.0), 3.0) * 16.0 * waterFallAlive;
		//texcoord = nvec3(gbufferProjection * nvec4(vP.xyz)).xy * 0.5 + 0.5;
	}

	/*
	if(waterFall > 0.01){
	vec2 noiseCoord = texcoord * vec2(1.0, 0.56) + vec2(0.0, frameTimeCounter * 0.13);

	float angle = clamp(abs(texture2D(gaux2, texcoord).a * 2.0 - 1.0), 0.0, 0.97);

	float waterFallNoise  = texture2D(noisetex, noiseCoord / 4.0).x * 4.0;
			  waterFallNoise += texture2D(noisetex, noiseCoord / 2.0).x * 2.0;
				waterFallNoise += texture2D(noisetex, noiseCoord).x;
				waterFallNoise += texture2D(noisetex, noiseCoord * 2.0).x * 0.5;
				waterFallNoise += texture2D(noisetex, noiseCoord * 4.0).x * 0.25;
				waterFallNoise = waterFallNoise / 7.75 - 0.5;

	texcoord += waterFallNoise / res * 128.0 * waterFall * (1.0 - angle);
	}
	*/

  vec3 color = (texture2D(gaux2, texcoord).rgb);

	#if defined(Enabled_TAA) && TAA_Sharpen_Factor > 0
		vec3 colorIndex = vec3(0.0);

		for(float i = -1.0; i <= 1.0; i += 1.0){
			for(float j = -1.0; j <= 1.0; j += 1.0){
				colorIndex += texture2D(gaux2, texcoord + pixel * vec2(i, j) * TAA_Color_Sampler_Size).rgb;
			}
		}

		colorIndex -= color;
		colorIndex *= 0.125;

		//if(texcoord.x < 0.5)
		color += (color - colorIndex) * 0.025 * TAA_Sharpen_Factor * min(1.0, abs(254.0 - round(texture2D(gdepth, texcoord).z * 255.0)));
		color = clamp01(color);
	#endif

	//if(texcoord.x < 0.5) color = texture2D(gaux2, texcoord).aaa;

	color = rgb2L(color);

	//if(int(texture2D(gdepth, texcoord).z * 255) == 254) color = vec3(0.0);
	//color = vec3(texture2D(gdepth, texcoord).z);


/*
	vec3 avgColor = vec3(0.0);

	for(float i = -1.0; i <= 1.0; i += 1.0){
		for(float j = -1.0; j <= 1.0; j += 1.0){
			if(i != 0.0 && j != 0.0)
			avgColor += rgb2L(texture2D(gaux2, texcoord + pixel * vec2(i, j) * 0.61).rgb);
		}
	}

	avgColor /= 8.0;

	color += (color - avgColor) * (Sharpen_Factor / 100.0);
	color = (clamp01(color));
*/
	//color *= 5.0;

	//color = clamp(color, vec3(0.0), vec3(1.0));
	//color = pow(color, vec3(2.2));

	float vignetteInHurt = pow(distance(texcoord, vec2(0.5)), 2.4);

	//color = mix(color, bloom, vignetteInHurt * getHereMedic);
	//color = mix(color, vec3(dot(bloom * 0.1, vec3(0.2126, 0.7152, 0.0722))), float(RUA));

	//color = mix(color, waterColor.rgb, pow(max(0.0, waterColor.a), 2.0));

	//color = color / (color + 0.84);
	//color = Uncharted2Tonemap(color, 1.0);

	#ifdef Enabled_Bloom
	//CalculateBloom(color, texcoord);
	//color = texture2D(gaux3, texcoord).rgb;
	#endif

	//color *= 2.0;

	//color = color / (color + 0.679);
	color = (color / (1.0 + getLum(color))) / (1.0 / (1.0 + getLum(color)));
	color *= 3.0;

	color = Uncharted2Tonemap(color, 1.0);
	//color = ACESToneMapping(color, 1.0);

	float lum = getLum(color);
	color = clamp01(lum + (color - lum) * 1.04);

	//color += vec3(test);

	color = L2rgb(color);

	#ifdef RawOut
		color = texture2D(gaux4, texcoord).rgb * 2.0;
	#endif
/*
	texcoord = texcoord * 2.0 - 1.0;
	texcoord *= mix(1.0, length(texcoord), Screen_Bias);
	texcoord = clamp(texcoord * 0.5 + 0.5, pixel, 1.0 - pixel);
	color = texture2D(gaux2, texcoord).rgb;
*/
  gl_FragColor = vec4(color, 1.0);
}

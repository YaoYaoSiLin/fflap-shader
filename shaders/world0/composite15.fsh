#version 130

#extension GL_ARB_shader_texture_lod : require

uniform sampler2D gdepth;
uniform sampler2D composite;
uniform sampler2D gaux2;
uniform sampler2D gaux4;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

#define lastColorSampler gaux4
#define colorSampler gaux2

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform float frameTimeCounter;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

in vec2 texcoord;

const bool gaux4Clear = false;
const bool compositeMipmapEnabled = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/jittering.glsl"

#define aaDepthTexture depthtex0
#define aaDepthComp 0

#include "../libs/antialiased.glsl"

/* DRAWBUFFERS:57 */

#define TAA_ToneMapping 0x7ff 		//[OFF 0xff 0x1ff 0x2ff 0x3ff 0x7ff 0xfff 0x1ffff 0x3fff 0x7fff 0xffff]

//https://graphicrants.blogspot.com/2013/12/tone-mapping.html
vec3 invKarisToneMapping(in vec3 color){
	float a = 1e-4;
	float b = float(1023.0) / 65535.0;

	float lum = maxComponent(color);

	if(bool(step(lum, a))) color;

	return color/lum*((a*a-(2.0*a-b)*lum)/(b-lum));
}

void main(){
  vec3 antialiased = texture2D(gaux2, texcoord).rgb;

  #ifdef Enabled_TAA
    antialiased = CalculateTAA(gaux2, gaux4);
  #endif

  float exposureCurr = luminance(texture2DLod(composite, vec2(0.5 + jitter), viewWidth).rgb);
        exposureCurr = pow(exposureCurr, 2.2) * decodeHDR * 16.0 * 3.0;
        exposureCurr = 1.0 - exp(-exposureCurr);
        //exposureCurr = exposureCurr / (exposureCurr + 1.0);

  float exposurePrev = texture2D(gaux4, vec2(0.5)).a;

  float maxExp = max(exposureCurr, exposurePrev);
  float minExp = min(exposureCurr, exposurePrev);

  //help me plz
  float exposure = float(1 + frameCounter) / (0.01666 + frameTimeCounter);
        exposure = clamp(0.7071 / exposure, 0.008, 0.48);
        exposure = mix(exposurePrev, exposureCurr, exposure);
  //if(exposurePrev == 0.0) exposure = 0.5;

  //antialiased = texture2D(gaux2, texcoord + jitter).rgb;

  //antialiased = decodeGamma(antialiased);

  //vec3 buff = antialiased;
  //#if TAA_ToneMapping > OFF && defined(Enabled_TAA)
  ////buff = invKarisToneMapping(buff);
  //#endif

  //buff = encodeGamma(buff);
  //antialiased = encodeGamma(antialiased);

  gl_FragData[0] = vec4(antialiased, exposure);
  gl_FragData[1] = vec4(antialiased, exposure);
}

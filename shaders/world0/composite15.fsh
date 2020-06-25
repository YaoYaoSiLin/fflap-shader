#version 130

#extension GL_ARB_shader_texture_lod : require

uniform sampler2D gdepth;
uniform sampler2D gaux1;
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
const bool gaux2MipmapEnabled = true;
const bool gaux4MipmapEnabled = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/jittering.glsl"

#define aaDepthTexture depthtex0
#define aaDepthComp 0

#include "../libs/antialiased.glsl"

void main(){
/* DRAWBUFFERS:57 */

  vec3 antialiased = texture2D(gaux2, texcoord).rgb;

  #ifdef Enabled_TAA
    antialiased = CalculateTAA(gaux2, gaux4);
  #endif

  float exposureCurr = dot03(texture2DLod(gaux2, vec2(0.5), viewWidth).rgb);
  float exposurePrev = texture2D(gaux4, vec2(0.5)).a;

  float maxExp = max(exposureCurr, exposurePrev);
  float minExp = min(exposureCurr, exposurePrev);

  //help me plz
  float exposure = float(1 + frameCounter) / (0.01666 + frameTimeCounter);
        exposure = clamp(0.7071 / exposure, 0.008, 0.48);
        exposure = mix(exposurePrev, exposureCurr, exposure);
  //if(exposurePrev == 0.0) exposure = 0.5;

  gl_FragData[0] = vec4(antialiased, exposure);
  gl_FragData[1] = vec4(antialiased, exposure);
}

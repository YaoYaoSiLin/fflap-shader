#version 130

#extension GL_ARB_shader_texture_lod : require

#define TAA_Color_Sampler_Size 0.55   	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]
#define TAA_Depth_Sampler_Size 0.55   	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]

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

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

in vec2 texcoord;

const bool gaux4Clear = false;
const bool gaux2MipmapEnabled = true;
const bool gaux4MipmapEnabled = true;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "libs/common.inc"
#include "libs/jittering.glsl"

#define aaDepthTexture depthtex0
#define aaDepthComp 0

#include "libs/antialiased.glsl"

void main(){
/* DRAWBUFFERS:57 */

  vec3 antialiased = texture2D(gaux2, texcoord).rgb * overRange;
  vec3 colorLastFrame = vec3(0.0);

  #ifdef Enabled_TAA
    antialiased = CalculateTAA(gaux2, gaux4, 1.0, 0.03125, 10.0);
  #endif

  //float exposure = mix(dot03(texture2DLod(gaux2, vec2(0.5), viewWidth).rgb) * 38.0, dot03(texture2DLod(gaux4, vec2(0.5), viewWidth).rgb), 0.99);

  gl_FragData[0] = vec4(antialiased, 1.0);
  gl_FragData[1] = vec4(antialiased, 1.0);
}

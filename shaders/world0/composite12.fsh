#version 130

#define SMAA_PRESET_HIGH

#if defined(SMAA_PRESET_LOW)
  #define SMAA_THRESHOLD 0.15
  #define SMAA_MAX_SEARCH_STEPS 4
  #define SMAA_DISABLE_DIAG_DETECTION
  #define SMAA_DISABLE_CORNER_DETECTION
#elif defined(SMAA_PRESET_MEDIUM)
  #define SMAA_THRESHOLD 0.1
  #define SMAA_MAX_SEARCH_STEPS 8
  #define SMAA_DISABLE_DIAG_DETECTION
  #define SMAA_DISABLE_CORNER_DETECTION
#elif defined(SMAA_PRESET_HIGH)
  #define SMAA_THRESHOLD 0.1
  #define SMAA_MAX_SEARCH_STEPS 16
  #define SMAA_MAX_SEARCH_STEPS_DIAG 8
  #define SMAA_CORNER_ROUNDING 25
#elif defined(SMAA_PRESET_ULTRA)
  #define SMAA_THRESHOLD 0.05
  #define SMAA_MAX_SEARCH_STEPS 32
  #define SMAA_MAX_SEARCH_STEPS_DIAG 16
  #define SMAA_CORNER_ROUNDING 25
#endif

uniform sampler2D gaux2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define Gaussian_Blur

#include "../libs/common.inc"

#define fma(a, b, c) a * b + c
#define mad(a, b, c) a * b + c

void SMAAEdgeDetectionVS(in vec2 texcoord, out vec4 offset[3]){
  offset[0] = fma(pixel.xyxy, vec4(-1.0, 0.0, 0.0, -1.0), texcoord.xyxy);
  offset[1] = fma(pixel.xyxy, vec4( 1.0, 0.0, 0.0,  1.0), texcoord.xyxy);
  offset[2] = fma(pixel.xyxy, vec4(-2.0, 0.0, 0.0, -2.0), texcoord.xyxy);
}


vec2 SMAAColorEdgeDetectionPS(in vec2 texcoord){
  vec4 offset[3];
  SMAAEdgeDetectionVS(texcoord, offset);

  vec2 threshold = vec2(SMAA_THRESHOLD);

  vec4 delta;
  vec3 C = texture2D(gaux2, texcoord).rgb;

  vec3 Cleft = texture2D(gaux2, offset[0].xy).rgb;
  vec3 t = abs(C - Cleft);
  delta.x = maxComponent(t);

  vec3 CTop = texture2D(gaux2, offset[0].zw).rgb;
  t = abs(C - CTop);
  delta.y = maxComponent(t);

  vec2 edges = step(threshold, delta.xy);
  if(dot(edges, vec2(1.0)) == 0.0) return vec2(-1.0);

  vec3 Cright = texture2D(gaux2, offset[1].xy).rgb;

  return vec2(0.0);
}

void main(){
  vec4 offset[3];
  SMAAEdgeDetectionVS(texcoord, offset);
  vec3 color = texture2D(gaux2, offset[0].xy).rgb;

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color, 1.0);
}

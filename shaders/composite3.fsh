#version 130

#extension GL_ARB_shader_texture_lod : require

uniform sampler2D gdepth;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform int isEyeInWater;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

in vec2 texcoord;

in vec4 waterColor;

const bool gaux3Clear = false;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "libs/common.inc"
#include "libs/jittering.glsl"

#define Reflection_Scale Medium                 //[Low Medium High]
  #define Reflection_Scale_Type Checker_Board   //[Checker_Board Render_Scale]
  #define Reflection_Filter     Temporal        //[Simple_Filter Temporal]

#if Reflection_Scale == Low
  #define reflection_resolution_scale 4
#elif Reflection_Scale == Medium
  #define reflection_resolution_scale 2
#elif Reflection_Scale == High
  #define reflection_resolution_scale 1
#endif

#if reflection_resolution_scale > 1
vec4 GetCheckerBoardColor(in sampler2D colorSampler, in vec2 coord){
  vec2 fragCoord = floor(coord * resolution);
  //vec2 checkerBoard = vec2(mod(fragCoord.x, 2), mod(fragCoord.y, 2));

  vec4 color = texture2D(colorSampler, coord);
/*
  if(texcoord.x < 0.5 && checkerBoard.y < 0.5){
    color = vec4(1.0, 0.0, 0.0, 1.0);
    color  = texture2D(colorSample, texcoord - vec2(pixel.x / scale * (1.0 - checkerBoard.y), 0.0));
    color += texture2D(colorSample, texcoord - vec2(0.0, pixel.y / scale * (1.0 - checkerBoard.y)));
    color += texture2D(colorSample, texcoord + vec2(pixel.x / scale * (1.0 - checkerBoard.y), 0.0));
    color += texture2D(colorSample, texcoord + vec2(0.0, pixel.y / scale * (1.0 - checkerBoard.y)));

    float weight = (1.0 - checkerBoard.x) * 2.0 + (1.0 - checkerBoard.y) * 2.0;

    if(weight > 0.0) color /= weight;
  }
*/
  //if(checkerBoard.x + checkerBoard.y > 0.5){
  //  color = vec4(1.0);
  //}
  /*
  if(checkerBoard.x + checkerBoard.y > 1.5){
    //color = vec4(1.0, 0.0, 0.0, 1.0);
    color  = texture2D(colorSample, coord + checkerBoard / scale * pixel);
    color += texture2D(colorSample, coord - checkerBoard / scale * pixel);
    color *= 0.5;

    //float weight = 2.0;
    //if(weight > 0.0) color /= weight;
  }else
  */

  vec3 checkerBoard = vec3(mod(fragCoord.x, 2), mod(fragCoord.y, 2), 0.0);

  #if reflection_resolution_scale == 2
  //color = vec4(0.0);

  checkerBoard.z = mod(fragCoord.x + fragCoord.y, 2);

  if(checkerBoard.z > 0.5){

    color  = texture2D(colorSampler, coord + vec2(pixel.x, 0.0));
    color += texture2D(colorSampler, coord + vec2(0.0, pixel.y));
    color *= 0.5;
    //color /= checkerBoard.z;
    //float weight = (checkerBoard.x) + (checkerBoard.y);
    //if(weight > 0.0) color /= weight;

    //color = vec4(1.0);
  }
  #endif

  #if reflection_resolution_scale == 4
  checkerBoard.z = mod(fragCoord.x * fragCoord.y, 2);
  checkerBoard.xy = 1.0 - checkerBoard.xy;

  if(checkerBoard.z < 0.5){
    color += texture2D(colorSampler, coord + (checkerBoard.xy * pixel));
    color += texture2D(colorSampler, coord - (checkerBoard.xy * pixel));
    color *= 0.5;
    //color *= 0.
    //color /= 1.0 + max(checkerBoard.x,checkerBoard.y);
    //color /= 1.0 + (1.0 - checkerBoard.x) * (1.0 - checkerBoard.y);
    //color = vec4(1.0);
  }

  /*
  if(mod(fragCoord.x + fragCoord.y, 2) > 0.5){
    color  = texture2D(colorSample, coord + vec2((checkerBoard.x) * pixel.x, 0.0));
    color += texture2D(colorSample, coord + vec2(0.0, (checkerBoard.y) * pixel.y));
    float weight = (checkerBoard.x) + (checkerBoard.y);
    if(weight > 0.0) color /= weight;
  }else if(checkerBoard.x + checkerBoard.y > 0.5){
    color  = texture2D(colorSample, coord + (checkerBoard.xy * pixel));
    color += texture2D(colorSample, coord - (checkerBoard.xy * pixel));
    color *= 0.5;
  }
  */
  #endif


/*
  if(checkerBoard.x < 0.5){
    //color = vec4(1.0, 0.0, 0.0, 1.0);

    color = texture2D(colorSample, texcoord + vec2((1.0 - checkerBoard.x) / scale * pixel.x, 0.0));
    color += texture2D(colorSample, texcoord + vec2(0.0, (1.0 - checkerBoard.y) / scale * pixel.y));

    float weight = (1.0 - checkerBoard.x) + (1.0 - checkerBoard.y);
    if(weight > 0.0) color /= weight;
  }else if(checkerBoard.y > 0.5){
    color = texture2D(colorSample, texcoord + vec2((checkerBoard.x) / scale * pixel.x, 0.0));
    color += texture2D(colorSample, texcoord + vec2(0.0, (checkerBoard.y) / scale * pixel.y));

    float weight = (checkerBoard.x) + (checkerBoard.y);
    if(weight > 0.0) color /= weight;
  }
*/
/*
  if(checkerBoard.y < 0.5){
    color  = texture2D(colorSample, texcoord - vec2(pixel.x / scale * (1.0 - checkerBoard.y), 0.0)) * 0.5
           + texture2D(colorSample, texcoord - vec2(0.0, pixel.y / scale * (1.0 - checkerBoard.y))) * 0.5;
  }else if(checkerBoard.x < 0.5){
    color  = texture2D(colorSample, texcoord + vec2(pixel.x / scale * (1.0 - checkerBoard.x), 0.0)) * 0.5
           + texture2D(colorSample, texcoord + vec2(0.0, pixel.y / scale * (1.0 - checkerBoard.x))) * 0.5;
  }
*/

  return color;
}
#endif

/*
vec3 CalculateTemporalReflection(){
  vec3 reflection = vec3(0.0);

  vec4 pvP = gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0);
       pvP /= pvP.w;
       pvP = gbufferModelViewInverse * pvP;
       pvP.xyz += cameraPosition - previousCameraPosition;
       pvP = gbufferPreviousModelView * pvP;
       pvP = gbufferPreviousProjection * pvP;
       pvP /= pvP.w;

  float depthFix = min(1.0, length(vP.xyz));
  if(int(round(texture2D(gdepth, texcoord).z * 255.0)) == 254.0) depthFix *= MC_HAND_DEPTH * MC_HAND_DEPTH;

  vec2 previousCoord = (pvP.xy * 0.5 + 0.5);
       previousCoord = (unjitterUV - previousCoord) * depthFix;
       previousCoord = -previousCoord + texcoord;

  #if reflection_resolution_scale == 1
    vec4 reflectionResult = texture2D(gaux1, texcoord);
    float depth = step(texture2D(gaux1, previousCoord).a, reflectionResult.a);
  #else
    vec4 reflectionResult = GetCheckerBoardColor(gaux1, texcoord);

    float depth = reflectionResult.a;
          depth = step(GetCheckerBoardColor(gaux1, previousCoord).a, depth);
  #endif

  reflection = reflectionResult.rgb * overRange;
  reflection = mix(reflection, texture2D(gaux3, previousCoord).rgb, 0.875 * depth);

  return reflection;
}
*/
void main(){
  vec3 color = texture2D(gaux2, texcoord).rgb;

  bool isSky = int(round(texture2D(gdepth, texcoord).z * 255.0)) == 255;

  vec2 unjitterUV = texcoord - R2sq2[int(mod(frameCounter, 16))] * pixel;
  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));

  //vec2 coord = nvec3(gbufferProjection * nvec4(vP + vP * vec3(.0, .0, 0.5))).xy * 0.5 + 0.5;
  //color = texture2D(gaux2, coord).rgb;

  vec3 reflection = vec3(0.0);

  if(!isSky){
    vec3 f = texture2D(composite, texcoord).rgb;
    float brdf = texture2D(composite, texcoord).a;

    #if Reflection_Filter == Temporal
      vec4 pvP = gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0);
           pvP /= pvP.w;
           pvP = gbufferModelViewInverse * pvP;
           pvP.xyz += cameraPosition - previousCameraPosition;
           pvP = gbufferPreviousModelView * pvP;
           pvP = gbufferPreviousProjection * pvP;
           pvP /= pvP.w;

      float depthFix = min(1.0, length(vP.xyz));
      if(int(round(texture2D(gdepth, texcoord).z * 255.0)) == 254.0) depthFix *= MC_HAND_DEPTH * MC_HAND_DEPTH;

      vec2 previousCoord = (pvP.xy * 0.5 + 0.5);
           previousCoord = (unjitterUV - previousCoord) * depthFix;
           previousCoord = -previousCoord + texcoord;

      #if reflection_resolution_scale == 1
        vec4 reflectionResult = texture2D(gaux1, texcoord);
        float lastFrameDepth = texture2D(gaux1, previousCoord).a;
        float depth = float(lastFrameDepth < reflectionResult.a && lastFrameDepth + 0.001 > reflectionResult.a);
      #else
        vec4 reflectionResult = GetCheckerBoardColor(gaux1, texcoord);

        float lastFrameDepth = texture2D(gaux1, previousCoord).a;
        float depth = float(lastFrameDepth < reflectionResult.a && lastFrameDepth + 0.001 > reflectionResult.a);
      #endif

      reflection = reflectionResult.rgb * overRange;
      reflection = mix(reflection, texture2D(gaux3, previousCoord).rgb, 0.875 * depth);
    #elif Reflection_Filter == Simple_Filter
      #if reflection_resolution_scale == 1
        reflection = texture2D(gaux1, texcoord).rgb * overRange;
      #else
        reflection = GetCheckerBoardColor(gaux1, texcoord).rgb * overRange;
      #endif
    #endif

    color += reflection / overRange * f * brdf;
  }

/* DRAWBUFFERS:56 */
  gl_FragData[0] = vec4(color, 1.0);
  gl_FragData[1] = vec4(reflection.rgb, 1.0);
}

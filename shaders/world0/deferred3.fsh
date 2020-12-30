#version 130

#define SHADOW_MAP_BIAS 0.9

uniform sampler2D depthtex0;
uniform sampler2D depthtex2;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform int frameCounter;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

const vec4 bitShL = vec4(16777216.0, 65536.0, 256.0, 1.0);
const vec4 bitShR = vec4(1.0/16777216.0, 1.0/65536.0, 1.0/256.0, 1.0);

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"

void main(){
  vec2 uv = texcoord;
       //uv = uv * 2.0 - 1.0;
       //uv /= mix(1.0, length(uv), SHADOW_MAP_BIAS) / 0.95;
       //uv = uv * 0.5 + 0.5;
       uv *= 0.8;

  vec4 data = vec4(texture(shadowtex1, uv).x, vec3(0.0));

  //data.xy = fract(data.x * vec2(256.0, 1.0));
  //data.y -= data.x / 256.0;
  data.xy = fract(data.x * bitShL.zw);
  data.y -= data.x / 256.0;
  data.z = texture(shadowtex1, uv).x;

  vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
  vec4 worldPosition = gbufferModelViewInverse * nvec4(viewPosition);

  vec4 shadowCoord = shadowProjection * shadowModelView * worldPosition;
       shadowCoord /= shadowCoord.w;
       shadowCoord = shadowCoord * 0.5 + 0.5;

  vec2 shadowMapPixel = vec2(2048.0, 1.0 / 2048.0);

  float receiver = shadowCoord.z - shadowMapPixel.y;

  float shading = 0.0;

  vec2 dither = vec2(GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jittering), GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y * 0.5, jittering));

      for(int i = 0; i < 4; i++){
        float angle = float(i + 1.0) * 0.25 * 2.0 * Pi;
        vec2 direction = vec2(cos(angle), sin(angle));
             direction = RotateDirection(direction, dither);
             direction *= 2.0;

        vec2 coord = shadowCoord.xy + direction * shadowMapPixel.y;
             coord = coord * 2.0 - 1.0; float distortion = mix(1.0, length(coord), SHADOW_MAP_BIAS) / 0.95;
             coord /= mix(1.0, length(coord), SHADOW_MAP_BIAS) / 0.95;
             coord = coord * 0.5 + 0.5;
             coord.xy = coord.xy * 0.8;

        if(coord.x < shadowMapPixel.y || coord.y < shadowMapPixel.y || coord.x > 0.8 - shadowMapPixel.y || coord.y > 0.8 - shadowMapPixel.y) continue;

        float depth = texture(shadowtex0, coord).x;

        shading += step(receiver - shadowMapPixel.y * 0.0, depth);
      }

  shading *= 0.25;

  data.rgb = vec3(shading);

  if(bool(step(shading, 0.01))) data.rgb = vec3(1.0, .0, 0.0);
  if(bool(step(0.9, shading))) data.rgb = vec3(1.0, 0.0, 1.0);

  //data.z = shading;
  //data.a = 1.0;

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = data;
}

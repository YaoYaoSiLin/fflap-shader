#version 130

#define SSAO_Steps 16    //[8 16 24]

#define gdepth colortex1
#define composite colortex3
#define gnormal colortex2
#define gaux2 colortex5

uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;

uniform sampler2D shadowtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 shadowModelViewInverse;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

uniform int frameCounter;

in vec2 texcoord;

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"

float GetDepth(in vec2 coord){
  float depth = texture(depthtex0, coord).x;
  float depthParticle = texture(gaux2, coord).x;

  if(0.0 < depthParticle && depthParticle < depth) depth = depthParticle;
  return depth;
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance = 140.0;

float shadowPixel = 1.0 / float(shadowMapResolution);

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

vec3 wP2sP(in vec4 wP, out float bias){
	vec4 sP = shadowModelView * wP;
       sP = shadowProjection * sP;
       sP /= sP.w;

  bias = 1.0 / (mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95);

  sP.xy *= bias;
  sP.z /= max(far / shadowDistance, 1.0);
  sP = sP * 0.5 + 0.5;

	return sP.xyz;
}

vec2 GetCoord(in vec2 coord){
  //vec2 resolution = vec2(viewWidth, viewHeight);

  //coord = round(coord * resolution * 0.25) / resolution * 4.0;
  coord = coord * 2.0 - 1.0;
  //coord.x *= aspectRatio;
  //coord *= 0.125;
  float bias = 1.0 / (mix(1.0, length(coord), SHADOW_MAP_BIAS) / 0.95);
  coord = coord * bias * 0.5 + 0.5;

  //float dither = bayer_16x16(texcoord, resolution);

  return round(coord * float(shadowMapResolution)) * shadowPixel;
}

#ifndef MC_RENDER_QUALITY
  #define MC_RENDER_QUALITY 1.0
#endif

/* DRAWBUFFERS:5 */

void main() {
  vec4 color = vec4(0.0);
  #if 1

  if(texcoord.x < 0.5 && texcoord.y < 0.5){
    vec2 coord = GetCoord(texcoord*2.0);
    color.rgb = texture2D(shadowcolor1, coord).rgb;
  }
  if(texcoord.x > 0.5 && texcoord.y < 0.5){
    vec2 coord = GetCoord(texcoord*2.0-vec2(1.0,0.0));
    //color = normalDecode(texture2D(shadowcolor0, coord).gb);
    color.rgb = texture2D(shadowcolor0, coord).rgb * 2.0 - 1.0;
    color.rgb = mat3(shadowModelView) * color.rgb;
    color.rgb = color.rgb * 0.5 + 0.5;
  }
  if(texcoord.x < 0.5 && texcoord.y > 0.5){
    vec2 coord = GetCoord(texcoord*2.0-vec2(0.0,1.0));
    color.r = texture(shadowtex0, coord).r;
  }
  //if()

  //float depth = GetDepth(texcoord);

  #else
    vec2 fragCoord = floor(texcoord * resolution);
    bool checkerBoard = bool(mod(fragCoord.x + fragCoord.y, 2));

    //if()
    //color = vec3(float(checkerBoard)) * 0.1;
    if(checkerBoard){
      vec2 coord = GetCoord(texcoord);
      //color.rgb = texture2D(shadowcolor1, coord).rgb;

      for(float i = -1.0; i <= 1.0; i += 1.0){
        for(float j = -1.0; j <= 1.0; j += 1.0){
          color.rgb += texture2D(shadowcolor1, coord + vec2(i, j) * pixel).rgb;
        }
      }

      color.rgb /= 9.0;

    }else{
      //color = vec4(1.0, 0.0, 0.0, 0.0);
      vec2 coord = GetCoord(texcoord - vec2(pixel.x, 0.0));

      for(float i = -1.0; i <= 1.0; i += 1.0){
        for(float j = -1.0; j <= 1.0; j += 1.0){
          color.rgb += texture2D(shadowcolor1, coord + vec2(i, j) * pixel).rgb;
          color.a += texture(shadowtex0, coord + vec2(i, j) * pixel).x;
        }
      }

      color /= 9.0;
      color.rgb = mat3(shadowModelView) * (color.rgb * 2.0 - 1.0) * 0.5 + 0.5;

      /*
      color.rgb = texture2D(shadowcolor0, coord).rgb * 2.0 - 1.0;
      color.rgb = mat3(shadowModelView) * color.rgb * 0.5 + 0.5;
      color.a = texture(shadowtex0, coord).x;
      */
    }


  #endif
  gl_FragData[0] = color;
}

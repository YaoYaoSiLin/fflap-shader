#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;

uniform sampler2D depthtex0;

uniform sampler2D depthtex2;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 shadowLightPosition;
uniform vec3 upPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int isEyeInWater;
uniform int frameCounter;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 SecondaryIndirect(){
  vec3 indirect = vec3(0.0);

  vec2 blueNoise = vec2(GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jittering * 1.0),
                        GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y * 0.5, jittering * 1.0));

  vec3 normal = normalDecode(texture2D(composite, texcoord).xy);
  vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
  float viewLength = length(viewPosition);

  float radius = 1000.0 / 3.0 / viewLength;

  for(int j = 0; j < 3; j++){
    for(int i = 0; i < 4; i++){
      float angle = (i + 1) * 0.25 * 2.0 * Pi;
      vec2 direction = RotateDirection(vec2(cos(angle), sin(angle)), blueNoise) * radius;

      vec2 uv = texcoord - pixel + direction * pixel;
      if(floor(texcoord) != vec2(0.0)) break;
      vec3 samplePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(uv, texture(depthtex0, uv).x) * 2.0 - 1.0));

      vec3 lightPosition = normalize(samplePosition - viewPosition);
      float pathLength = length(samplePosition - viewPosition);

      vec3 sampleColor = decodeGamma(texture2D(gdepth, min(0.5 - pixel, uv * 0.5)).rgb);
      vec3 sampleNormal = normalDecode(texture2D(composite, uv).xy);

      float ndotl = saturate(dot(lightPosition, normal));
      float irrdiance = saturate(dot(-lightPosition, sampleNormal));

      float l = min(3.0, 1.0 / pow4(pathLength));

      indirect += sampleColor * ndotl * irrdiance * l;
    }
  }

  return indirect / 12.0;
}

void main(){
  vec3 indirect = decodeGamma(texture2D(gdepth, min(texcoord * 0.5, 0.5 - pixel)).rgb);
       indirect += SecondaryIndirect();

  indirect = encodeGamma(indirect);

  /* DRAWBUFFERS:1 */
  gl_FragData[0] = vec4(indirect, texture(depthtex0, texcoord).x);
}

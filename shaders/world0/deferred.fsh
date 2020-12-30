#version 130

uniform sampler2D gnormal;
uniform sampler2D composite;

uniform sampler2D depthtex0;

uniform sampler2D depthtex2;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

float ComputeAO(in vec3 p, in vec3 n, in vec3 s, in float r){
  vec3 v = s - p;
  float vdotv = dot(v, v);
  float vdotn = dot(-normalize(v), n) * inversesqrt(vdotv);

  float falloff = 1.0 + 1.0 / sqrt(r * r - vdotv);

  return saturate(max(0.0, vdotn - 1e-5) * falloff);
}

float AccumulateAO(in vec2 rayDirection, in vec2 texcoord){
  return 0.0;
}

#define AO_Direction_Steps 3 //[0 2 3 4 8]

float ComputeCoarseAO(in vec2 coord){

  if(texture(depthtex0, coord).x > 0.9999) return 1.0;

  float bias = 0.05;

  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex0, coord).x) * 2.0 - 1.0));

  vec3 normal = normalDecode(texture2D(composite, coord).xy);
  vec3 normalFlat = normalDecode(texture2D(gnormal, coord).xy);
  //normal = normalFlat;
  //if(dot(normalize(-vP), normal) < 0.0) normal *= -1.0;

  float LargeScaleAO = 0.0;
  float SmallScaleAO = 0.0;

  //float dither = R2sq(coord * resolution * 0.5 - jittering) * 0.8 + 0.2;
  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y * 0.5, jittering);
  float dither2 = GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y * 0.5, jittering);

  vP -= normal * pixel.x * 4.0 * vP.z;

  int stepSample = 3;
  int dirSample = int(AO_Direction_Steps);

  float RadiusPixels = 4096.0 * inversesqrt(vP.z * vP.z);
  float StepSizePixels = (RadiusPixels / 12.0) / float(stepSample);
  float rayPixel = (1e-5 + StepSizePixels);

  float invdirSample = 1.0 / float(dirSample);
  float alpha = invdirSample * 2.0 * Pi;

  float falloffcoe = 14.0 / 12.0;

  int count = 0;

  for(int stepIndex = 1; stepIndex <= stepSample; stepIndex++){
    #if AO_Direction_Steps > 1
    for(int dirIndex = 1; dirIndex <= dirSample; dirIndex++){
      float angle = alpha * float(dirIndex);
      vec2 rayDirection = vec2(cos(angle), sin(angle));
           rayDirection = RotateDirection(rayDirection, vec2(dither, dither2));

      vec2 SnappedUV = (rayPixel * rayDirection) * pixel + texcoord;
      if(floor(SnappedUV) != vec2(0.0)) {break;}

      vec3 v = nvec3(gbufferProjectionInverse * nvec4(vec3(SnappedUV, texture(depthtex0, SnappedUV).x) * 2.0 - 1.0));

      float r = falloffcoe * (float(stepIndex) * 0.5 + 0.5);
      LargeScaleAO += ComputeAO(v, normal, vP, r);
      count++;
    }
    #else
      float angle = dither * 2.0 * Pi;
      vec2 rayDirection = vec2(cos(angle), sin(angle));
      //     rayDirection = RotateDirection(rayDirection, vec2(dither, dither2));

      vec2 SnappedUV = (rayPixel * rayDirection) * pixel + texcoord;
      if(floor(SnappedUV) != vec2(0.0)) {break;}

      vec3 v = nvec3(gbufferProjectionInverse * nvec4(vec3(SnappedUV, texture(depthtex0, SnappedUV).x) * 2.0 - 1.0));
      if(dot(normalize(-v), normal) < 0.0) continue;

      float r = falloffcoe * (float(stepIndex) * 0.5 + 0.5);
      LargeScaleAO += ComputeAO(v, normal, vP, r);
      count++;
    #endif

    rayPixel += StepSizePixels;
  }

  float ao = LargeScaleAO / float(count);

  return saturate(1.0 - ao * 2.0);
}

void main() {
  vec2 uvjittering = texcoord + jittering * pixel;
  vec4 data = vec4(ComputeCoarseAO(uvjittering), vec2(0.0), 1.0);

  //if(texture(depthtex0, texcoord).x > 0.7) data = vec4(0.0);
  if(texture(depthtex0, texcoord).x < 0.7) data = vec4(0.0);

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = data;
}

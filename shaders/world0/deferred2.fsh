#version 130

#define GI_Rendering_Scale 0.5

#define composite colortex3
#define gaux2 colortex5

uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;

in vec2 texcoord;

/* DRAWBUFFERS:5 */

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

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;


// RotationCosSin is (cos(alpha),sin(alpha)) where alpha is the rotation angle
// A 2D rotation matrix is applied (see https://en.wikipedia.org/wiki/Rotation_matrix)
vec2 RotateDirection(vec2 V, vec2 RotationCosSin) {
    return vec2(V.x*RotationCosSin.x - V.y*RotationCosSin.y,
                V.x*RotationCosSin.y + V.y*RotationCosSin.x);
}

float ComputeCoarseAO(in vec2 coord){
  //return 1.0;
  if(floor(coord) != vec2(0.0)) return 1.0; else {
  float bias = 0.1;

  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex0, coord).x) * 2.0 - 1.0));
  vec3 normal = normalDecode(texture2D(composite, coord).xy);

  float ao = 0.0;

  int steps = 8;
  float invsteps = 1.0 / float(steps);

  float alpha = invsteps * 2.0 * Pi;

  float dither = R2sq(coord * resolution * 0.5 - jittering) * 0.8 + 0.2;

  float RadiusPixels = 4096.0 / vP.z;
  float StepSizePixels = (RadiusPixels / 4.0) / float(steps + 1);

  float NegInvR2 = -1.0 / (2.0 * 2.0);

  vP -= normal * pixel.x * 3.0 * vP.z;

  for(int i = 0; i < steps; ++i){
    float angle = alpha * float(i);
    vec2 direction = RotateDirection(vec2(cos(angle), sin(angle)), vec2(dither, 1.0 - dither));

    float rayPixel = (0.0001 + StepSizePixels);

    vec2 SnappedUV = round(rayPixel * direction) * pixel + coord;
    rayPixel += StepSizePixels;

    vec3 v = nvec3(gbufferProjectionInverse * nvec4(vec3(SnappedUV, texture(depthtex0, SnappedUV).x) * 2.0 - 1.0));

    v = (v - vP);
    float vdotv = dot(v, v);
    v = normalize(v);
    float ndotv = dot(normal, v) * inversesqrt(vdotv);

    float falloff = clamp01(vdotv * NegInvR2 + 1.0);

    ao += clamp01(ndotv - bias) * falloff;
  }

  return 1.0 - ao * invsteps;
  }

}

void main() {
  vec4 data = vec4(vec3(0.0), 1.0);

  vec2 fragCoord = texcoord * resolution;

  //if(texcoord.x < 0.5 + pixel.x && texcoord.y < 0.5 + pixel.y){
  //}
  data = texture2D(gaux2, texcoord);

  //vec2 halfCoord = texcoord * 2.0;
  //vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(half, texture(depthtex0, half).x) * 2.0 - 1.0));
  //vec3 normal = normalDecode(texture2D(composite, halfCoord).xy);

  if(texcoord.x > 0.5){
    if(texcoord.y > 0.5) data.x = (texture(depthtex0, (texcoord - vec2(0.5)) * 2.82843).x);
    else data.xyz = normalDecode(texture2D(composite, (texcoord - vec2(0.5, 0.0)) * 2.82843).xy) * 0.5 + 0.5;
  }else{
    if(texcoord.y > 0.5){
      data.x = ComputeCoarseAO((texcoord - vec2(0.0, 0.5)) * 2.0);
      data.y = texture(depthtex0, (texcoord - vec2(0.0, 0.5)) * 2.0).x;
    }
  }


  gl_FragData[0] = data;
}

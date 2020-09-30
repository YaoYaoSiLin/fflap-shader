#version 130

#define GI_Rendering_Scale 0.5 //[0.353553 0.5]

#define gcolor colortex0
#define composite colortex3
#define gaux2 colortex5

uniform sampler2D gcolor;
uniform sampler2D composite;
uniform sampler2D gaux2;

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

  //float dither = R2sq(coord * resolution * 0.5 - jittering) * 0.8 + 0.2;
  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y, jittering);

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

void SecondaryIndirect(inout vec4 tex){
  tex.rgb = vec3(0.0);

  float blueNoise = GetBlueNoise(depthtex2, texcoord, resolution.y, jittering);

  vec2 coord = texcoord / GI_Rendering_Scale;
  vec3 rayOrigin = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex0, coord).x) * 2.0 - 1.0));

  vec3 normal = normalDecode(texture2D(composite, coord).xy);
  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

  vec3 upVector = abs(worldNormal.z) < 0.4999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  upVector = mat3(gbufferModelView) * upVector;

  vec3 t = normalize(cross(upVector, normal));
  vec3 b = cross(normal, t);
  mat3 tbn = mat3(t, b, normal);

  float CosTheta = sqrt((1 - blueNoise) / ( 1 + (0.7 - 1) * blueNoise));
  float SinTheta = sqrt(1 - CosTheta * CosTheta);

  float r = abs(blueNoise - 0.5) * 2.0 * 2.0 * Pi;
  vec3 ramdomDirection = vec3(cos(r) * SinTheta, sin(r) * SinTheta, 1.0);
       ramdomDirection = normalize(tbn * ramdomDirection);
  vec3 rayDirection = normalize(reflect(normalize(rayOrigin), ramdomDirection));

  int steps = 6;
  float invsteps = 1.0 / float(steps);

  float radius = 1.4;
  rayDirection *= radius;

  vec3 rayStart = rayOrigin + rayDirection;

  for(int i = 0; i < steps; i++){
    vec3 testPoint = rayStart - rayDirection * invsteps;

    vec3 Coord = nvec3(gbufferProjection * nvec4(testPoint)) * 0.5 + 0.5;
    if(clamp(Coord.xy, pixel * 2.0, 1.0 - pixel * 2.0) != Coord.xy) break;

    float sampleDepth = texture(depthtex0, Coord.xy).x;
    vec3 samplePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(Coord.xy, texture(depthtex0, Coord.xy).x) * 2.0 - 1.0));
    vec3 halfPosition = rayOrigin - samplePosition;

    if(Coord.z < sampleDepth) continue;
    //if(length(samplePosition - testPoint) < 1.0 - float(i) * invsteps) continue;

    vec3 sampleColor = texture2D(gaux2, Coord.xy * GI_Rendering_Scale).rgb * texture2D(gcolor, Coord.xy).rgb;
    vec3 sampleNormal = normalDecode(texture2D(composite, Coord.xy).xy);

    vec3 lightingDirection = normalize(testPoint - rayOrigin);
    float sampleNdotl = step(0.01, dot(lightingDirection, normal)) * step(0.01, dot(-lightingDirection, sampleNormal));
    //float sampleNdotl = saturate(dot(lightingDirection, normal) * 1.0) * saturate(dot(-lightingDirection, sampleNormal) * 1.0);

    float sampleFading = pow4(length(halfPosition));

    tex.rgb += sampleColor * sampleNdotl / max(1.0, sampleFading);
  }

  tex.rgb += texture2D(gaux2, texcoord).rgb;
}

void main() {
  vec4 data = vec4(vec3(0.0), 1.0);

  vec2 fragCoord = texcoord * resolution;

  //if(texcoord.x < 0.5 + pixel.x && texcoord.y < 0.5 + pixel.y){
  //}
  //data = texture2D(gaux2, texcoord);
  if(floor(texcoord.xy * 2.0) == vec2(0.0)){
    data = vec4(1.0);
    data = texture2D(gaux2, texcoord);
    //SecondaryIndirect(data);
  }

  //vec2 halfCoord = texcoord * 2.0;
  //vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(half, texture(depthtex0, half).x) * 2.0 - 1.0));
  //vec3 normal = normalDecode(texture2D(composite, halfCoord).xy);

  if(texcoord.x > 0.5){
    if(texcoord.y > 0.5) data.x = (texture(depthtex0, (texcoord - vec2(0.5)) / GI_Rendering_Scale).x);
    else data.xyz = normalDecode(texture2D(composite, (texcoord - vec2(0.5, 0.0)) / GI_Rendering_Scale).xy) * 0.5 + 0.5;
  }else{
    if(texcoord.y > 0.5){
      data.x = ComputeCoarseAO((texcoord - vec2(0.0, 0.5)) * 2.0);
      data.y = texture(depthtex0, (texcoord - vec2(0.0, 0.5)) * 2.0).x;
    }
  }


  gl_FragData[0] = data;
}

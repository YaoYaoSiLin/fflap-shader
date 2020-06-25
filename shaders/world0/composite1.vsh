#version 130

uniform float viewWidth;
uniform float viewHeight;
uniform int frameCounter;

uniform vec3 sunPosition;

uniform mat4 gbufferModelViewInverse;

out float night;
out float remap;

out vec2 texcoord;
out vec2 jitter;

out vec3 lightVector;
out vec3 worldLightVector;
out vec3 nworldLightVector;

#include "../libs/jittering.glsl"

void main() {
  vec2 resolution = vec2(viewWidth, viewHeight);
  vec2 pixel = 1.0 / resolution;

  texcoord = gl_MultiTexCoord0.st;
  jitter = jittering;

  float m = 5892.0;
  float shattered_savanna = 240.0;
  float remap = m / shattered_savanna;

  vec3 worldLightPosition = mat3(gbufferModelViewInverse) * normalize(sunPosition);
  night = step(worldLightPosition.y, -0.1);

  lightVector = sunPosition;
  if(bool(night)) lightVector = -lightVector;

  worldLightVector = mat3(gbufferModelViewInverse) * lightVector;
  lightVector = normalize(lightVector);
  nworldLightVector = mat3(gbufferModelViewInverse) * lightVector;

  gl_Position = ftransform();
}

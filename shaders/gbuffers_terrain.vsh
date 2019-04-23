#version 130

attribute vec4 at_tangent;
attribute vec4 mc_Entity;
attribute vec4 mc_midTexCoord;

uniform sampler2D noisetex;

uniform float frameTimeCounter;

uniform vec3 cameraPosition;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

//out float cutoutBlock;

out float id;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 tangent;
out vec3 binormal;

out vec3 vP;

out vec4 color;

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  color = gl_Color;

  vec4 position = gl_Vertex;
  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;
  vec3 wP = mat3(gbufferModelViewInverse) * vP + cameraPosition;
       //position = gbufferModelViewInverse * position;
       //position.xyz += cameraPosition;
  id = 0.0;
  //cutoutBlock = 0.0;

  bool leaves = mc_Entity.x == 18;

  bool plant = mc_Entity.x == 31;

  bool double_plant_upper = mc_Entity.x == 175;
  bool double_plant_lower = mc_Entity.x == 176;
  bool double_plant = double_plant_upper || double_plant_lower;

  //bool farm = mc_Entity.x == 59 || mc_Entity.x == 141 || mc_Entity.x == 142 || mc_Entity.x == 207;

  if(double_plant || plant) {
    vec2 noise = texture2D(noisetex, (position.xz) / 64.0).xz * 80.0;
    float time = frameTimeCounter * 0.74;
    float time2 = sin(frameTimeCounter * 0.48) * 1.21;
    vec2 noise2 = vec2(sin(time2 + noise.x * 1.3), sin(time2 + noise.y * 1.3));

    vec2 wave = vec2(sin((time + noise.x) * 3.14159 * 0.5 + noise2.y + time2), sin((time + noise.y) * 3.14159 * 0.5 + noise2.x + time2)) * 0.016;

    if(double_plant_lower || plant) {
      wave *= float(mc_midTexCoord.y > gl_MultiTexCoord0.y);
    }else{
      wave *= 1.0 + (mc_midTexCoord.y - gl_MultiTexCoord0.y);
    }

    if(double_plant) wave *= 0.632;

    position.xz += wave * lmcoord.y;
    id = 31.0;
    //cutoutBlock = 1.0;
  }

  if(leaves) {
    float time2 = sin(frameTimeCounter * 1.07);
    float time = frameTimeCounter * 1.14;

    vec3 noise  = texture2D(noisetex, (position.xz + vec2(time2, time * 0.45) * 0.019) / 8.0).xyz * 4.0;
         noise += texture2D(noisetex, (position.xz - vec2(time * 0.45, time2) * 0.019) / 4.0).xyz * 2.0;
         noise *= 0.74;

    //vec3 noise2 = vec3(sin(noise.x * time2), sin(noise.y * time2), sin(noise.z * time2)) * 0.1;

    vec3 wave = vec3(sin(time + noise.x), sin(time + noise.y), sin(time + noise.z));

    position.xyz += wave * 0.021 * lmcoord.y;
    id = 18.0;
    //cutoutBlock = 1.0;
  }

  //position.xyz -= cameraPosition;
  //position = gbufferModelView * position;

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;
  #ifdef Enabled_TAA
  gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif

  normal  = normalize(gl_NormalMatrix * gl_Normal);
  tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
  binormal = cross(tangent, normal);

}

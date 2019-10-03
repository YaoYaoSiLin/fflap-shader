#version 130

/*
*/

//#define Continuum2_Texture_Format

uniform sampler2D gaux1;

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform mat4 gbufferProjectionInverse;

uniform float viewWidth;
uniform float viewHeight;

uniform vec3 upPosition;

in float id;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 vP;

in vec4 color;

#include "libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;

  if(albedo.a < 0.05 || length(vP) > max(32.0, far)) discard; //disable sky texture & low alpha pixel
  albedo.a = 1.0;

  vec2 screenCoord = gl_FragCoord.xy / vec2(viewWidth, viewHeight);

  //float depth = texture2D(gaux1, screenCoord).a;
  //vec3 particlesPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(screenCoord, depth) * 2.0 - 1.0));
  //float particleDistance = length(particlesPosition);
  //if(length(vP) > particleDistance && depth > 0.0) discard;

  vec4 speculars = texture2D(specular, texcoord);
       speculars.a = step(0.001, speculars.a + albedo.a);

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  if(speculars.r == speculars.b) speculars.rgb = vec3(speculars.r, 0.0, 0.0);


  speculars.b *= 0.06;
  speculars.r = clamp(speculars.r, 0.00001, 0.009);
  speculars.a = 1.0;

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

/* DRAWBUFFERS:01234 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, 253.0 / 255.0, 1.0);
  gl_FragData[2] = vec4(vec2(0.0), 1.0, 0.0);
  gl_FragData[3] = speculars;
  gl_FragData[4] = vec4(0.0, 0.0, 0.0, gl_FragCoord.z);
}

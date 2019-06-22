#version 130

/*
*/

#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform vec3 upPosition;

uniform mat4 gbufferModelView;

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
       albedo.a = step(0.001, albedo.a);

  if(albedo.a < 0.005 || length(vP) > max(32.0, far * 0.5)) discard; //disable sky texture & low alpha pixel

  vec4 speculars = texture2D(specular, texcoord);
       speculars.a = step(0.001, speculars.a + albedo.a);

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  speculars.b = 0.0;
  speculars.r = clamp(speculars.r, 0.00001, 0.009);
  speculars.a = 1.0;

  #if MC_VERSION > 11202
  speculars = vec4(0.001, 0.0, 0.0, 1.0);
  #endif

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, 253.0 / 255.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalize(vP.xyz * vec3(1.0, 0.0, -1.0))), 1.0, 1.0);
  gl_FragData[3] = speculars;
}

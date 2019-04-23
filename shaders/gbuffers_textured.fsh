#version 120

#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform vec3 upPosition;

uniform mat4 gbufferProjection;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 normal;
varying vec3 vP;

varying vec4 color;

#include "libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  float depth = nvec3(gbufferProjection * nvec4(vP)).z * 0.5 + 0.5;
  if(depth > 0.999 || depth < 0.05) discard;

  vec4 albedo = texture2D(texture, texcoord) * color;
  if(albedo.a < 0.001) discard;

  //if(depth < 0.999) albedo *= color;

  //albedo.a = step(0.2, albedo.a);
  //albedo.a = 1.0;

  //if(albedo.a < 0.7) discard;

  //if(albedo.a > 0.999){
    //albedo.rgb = pow(albedo.rgb, vec3(2.2));
    //albedo.rgb *= pow(max(lmcoord.x, lmcoord.y), 2.0) * 0.25;
    //albedo.rgb = pow(albedo.rgb, vec3(1.0 / 2.2));

    //albedo.rgb = pow(albedo.rgb, vec3(0.5));
  //}

  vec4 speculars = texture2D(specular, texcoord);
       speculars.a = 1.0;

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  //if(albedo.a * 10.0 < 0.1) discard;

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, 0.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalize(upPosition)).xy, 1.0, 1.0);
  gl_FragData[3] = speculars;
}

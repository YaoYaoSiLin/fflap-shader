#version 130

//#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D specular;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;

in vec4 color;

#include "/libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord);
       albedo *= color;

  if(albedo.a < 0.05) discard;
  //albedo.rgb = vec3(0.5, 0.5, 1.0);

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  speculars.r = clamp(speculars.r, 0.001, 0.999);
  speculars.b = 0.12;

  float selfShadow = 1.0;
  float emissive = 0.06;
  vec4 lightmap = vec4(pack2x8(lmcoord), selfShadow, emissive, 1.0);

/* DRAWBUFFERS:01 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = lightmap;
}

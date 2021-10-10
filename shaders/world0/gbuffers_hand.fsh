#version 130

//#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform vec3 shadowLightPosition;

uniform mat4 gbufferProjection;

uniform float wetness;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;
in vec3 vP;

in vec4 color;

#include "/libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;

  vec3 nvP = normalize(vP);
  bool backFace = dot(normal, nvP) > 0.0;

  if(albedo.a < 0.05) discard;
  albedo.a = 1.0;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  if(speculars.r == speculars.b) speculars.rgb = vec3(speculars.r, 0.0, 0.0);
  //speculars.a *= 0.06;

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  speculars.r = clamp(speculars.r, 0.001, 0.999);
  speculars.b = speculars.b > 64.5 / 255.0 ? 0.0 : speculars.b;
  //speculars.a = 1.0;


  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  vec3 flatNormal = normal;

  vec3 texturedNormal = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
       texturedNormal = normalize(tbnMatrix * texturedNormal);

  if(backFace){
    flatNormal = -flatNormal;
    texturedNormal = -texturedNormal;
  }

  float selfShadow = 1.0;//step(0.1, dot(normalize(shadowLightPosition), flatN));
  float emissive = speculars.a * step(speculars.a, 0.999) * (255.0 / 254.0);

  vec4 lightmap = vec4(pack2x8(lmcoord), selfShadow, pack2x8(vec2(emissive, 1.0)), 1.0);

  vec2 encodeNormalFlat = normalEncode(flatNormal);
  vec2 encodeNormalTextured = normalEncode(texturedNormal);

  float mask = 248.0 / 255.0;

  float specularPackge = pack2x8(speculars.rg);

  /* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = lightmap;
  gl_FragData[2] = vec4(encodeNormalFlat, mask, 1.0);
  gl_FragData[3] = vec4(encodeNormalTextured, specularPackge, speculars.b);
}

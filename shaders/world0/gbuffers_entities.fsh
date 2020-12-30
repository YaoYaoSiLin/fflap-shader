#version 130

//#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform vec3 shadowLightPosition;

uniform vec4 entityColor;
uniform int entityId;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec3 vP;

in vec4 color;

#include "../libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);

  if(albedo.a < 0.2) discard;
  albedo.a = 1.0;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  if(speculars.r == speculars.b) speculars.rgb = vec3(speculars.r, 0.0, 0.0);

  //speculars.b *= 0.12;
  speculars.r = clamp(speculars.r, 0.0001, 0.9999);
  speculars.a = 1.0;

  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  vec3 flatNormal = normal;

  vec3 texturedNormal = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
       texturedNormal = normalize(tbnMatrix * texturedNormal);

  if(!gl_FrontFacing) {
    texturedNormal = -texturedNormal;
    flatNormal = -flatNormal;
  }

  float mask = 249.0;

  //if(entityId == 77 || entityId == 9) {
  //  mask = 250.0;
  //  selfShadow = 1.0;
  //}

  //float normalGlitchE = CalculateMaskID(77.0, float(entityId)) + CalculateMaskID(9.0, float(entityId));
  //float particelsID = 250.0;
  //mask += (particelsID - mask) * normalGlitchE;
  //selfShadow = (1.0 - selfShadow) * normalGlitchE;

  mask /= 255.0;
  //discard;

  float emissive = speculars.b;
  vec4 lightmap = vec4(pack2x8(lmcoord), 1.0, emissive, 1.0);

  float specularPackge = pack2x8(speculars.rg);

  vec2 encodeNormal0 = normalEncode(flatNormal);
  vec2 encodeNormal1 = normalEncode(texturedNormal);

  /* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = lightmap;
  gl_FragData[2] = vec4(encodeNormal0, mask, 1.0);
  gl_FragData[3] = vec4(encodeNormal1, specularPackge, 1.0);
}

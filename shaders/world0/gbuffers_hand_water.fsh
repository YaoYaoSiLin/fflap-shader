#version 130

//#define Continuum2_Texture_Format

uniform sampler2D gaux1;

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform mat4 gbufferProjectionInverse;

uniform float viewWidth;
uniform float viewHeight;

uniform int entityId;
uniform int blockEntityId;

uniform vec3 upPosition;

in float portal;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 vP;

in vec3 normal;
in vec3 binormal;
in vec3 tangent;

in vec4 color;

#include "../libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  if(albedo.a < 0.05) discard;
  albedo.a = 1.0;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  speculars.r = clamp(speculars.r, 0.01, 0.99);
  speculars.g = max(0.02, speculars.g);
  speculars.b *= 0.06;

  float mask = 252.0 / 255.0;

  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  vec3 flatNormal = normal;

  vec3 texturedNormal = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
       texturedNormal = normalize(tbnMatrix * texturedNormal);

  vec3 visibleNormal = texturedNormal;
  if(bool(step(dot(texturedNormal, normalize(-vP)), 0.2))) visibleNormal = flatNormal;

  if(!gl_FrontFacing){
    texturedNormal = -texturedNormal;
    flatNormal = -flatNormal;
    visibleNormal = -visibleNormal;
  }

  vec2 encodeNormal = normalEncode(visibleNormal);

  float lightmapPackge = pack2x8(lmcoord);
  float emissive = max(floor(lmcoord.x * 15.0 - 13.0), speculars.b) * 0.06;
  vec4 lightmap = vec4(lightmapPackge, 1.0, emissive, 1.0);

  float specularPackge = pack2x8(speculars.rg);

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = lightmap;
  gl_FragData[2] = vec4(albedo.a, 0.0, mask, 1.0);
  gl_FragData[3] = vec4(encodeNormal, specularPackge, 1.0);
  //gl_FragData[4] = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
}

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

in vec4 color;

#include "../libs/common.inc"

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

vec3 CalculateParticleNormal(){
  vec3 facingToPlayer = nvec3(gbufferProjectionInverse * nvec4(vec3(0.5, 0.5, 0.7) * 2.0 - 1.0));
  return normalize(-facingToPlayer);
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  if(albedo.a < 0.05) discard;
  albedo.a = 1.0;

  vec2 screenCoord = gl_FragCoord.xy / vec2(viewWidth, viewHeight);

  float depth = texture2D(gaux1, screenCoord).a;
  vec3 particlesPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(screenCoord, depth) * 2.0 - 1.0));
  float particleDistance = length(particlesPosition);
  //if(length(vP) > particleDistance && depth > 0.0) discard;

  //albedo.rgb = texture2D(gaux2, screenCoord).rgb;

  vec4 speculars = texture2D(specular, texcoord);
       speculars.r = clamp(speculars.r, 0.01, 0.99);
       speculars.g = max(0.02, speculars.g);

  //#ifdef Continuum2_Texture_Format
  //  speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  //#endif

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  //speculars.r = clamp(speculars.r, 0.001, 0.999);
  //speculars.r = 0.0;
  //speculars.b = 1.0;                                    //world boder and marker is emissive
  //speculars.a = 1.0;

  float mask = 252.0 / 255.0;

  vec2 encodeNormal = normalEncode(normal);

  //if(entityId == 10) discard;

  float lightmapPackge = pack2x8(lmcoord);
  float emissive = max(floor(lmcoord.x * 15.0 - 13.0), speculars.b) * 0.06;
  vec4 lightmap = vec4(lightmapPackge, 1.0, emissive, 1.0);

  float specularPackge = pack2x8(speculars.rg);

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = lightmap;
  gl_FragData[2] = vec4(encodeNormal, mask, 1.0);
  gl_FragData[3] = vec4(encodeNormal, specularPackge, 1.0);
  //gl_FragData[4] = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0);
}

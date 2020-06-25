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
  speculars.b *= 0.06;

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  speculars.r = clamp(speculars.r, 0.001, 0.999);
  //speculars.b *= 0.12;
  speculars.a = 1.0;

  vec3 normalBase = normal;

  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  vec3 normalSurface = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
       normalSurface = normalize(tbnMatrix * normalSurface);

  if(backFace){
    normalBase = -normalBase;
    normalSurface = -normalSurface;
  }

  vec3 normalVisible = normalSurface;
  if(-0.15 < dot(nvP, normalSurface)) normalVisible = normalBase;

  normalSurface.xy = normalEncode(normalSurface);
  normalVisible.xy = normalEncode(normal);

  float selfShadow = step(0.1, dot(normalize(shadowLightPosition), normalBase));

  float mask = 248.0 / 255.0;

  /* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, mask, speculars.b);
  gl_FragData[2] = vec4(normalVisible.xy, speculars.r, selfShadow);
  gl_FragData[3] = vec4(normalSurface.xy, speculars.g, 1.0);
}

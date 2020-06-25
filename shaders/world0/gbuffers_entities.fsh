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

#define CalculateMaskID(id, x) step(id - 0.5, x) * step(x, id + 0.5)

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);

  vec3 nvP = normalize(vP);
  bool backFace = dot(normal, nvP) > 0.0;

  if(albedo.a < 0.3) discard;
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

  vec3 normalBase = normal;

  vec3 normalSurface = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
       normalSurface = normalize(tbnMatrix * normalSurface);

  if(backFace) {
    normalSurface = -normalSurface;
    normalBase = -normalBase;
  }

  vec3 normalVisible = normalSurface - (normalSurface - normalBase) * step(-0.15, dot(nvP, normalSurface));

  float selfShadow = step(0.1, dot(normalize(shadowLightPosition), normalBase));

  normalSurface.xy = normalEncode(normalSurface);
  normalVisible.xy = normalEncode(normalBase);

  float mask = 249.0;

  //if(entityId == 77 || entityId == 9) {
  //  mask = 250.0;
  //  selfShadow = 1.0;
  //}

  float normalGlitchE = CalculateMaskID(77.0, float(entityId)) + CalculateMaskID(9.0, float(entityId));
  float particelsID = 250.0;
  mask += (particelsID - mask) * normalGlitchE;
  selfShadow = (1.0 - selfShadow) * normalGlitchE;

  mask /= 255.0;
  //discard;

  /* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, mask, speculars.b);
  gl_FragData[2] = vec4(normalVisible.xy, speculars.r, selfShadow);
  gl_FragData[3] = vec4(normalSurface.xy, speculars.g, 1.0);
}

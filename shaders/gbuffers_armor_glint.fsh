#version 120

uniform sampler2D texture;
uniform sampler2D specular;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;

in vec4 color;

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
  vec4 albedo = texture2D(texture, texcoord);
       albedo *= color;

  if(albedo.a < 0.05) discard;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  speculars.r = clamp(speculars.r, 0.001, 0.999);
  speculars.b = 0.12;

/* DRAWBUFFERS:03 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = speculars;
}

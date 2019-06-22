#version 130

#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform vec3 upPosition;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 vP;

in vec4 color;

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  if(albedo.a < 0.001) discard;
  albedo.a = 1.0;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
    speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  #if MC_VERSION > 11202
  speculars = vec4(0.001, 0.0, 0.0, 1.0);
  #endif

  speculars.a = 1.0;
  speculars.r = clamp(speculars.r, 0.00001, 0.999);
  speculars.b = 1.0;                                    //world boder and marker is emissive

/* DRAWBUFFERS:456 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = speculars;                           //PBR data form texture_s.png
  gl_FragData[2] = vec4(lmcoord, gl_FragCoord.z, 1.0);  //
}

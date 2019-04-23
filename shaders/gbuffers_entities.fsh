#version 120

#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D noisetex;

uniform vec4 entityColor;

uniform float wetness;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 normal;
varying vec3 tangent;
varying vec3 binormal;

varying vec4 color;

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  if(albedo.a < 0.001) discard;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, 0.0, speculars.a);
  #endif

  speculars.a = 1.0;

  albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);

  vec3 normalTexture = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
  mat3 tbnMatrix = mat3(tangent, binormal, normal);
  normalTexture = normalize(tbnMatrix * normalTexture);
  normalTexture.xy = normalEncode(normalTexture);

  //speculars.rgb = mix(speculars.rgb, vec3(0.0), entityColor.a);

  //albedo.rgb *= lmcoord.y;

  /* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, 0.0, 1.0);
  gl_FragData[2] = vec4(normalTexture.xy, 1.0, 1.0);
  gl_FragData[3] = speculars;
}

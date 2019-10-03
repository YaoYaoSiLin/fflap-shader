#version 130

//#define Continuum2_Texture_Format

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D noisetex;

uniform vec4 entityColor;

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
  if(albedo.a < 0.05) discard;
  //albedo.a = 1.0;
  //if(dot(normal, normalize(vP)) >= 0.0) discard;

  vec4 speculars = texture2D(specular, texcoord);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  if(speculars.r == speculars.b) speculars.rgb = vec3(speculars.r, 0.0, 0.0);

  speculars.b *= 0.12;
  speculars.r = clamp(speculars.r, 0.0001, 0.999);
  speculars.a = 1.0;

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  albedo.rgb = mix(albedo.rgb, entityColor.rgb, entityColor.a);

  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  vec3 surfaceNormal = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
       surfaceNormal = normalize(tbnMatrix * surfaceNormal);
  if(-0.15 < dot(normalize(vP), surfaceNormal)) surfaceNormal = normal;
  if(dot(normal, normalize(vP)) >= 0.0) surfaceNormal = -surfaceNormal;
  surfaceNormal.xy = normalEncode(surfaceNormal);

  //speculars.rgb = mix(speculars.rgb, vec3(0.0), entityColor.a);


  //albedo.rgb *= lmcoord.y;

  /* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, 252.0 / 255.0, 1.0);
  gl_FragData[2] = vec4(surfaceNormal.xy, 1.0, 1.0);
  gl_FragData[3] = speculars;
}

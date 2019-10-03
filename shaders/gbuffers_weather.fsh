#version 130

uniform sampler2D texture;

uniform mat4 gbufferModelViewInverse;

uniform vec3 cameraPosition;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 vP;
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
  vec4 albedo = texture2D(texture, texcoord) * color;

  vec3 wP = mat3(gbufferModelViewInverse) * vP + cameraPosition;

  if(albedo.a < 0.05 || wP.y > 128.5) discard;
  //albedo.a = 1.0;

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = vec4(albedo.rgb, 1.0);
  gl_FragData[1] = vec4(lmcoord.xy, 0.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalize(normal)), albedo.a, 0.0);
  gl_FragData[3] = vec4(0.0, 0.0, 0.0, 1.0);
}

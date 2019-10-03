#version 130

uniform sampler2D texture;

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
  vec4 albedo = texture2D(texture, texcoord);
       albedo *= color;

  albedo.rgb /= 1.0 + max(albedo.r, max(albedo.g, albedo.b)) * 3.0;
  albedo.rgb *= 4.0;

  if(albedo.a < 0.05) discard;
  albedo.a = 1.0;

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(0.0, 0.0, 0.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normal), 1.0, 1.0);
  gl_FragData[3] = vec4(1.0, 0.0, 1.0, 1.0);
}

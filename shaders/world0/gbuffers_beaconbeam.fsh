#version 130

uniform sampler2D texture;

in vec2 lmcoord;
in vec2 texcoord;

in vec4 color;

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

/* DRAWBUFFERS:0123 */

void main() {
  float mask = 251.0 / 255.0;

    vec4 albedo = texture2D(texture, texcoord) * color;
    if(albedo.a < 0.2) discard;
    albedo.a = 1.0;

  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(0.0, 0.0, 1.0, 0.9);
  gl_FragData[2] = vec4(vec2(0.0), mask, 1.0);
  gl_FragData[3] = vec4(vec2(0.0), 0.0, 1.0);
}

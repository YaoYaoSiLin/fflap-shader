#version 130

uniform sampler2D texture;
uniform sampler2D gcolor;

in vec2 texcoord;

in vec4 color;

void main() {
  vec4 tex = texture2D(texture, texcoord) * color;

/* DRAWBUFFERS:0 */
  gl_FragData[0] = tex;
}

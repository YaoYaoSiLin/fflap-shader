#version 130

uniform sampler2D texture;

in vec2 texcoord;

in vec4 color;

void main() {
  vec4 tex = texture2D(texture, texcoord) * color;

  if(tex.a < 0.05) discard;

/* DRAWBUFFERS:0 */
  gl_FragData[0] = tex;
}

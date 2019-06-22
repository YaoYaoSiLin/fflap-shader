#version 120

uniform sampler2D texture;

varying vec2 texcoord;

varying vec4 color;

void main() {
  vec4 tex = texture2D(texture, texcoord) * color;

  if(tex.a < 0.004) discard;

/* DRAWBUFFERS:0 */
  gl_FragData[0] = tex;
}

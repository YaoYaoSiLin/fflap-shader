#version 120

varying vec4 color;

void main() {
/* DRAWBUFFERS:02 */
  gl_FragData[0] = vec4(color.rgb, 1.0);
  gl_FragData[1] = vec4(0.0, 0.0, 1.0, 1.0);
}

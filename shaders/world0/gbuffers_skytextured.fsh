#version 120

varying vec4 color;

void main() {
  discard;

  /* DRAWBUFFERS:01 */
  gl_FragData[0] = color;
  gl_FragData[1] = vec4(0.0, 0.0, 1.0, 1.0);
}

#version 120

const float sunPathRotation       = -35.0;

varying vec4 color;

void main() {
/* DRAWBUFFERS:01 */
  gl_FragData[0] = color;
  gl_FragData[1] = vec4(0.0, 0.0, 1.0, 1.0);
}

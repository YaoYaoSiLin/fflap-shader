#version 120

varying vec3 vP;
varying vec4 color;

void main() {
  color = gl_Color;

  gl_Position = gl_ModelViewMatrix * gl_Vertex;
  vP = gl_Position.xyz;
  gl_Position = gl_ProjectionMatrix * gl_Position;
}

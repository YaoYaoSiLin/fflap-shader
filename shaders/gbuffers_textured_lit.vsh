#version 130

out vec2 texcoord;
out vec2 lmcoord;

out vec3 vP;

out vec4 color;

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;

  color = gl_Color;

  gl_Position = ftransform();
}


out vec2 lmcoord;
out vec2 texcoord;
out vec4 color;

#if Normal == 1
  out vec3 normal;
#elif Normal == 2
  in vec4 at_tangent;

  out vec3 normal;
  out vec3 tangent;
  out vec3 binormal;
#endif

#if ViewPosition == 1
out vec3 vP;
#endif

#ifdef Enabled_TAA
  uniform int frameCounter;

  uniform float viewWidth;
  uniform float viewHeight;

  vec2 resolution = vec2(viewWidth, viewHeight);
  vec2 pixel = 1.0 / vec2(viewWidth, viewHeight);

  #include "libs/jittering.glsl"
#endif

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  color = gl_Color;

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

  #if ViewPosition == 1
    vP = (gl_ModelViewMatrix * gl_Vertex).xyz;
  #endif

  #if Normal == 1
    normal  = normalize(gl_NormalMatrix * gl_Normal);
  #elif Normal == 2
    normal  = normalize(gl_NormalMatrix * gl_Normal);
    tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
    binormal = cross(tangent, normal);
  #endif

  #ifdef Enabled_TAA
    gl_Position.xy += jittering * gl_Position.w * pixel;
  #endif
}

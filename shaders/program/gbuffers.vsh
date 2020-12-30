
out vec2 lmcoord;
out vec2 texcoord;
out vec4 color;

#if PortalParticlesFix
  out float portal;
  attribute vec4 mc_Entity;
#endif

#if Normal == 1
  out vec3 normal;
#elif Normal == 2
  in vec4 at_tangent;

  out vec3 normal;
  out vec3 tangent;
  out vec3 binormal;
#endif

#ifdef ViewVector
out vec3 vP;
#endif

uniform vec2 jitter;

#ifdef GbufferHand
  attribute vec4 mc_Entity;

  uniform mat4 gbufferProjectionInverse;
  uniform mat4 gbufferModelViewInverse;
  uniform mat4 gbufferModelView;

  uniform int heldItemId;
  uniform int heldItemId2;
#endif

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  color = gl_Color;

  //gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
  vec4 position = gl_Vertex;

  #ifdef GbufferHand
  if((position.x < -0.5 && heldItemId2 == 461) || (position.x > 0.5 && heldItemId == 461)){
    position.z += 0.0625 * 3.0;
    //position.y -= 0.0625 * 2.0;

    vec4 p = gbufferProjectionInverse * vec4(0.0, 0.0, 0.4, 1.0);
         p /= p.w;
         p.xyz = normalize(p.xyz);
         p.xyz = mat3(gbufferModelViewInverse) * p.xyz;

    //float angle = 0.0625 * (((-p.y + 0.5) / 1.5) * 4.0) * 2.0 * 3.14159265;

    float cameraAngle = dot(p.xyz, vec3(0.0, -1.0, 0.0));
    float inv = 1.0;
    if(cameraAngle < 0.0) inv = -1.0;

    //cameraAngle -= 0.0625;


    float angle = (cameraAngle * cameraAngle) * sign(cameraAngle) * 0.0625 * 6.0 + 0.0625;
          angle = angle * 2.0 * 3.14152965;

    //float angle = 0.14 * 2.0 * 3.14159265;
    mat2 rotate = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    position.yz = position.yz * rotate;
  }
  #endif

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;

  #if PortalParticlesFix
    portal = mc_Entity.x == 119;
  #endif

  #ifdef ViewVector
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
    gl_Position.xy += jitter * 2.0 * gl_Position.w;
  #endif
}

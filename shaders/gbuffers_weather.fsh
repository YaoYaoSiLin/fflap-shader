#version 120

uniform sampler2D texture;
uniform sampler2D gaux1;
uniform sampler2D gaux3;

uniform mat4 gbufferProjection;

uniform float far;
uniform float near;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 vP;

varying vec4 color;

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

void main() {
  vec4 tex = texture2D(texture, texcoord);
       tex *= color;

  //tex.a = pow(tex.a, 0.1);
  //tex.a = 0.0;

  float particlesDistance = (length(vP) - near) / far;
/*
  vec2 uv = nvec3(gbufferProjection * nvec4(vP)).xy * 0.5 + 0.5;

  vec4 lastParticles = texture2D(gaux1, uv);
  float lastParticlesDistance = texture2D(gaux3, uv).z;


  lastParticles.a *= step(particlesDistance, lastParticlesDistance);
*/
  //tex.rgb = mix(tex.rgb, lastParticles.rgb, lastParticles.a);
  //tex.a = max(tex.a, lastParticles.a);

/* DRAWBUFFERS:46 */
  gl_FragData[0] = tex;
  gl_FragData[1] = vec4(0.0, 0.0, particlesDistance, 1.0);
}

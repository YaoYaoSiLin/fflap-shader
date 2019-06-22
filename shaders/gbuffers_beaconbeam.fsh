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
varying vec3 normal;

varying vec4 color;

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 tex = texture2D(texture, texcoord);
       tex *= color;

  if(tex.a < 0.001) discard;

/* DRAWBUFFERS:45 */
  gl_FragData[0] = tex;
  gl_FragData[1] = vec4(lmcoord.x, 1.0, gl_FragCoord.z, 1.0); //disable torchlighting and skylighting, b is depth of beam
}

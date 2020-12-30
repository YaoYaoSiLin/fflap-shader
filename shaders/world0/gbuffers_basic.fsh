#version 130

const int RGBA8   = 1;
const int RGB16   = 2;
const int RGBA16  = 2;
const int RGBA32UI = 3;

const int colortex0Format  = RGBA8;
const int colortex1Format  = RGBA16;
const int colortex2Format  = RGBA16;
const int colortex3Format  = RGBA16;
const int colortex4Format  = RGBA16;
const int colortex5Format  = RGBA16;
const int colortex6Format  = RGBA32UI;
const int colortex7Format  = RGBA16;

const float sunPathRotation       = -35.0;    //[-35.0 -30.0 -25.0 -20.0 -15.0 -10.0 -5.0 0.0]
const float ambientOcclusionLevel = 0.0;

uniform mat4 gbufferProjectionInverse;

in vec2 lmcoord;
in vec2 texcoord;

in vec3 normal;

in vec4 color;

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

/* DRAWBUFFERS:0123 */

void main() {
  vec2 encodeNormal = vec2(0.0);
  float mask = 247.0 / 255.0;

  gl_FragData[0] = color;
  gl_FragData[1] = vec4(0.0, 1.0, 0.0, 0.0);
  gl_FragData[2] = vec4(encodeNormal, mask, 1.0);
  gl_FragData[3] = vec4(encodeNormal, 0.0, 1.0);
}

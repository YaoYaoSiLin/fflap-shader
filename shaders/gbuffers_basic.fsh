#version 130

const int RGB8    = 1;
const int RGBA8   = 1;
const int RGB16   = 2;
const int RGBA16  = 2;

const int colortex0Format  = RGBA8;
const int colortex1Format  = RGBA8;
const int colortex2Format  = RGBA16;
const int colortex3Format  = RGBA16;
const int colortex4Format  = RGBA16;
const int colortex5Format  = RGBA16;
const int colortex6Format  = RGBA16;
const int colortex7Format  = RGBA16;

const float ambientOcclusionLevel = 0.0;

in vec2 lmcoord;

in vec4 color;

void main() {
/* DRAWBUFFERS:013 */
  gl_FragData[0] = color;
  gl_FragData[1] = vec4(vec2(0.0), 252.0 / 255.0, 1.0);
  gl_FragData[2] = vec4(vec3(0.0), 1.0);
}

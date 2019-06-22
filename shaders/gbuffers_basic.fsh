#version 120

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

/*
gbuffers
 0 color
 1 lightmap id parallaxShadow
 2 normal x x
 3 PBR data
 4 particels
 5 x
 6 x x particelDepth x
 7 x

deferred
 0 ~
 1 ~
 2 ~
 3 ~
 4 ~
 5 color issky
 6 ao->rays x ~ ~

water



composite
 0 ~
 1 ?
 2 normal ~ ~
 3 PBR data
 4 ~
 5 color
 6 ~ ~ particelDepth ~
 7 temp
*/

varying vec2 lmcoord;

varying vec4 color;

void main() {
/* DRAWBUFFERS:023 */
  gl_FragData[0] = color;
  gl_FragData[2] = vec4(vec2(0.0), 1.0, 0.0);
  gl_FragData[3] = vec4(vec3(0.0), 1.0);
}

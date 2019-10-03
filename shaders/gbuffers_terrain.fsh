#version 130

#extension GL_ARB_shader_texture_lod : require

const int noiseTextureResolution = 64;

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D noisetex;

uniform sampler2D gaux2;

uniform float rainStrength;

uniform ivec2 atlasSize;

uniform vec3 upPosition;
uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;

//in float cutoutBlock;
in float id;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec3 vP;
in vec3 viewVector;

in vec4 color;
/*
float haltonSequence(in int index, in float base) {
  float result = 0.0;
  float ib = 1.0 / base;
  float f = ib;

  float i = float(index);

  while(i > 0.0){
    result += f * mod(i, base);
    i = floor(i / base);
    f *= ib;
  }

  return result;
}

vec2 haltonSequenceOffsets2n3(in int index) {
  return vec2(haltonSequence(index, 2.0), haltonSequence(index, 3.0));
}
*/
vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

#include "libs/common.inc"

#define POM_Mode
#define POM_Depth 3.2         //[1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2]
#define POM_Steps 8           //[8 12 16 20 24 28 32]

#define tileResolution 16      //[auto 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192]

//#define Continuum2_Texture_Format

#if tileResolution == auto
#undef tileResolution
#define tileResolution sqrt(textureSize(normals, 0).x) * 2.0
#endif

vec2 OffsetCoord(in vec2 coord, in vec2 offset, in ivec3 tileSize){
  vec2 textureSize = (tileSize.z) / vec2(tileSize.xy);

	vec2 offsetCoord = coord + mod(offset.xy, textureSize);

	vec2 minCoord = vec2(coord.x - mod(coord.x, textureSize.x), coord.y - mod(coord.y, textureSize.y));
	vec2 maxCoord = minCoord + textureSize;
/*
  if(offsetCoord.x < minCoord.x){
    offsetCoord.x += textureSize.x;
  }
  if(maxCoord.x < offsetCoord.x){
    offsetCoord.x -= textureSize.x;
  }
  if(offsetCoord.y < minCoord.y){
    offsetCoord.y += textureSize.y;
  }
  if(maxCoord.y < offsetCoord.y){
    offsetCoord.y -= textureSize.y;
  }
*/

  offsetCoord.x += textureSize.x * step(offsetCoord.x, minCoord.x);
  offsetCoord.x -= textureSize.x * step(maxCoord.x, offsetCoord.x);
  offsetCoord.y += textureSize.y * step(offsetCoord.y, minCoord.y);
  offsetCoord.y -= textureSize.y * step(maxCoord.y, offsetCoord.y);

	return offsetCoord;
}

vec2 ParallaxMapping(in vec2 coord, in vec3 vP, in float distance){
  int steps = POM_Steps;
  float istep = 1.0 / steps;

  float d = clamp((-distance + 32.0 - 2.0) / 8.0, 0.0, 1.0);

  if(texture2D(normals, coord).a < 1.0 && distance < 32.0){
    vec2 offset = vP.xy / (vP.z) * istep * d;
         offset *= 0.0078;
         offset /= vec2(atlasSize.x / atlasSize.y, 1.0);

    float layerHeight = istep;

    float height = 0.0;
    float heightMapLast = 0.0;

    float l = 0.0;

    for(int i = 0; i < steps; i++){
      height -= layerHeight;
      float heightMap = (texture2D(normals, coord).a - 1.0) * POM_Depth;

      if(height < heightMap) break;
      coord = OffsetCoord(coord, offset * (height + (heightMap - height)), ivec3(atlasSize, tileResolution));
    }
  }

  return coord;
}

//#define Enabled_Soft_Parallax_hadow

float ParallaxShadow(in vec2 coord, in vec3 vP, in float distance){
	float shading = 0.0;

  int steps = 4;

  float angle = vP.z * vP.z;

  float difference = clamp((-distance + 32.0) / 16.0, 0.0, 1.0);

  if(texture2D(normals, coord).a < 1.0 && distance < 32.0 && angle > 0.05 && angle < 0.95) {
    difference *= clamp01((1.0 - abs(angle - 0.5) * 2.0) * 4.0);

    vec2 offset = vP.xy / (vP.z) / steps;
         offset *= 0.0025;
         offset = -offset;

    float layerHeight = (1.0 / steps);
    float height = 0.0;

    coord = floor(coord * atlasSize) / (atlasSize);

    float heightMapRaw = (texture2D(normals, coord).a - 1.0) * POM_Depth;
    float heightMap = heightMapRaw;

    for(int i = 0; i < steps; i ++){
    height -= layerHeight;

    float offsetHeight = heightMap;
          offsetHeight = (height + (offsetHeight - height));

    coord = OffsetCoord(coord, offset * offsetHeight, ivec3(atlasSize, tileResolution));
    heightMap = (texture2D(normals, coord).a - 1.0) * POM_Depth;

    if(heightMap > height && heightMap - 0.004 > heightMapRaw) {
      #ifdef Enabled_Soft_Parallax_hadow
      shading += 1.0 / steps;
      #else
      shading = 1.0;
      break;
      #endif
    }
    }
  }

  shading = 1.0 - shading;
  #ifdef Enabled_Soft_Parallax_hadow
  shading = pow(shading, 5.0);
  #endif

	return mix(1.0, shading, difference);
}


void main() {
  //if(dot(normal, normalize(vP)) > 0.0) discard;
  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  bool backSide = dot(normal, normalize(vP)) > 0.0;

  //vec3 n = normal;
  //vec3 t = normalize(cross(normalize(upPosition), normal));
  //vec3 b = cross(n, t);
  //mat3 tbn = mat3(t, b, n);

	float distance = length(vP);

  vec2 uv = texcoord;
       uv = ParallaxMapping(uv, normalize(viewVector * tbnMatrix), distance);

  vec4 albedo = texture2D(texture, uv) * color;

  if(albedo.a < 0.2) discard;
  albedo.a = 1.0;

  float preShadow = dot(normalize(shadowLightPosition), normal);
        preShadow = clamp(pow5(preShadow * 10.0), 0.0, 1.0);
        preShadow *= ParallaxShadow(uv, normalize(shadowLightPosition * tbnMatrix), distance);
  //float preShadow = ParallaxShadow(uv, normalize(shadowLightPosition * tbnMatrix), distance);

  vec4 speculars = texture2D(specular, uv);
       //speculars = vec4(vec3(0.0), 1.0);
       speculars.a = 1.0;

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  if(speculars.r + speculars.g + speculars.b == speculars.r * 3.0) speculars.rgb = vec3(speculars.r, 0.0, 0.0);
  speculars.r = clamp(speculars.r, 0.001, 0.999);
  speculars.b *= 0.12;

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  vec3 surfaceNormal = texture2D(normals, uv).xyz * 2.0 - 1.0;
       surfaceNormal = normalize(tbnMatrix * surfaceNormal);
  if(backSide) surfaceNormal = -surfaceNormal;
  vec3 visibleNormal = surfaceNormal - (surfaceNormal - normal) * step(-0.15, dot(normalize(vP), surfaceNormal));

  surfaceNormal.xy = normalEncode(surfaceNormal);
  visibleNormal.xy = normalEncode(visibleNormal);

/* DRAWBUFFERS:01235 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, id / 255.0, 1.0);
  gl_FragData[2] = vec4(visibleNormal.xy, preShadow, 1.0);
  gl_FragData[3] = speculars;
  gl_FragData[4] = vec4(surfaceNormal.xy, 0.0, 1.0);
}

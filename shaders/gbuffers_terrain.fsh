#version 130

#define POM_Depth 1.0    //[0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0]  R3D POM(AddictioN974) about 1.2,
#define POM_Steps 8      //[8 12 16 20 24 28 32]

#define tileResolution 128    //[4 8 16 32 64 128 256 512 1024 2048 4096 8192]

#define Continuum2_Texture_Format

const int noiseTextureResolution = 64;

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D noisetex;

uniform float rainStrength;

uniform ivec2 atlasSize;

uniform vec3 upPosition;
uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

//in float cutoutBlock;
in float id;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec3 vP;

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

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

vec2 OffsetCoord(in vec2 coord, in vec2 offset)
{
  vec2 tileResolutionFix = (tileResolution) / vec2(atlasSize);

	vec2 offsetCoord = coord + mod(offset.xy, tileResolutionFix);

	vec2 minCoord = vec2(coord.x - mod(coord.x, tileResolutionFix.x), coord.y - mod(coord.y, tileResolutionFix.y));
	vec2 maxCoord = minCoord + tileResolutionFix;

	if (offsetCoord.x > maxCoord.x) {
		offsetCoord.x -= tileResolutionFix.x;
	} else if (offsetCoord.x < minCoord.x) {
		offsetCoord.x += tileResolutionFix.x;
	}

	if (offsetCoord.y > maxCoord.y) {
		offsetCoord.y -= tileResolutionFix.y;
	} else if (offsetCoord.y < minCoord.y) {
		offsetCoord.y += tileResolutionFix.y;
	}

	//offsetCoord /= atlasSize;

	return offsetCoord;
}

vec2 ParallaxMapping(in vec2 coord, in vec3 vP, in float distance){
  vec2 r = coord;

  int steps = int(POM_Steps);

  float d = clamp((-distance + 32.0 - 2.0) / 8.0, 0.0, 1.0);

  if(texture2D(normals, coord).a < 1.0 && distance < 32.0){
    vec2 dt = vP.xy / abs(vP.z) / steps;
         dt *= d * 0.001 * (POM_Depth * POM_Depth);

    float layerHeight = 1.0 / steps / (POM_Depth * POM_Depth);
          //layerHeight *= 0.1;


    /*
    *texture be more deep
    ^|
    ||       ...
    D|      _/
    e|    _/
    e|  _/
    p|_/
     |-------------
      i+ --->
============================
    *raycast
    ^|...
    ||   \_
    D|     \_
    e|       \_
    e|         \_
    p|           \
     |-------------
      i+ --->
============================
    ?_?
    */
    float height = 0.0;
/*
    for(int i = 0; i < steps; i++){
      float heightMap = texture2D(normals, coord).a - 1.0;
      height -= layerHeight;

      if(heightMap > height) break;
      coord = OffsetCoord(coord, dt * 10.0 * -(height));
    }
*/
    //r = OffsetCoord(coord, dt * 10.0);

    for(int i = 0; i < steps; i++){
    //if(texture2D(normals, coord).a < texture2D(normals, r).a)
    //r = OffsetCoord(r, dt * 10.0);
      float heightMap = texture2D(normals, coord).a - 1.0;
      height -= layerHeight;

      if(heightMap > height) break;

      coord = OffsetCoord(coord, dt * (-height + (-heightMap + height)));
    }
  }

  return coord;
}

//#define Enabled_Soft_Parallax_hadow

float ParallaxShadow(in vec2 coord, in vec3 vP, in float distance){
	float shading = 0.0;

  //return 1.0;

  int steps = 4;

  if(texture2D(normals, coord).a < 1.0 && distance < 32.0) {
    vec2 dt = vP.xy / abs(vP.z) / steps;
         dt *= 0.0005 * POM_Depth;
         //dt = -dt;

    float layerHeight = 1.0 / steps;

    float height = 0.0;

    float fixSize = 128.0 / tileResolution;
          fixSize *= 0.25;

    coord = floor(coord * tileResolution * tileResolution * fixSize) / (tileResolution * tileResolution * fixSize);

    float heightMapRaw = texture2DLod(normals, coord, 0).a;

    for(int i = 0; i < steps; i ++){
    height += layerHeight;
    coord = OffsetCoord(coord, dt * (height));
    float heightMap = texture2DLod(normals, coord, 0).a;

    if(heightMap < height + layerHeight && heightMap - 0.007 > heightMapRaw) {
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

  float d = clamp((-distance + 32.0 - 2.0) / 8.0, 0.0, 1.0);

	return mix(1.0, shading, d);
}


void main() {
  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  //vec3 n = normal;
  //vec3 t = normalize(cross(normalize(upPosition), normal));
  //vec3 b = cross(n, t);
  //mat3 tbn = mat3(t, b, n);

	float distance = length(vP);

  float l = 1.0;

  vec2 uvP = texcoord;
       uvP = ParallaxMapping(uvP, normalize(vP * tbnMatrix), distance);

	float preShadow = clamp(dot(normalize(shadowLightPosition), normalize(normal)), 0.0, 1.0);
        preShadow = clamp(pow(preShadow, 5.0) * 100000, 0.0, 1.0);
        preShadow *= ParallaxShadow(uvP, normalize(shadowLightPosition * tbnMatrix), distance);

  vec4 albedo = texture2D(texture, uvP) * color;
       albedo.a = step(0.01, albedo.a);
       if(albedo.a < 0.001) discard;

  //albedo.rgb = vec3(1.0);

  //albedo.rgb = vec3(1.022,0.782,0.344);

  vec4 speculars = texture2DLod(specular, uvP, 0.0);
       //speculars.a = 1.0;
       //speculars.r = pow(speculars.r, 1.2);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g * 0.0, speculars.a);
  #endif

  //speculars.b = max(speculars.b, texture2DLod(specular, uvP, 1).b + texture2DLod(specular, uvP, 2).b * 0.5);

  vec3 normalTexture = texture2D(normals, uvP).xyz * 2.0 - 1.0;

  //normalTexture.xy *= 1.0 - speculars.g * pow(speculars.r, 3.0) * 0.58;
  //normalTexture.xy *= pow(1.0 - speculars.r, 2.0);
  //normalTexture.xy *= .0;
  normalTexture = normalize(tbnMatrix * normalTexture);
  normalTexture.xy = normalEncode(normalTexture);

  //albedo.rgb = vec3(1.022, 0.782, 0.344);
  //speculars.r = 1.0;
  //speculars.g = 1.0;

  //speculars.r = max(speculars.r, metalBlock * 0.782);
  //speculars.g = max(speculars.g, metalBlock);

  //if(nvec3(gbufferProjection * nvec4(vP)).x * 0.5 + 0.5 < 0.5)albedo.rgb = vec3(1.0);

  //albedo.rgb = mat3(gbufferModelViewInverse) * gbufferModelView[1].xyz;

  //albedo.rgb = mat3(gbufferModelViewInverse) * tangent;
  //albedo.rgb = mat3(gbufferModelViewInverse) * normalize(mat3(t, b, n) * (texture2D(normals, uvP).xyz * 2.0 - 1.0));
  //albedo.rgb = max(vec3(0.0), albedo.rgb * 0.5 + 0.5);

/* DRAWBUFFERS:01235 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, id / 255.0, 1.0);
  gl_FragData[2] = vec4(normalTexture.xy, preShadow, 1.0);
  gl_FragData[3] = speculars;
  gl_FragData[4] = vec4(vec3(0.0), 1.0);
}

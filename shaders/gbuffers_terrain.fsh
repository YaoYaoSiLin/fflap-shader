#version 130

#extension GL_ARB_shader_texture_lod : require

#define Only_Normal_Alpha   0 //
#define Only_Texture_Alpha  1 //
#define Only_Normal_Blue    2 //

#define Texture_Alpha_first 0 //
#define Normal_Alpha_First  1 //
#define Normal_Blue_First   2 //

#define POM_Mode
#define POM_Depth 1.4         //[0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0]
#define POM_Steps 8           //[8 12 16 20 24 28 32]

#define tileResolution 128      //[0 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192]
#define POM_Self_Shadow_Fix 1.0 //[0.25 0.5 1.0 2.0 4.0]

#define Continuum2_Texture_Format

#if tileResolution == 0
#undef tileResolution
#define tileResolution sqrt(textureSize(normals, 0).x) * 2.0
#endif

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
uniform mat4 gbufferProjection;

//in float cutoutBlock;
in float id;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec3 vP;

in vec4 color;

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

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

#include "libs/common.inc"

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
  int steps = int(POM_Steps);

  float d = clamp((-distance + 32.0 - 2.0) / 8.0, 0.0, 1.0);

  if(texture2D(normals, coord).a < 1.0 && distance < 32.0){
    vec2 dt = vP.xy / length(vP.xyz) / steps * d;
         dt *= 0.01;
         //dt = dt * 0.5 + 0.5;
         dt /= vec2(atlasSize) / max(atlasSize.x, atlasSize.y);
         //dt = dt * 2.0 - 1.0;


    float layerHeight = (1.0 / steps);
    //#ifndef Continuum2_Texture_Format
    //      layerHeight /= POM_Depth * POM_Depth * 0.5;
    //#endif

    float height = 0.5;

    for(int i = 0; i < steps; i++){
      float heightMap = texture2D(normals, coord).a - 1.0;
      height -= layerHeight;

      if(heightMap > height) break;

      coord = OffsetCoord(coord, dt * (-height + (-heightMap + height)) * POM_Depth * POM_Depth * 0.5);
      //-height + (-heightMap + height)
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
    vec2 dt = vP.xy / steps;
         dt *= 0.02;
         //dt /= vec2(atlasSize) / max(atlasSize.x, atlasSize.y);
         //dt = -dt;

    float layerHeight = 1.0 / steps;

    float height = 0.0;

    vec2 tileSizeReSet = atlasSize;
    //float lowDetail = 1.0 / (tileResolutionAuto / 64.0);

    coord = floor(coord * tileSizeReSet) / (tileSizeReSet);
    //coord = OffsetCoord(coord, dt * -(layerHeight));

    float heightMapRaw = texture2D(normals, coord).a - 1.0;

    for(int i = 0; i < steps; i ++){
    height -= layerHeight;
    float heightMap = texture2D(normals, coord).a - 1.0;
    coord = OffsetCoord(coord, dt * (-height + (-heightMap + height)) * POM_Depth * POM_Depth * 0.5);

    if(heightMap > height && heightMap - 0.001 > heightMapRaw) {
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

  float d = clamp((-distance + 32.0) / 16.0, 0.0, 1.0);

	return mix(1.0, shading, d);
}


void main() {
  //if(dot(normal, normalize(vP)) > 0.0) discard;
  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  //vec3 n = normal;
  //vec3 t = normalize(cross(normalize(upPosition), normal));
  //vec3 b = cross(n, t);
  //mat3 tbn = mat3(t, b, n);

	float distance = length(vP);

  float l = 1.0;

  vec2 uvP = texcoord;
  //if(nvec3(gbufferProjection * nvec4(vP)).x * 0.5 + 0.5 < 0.5)
  uvP = ParallaxMapping(uvP, normalize(vP * tbnMatrix), distance);

	float preShadow = clamp(dot(normalize(shadowLightPosition), normalize(normal)), 0.0, 1.0);
        preShadow = clamp(pow(preShadow, 5.0) * 100000, 0.0, 1.0);
        preShadow *= ParallaxShadow(uvP, normalize(shadowLightPosition * tbnMatrix), distance);

  vec4 albedo = texture2D(texture, uvP) * color;

  if(albedo.a < 1.0 / tileResolution + 0.04) discard;
  albedo.a = step(0.001, albedo.a);
       //albedo.rgb = albedo.a > 0.2 ? albedo.rgb : vec3(0.5);
       //albedo.a = 1.0;

  //albedo.rgb = vec3(1.0);

  //albedo.rgb = vec3(1.022,0.782,0.344);

  vec4 speculars = texture2D(specular, uvP);
       speculars = vec4(vec3(0.0), 1.0);
       speculars.a = step(0.001, speculars.a);
       //speculars.r = pow(speculars.r, 1.2);

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g * 0.0, speculars.a);
  #endif

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  //if(speculars.a < 0.001) speculars.r = 0.21;

  speculars.b = 0.0;
  speculars.r = mix(0.2, speculars.r, speculars.a);

  //speculars.b = max(speculars.b, texture2DLod(specular, uvP, 1).b + texture2DLod(specular, uvP, 2).b * 0.5);

  vec3 normalTexture = texture2D(normals, uvP).xyz * 2.0 - 1.0;
       normalTexture.xy *= step(0.0, -dot(normalize(vP), normalize(tbnMatrix * normalTexture)));
       normalTexture = normalize(tbnMatrix * normalTexture);

  //if(clamp01(dot(normalize(vP), normalTexture)) <= 0.0) discard;


  vec2 c = vec2(atlasSize.xy) / max(atlasSize.x, atlasSize.y);

  int lowDetailLevel = 8;
  vec2 LoDResolution = vec2(tileResolution) / lowDetailLevel;
  vec2 LoDPixel = 1.0 / LoDResolution;

  vec2 tileSizePixel = vec2(atlasSize) / tileResolution;

  vec2 pixel4x4 = (round((texcoord * vec2(atlasSize) - 8.0 / lowDetailLevel) * LoDPixel)) / vec2(atlasSize) * LoDResolution;
  //vec2 pixel0 = (floor(texcoord * vec2(atlasSize) / tileResolution)) / vec2(atlasSize) * tileResolution;
  vec3 tileTextureAverage = vec3(0.0);
  vec2 pixel0 = floor(texcoord * tileSizePixel) / tileSizePixel + (1.0 / atlasSize);

  int count = 0;

  vec3 maxColor = vec3(0.0);
  vec3 minColor = vec3(1.0);

  float maxLum = 0.0;

  vec2 tileMaxCoord = texcoord + (1.0 / atlasSize) * tileResolution;
  //vec2 tileMinCoord =

  for(int i = 0; i < 16; i++) {
    vec2 ramdomForm = (pixel0 + haltonSequenceOffsets2n3(i) / atlasSize * tileResolution * 0.99);

    vec4 textureTemp = texture2D(texture, ramdomForm) * color;
    tileTextureAverage += textureTemp.rgb;
    maxColor = max(maxColor, textureTemp.rgb);
    //minColor = min(minColor, textureTemp.rgb);
    maxLum = max(maxLum, (textureTemp.r + textureTemp.g + textureTemp.b) * 0.3333);
  }

  tileTextureAverage /= 16.0;

  /*
  for(int i = 0; i < lowDetailLevel; i++){
    for(int j = 0; j < lowDetailLevel; j++){
      vec4 textureTemp = texture2D(texture, floor(pixel0 * LoDPixel * atlasSize + vec2(i, j) + 0.01) / vec2(atlasSize) * LoDResolution);

      //if(textureTemp.a > 0.001){
        tileTextureAverage += textureTemp.rgb;

        maxColor = max(textureTemp.rgb, maxColor);
        //minColor = max(textureTemp.rgb, minColor);

        count++;
      //}
    }
  }
  */

  //tileTextureAverage *= 1.0 / count;

  tileTextureAverage *= color.rgb;
  maxColor *= color.rgb;

  //float up =

  //albedo.rgb = tileTextureAverage;

  //speculars.r = 1.0 - (dot03(maxColor) / maxComponent(tileTextureAverage));// / maxComponent(tileTextureAverage);
  //speculars.r = pow2(speculars.r);

  //albedo.rgb = vec3( pow2(1.0 - (getLum(maxColor) / maxComponent(tileTextureAverage)) * 0.7 * getLum(maxColor)) );

  float highLight = (maxColor.r + maxColor.g + maxColor.b) * 0.333;
        //highLight = pow5(1.0 - abs(highLight - 1.0));
  //speculars.r = (dot03(maxColor) * highLight) / (maxComponent(tileTextureAverage)) * dot03(maxColor);
  //albedo.rgb = vec3((highLight * dot03(maxColor)) / (1.001 - maxComponent(tileTextureAverage)) * dot03(maxColor));

  //speculars.r = highLight / maxComponent(tileTextureAverage) * 0.7;
  //speculars.r = 1.0 - pow2(1.0 - (dot03(maxColor) / maxComponent(tileTextureAverage)) * 0.7 * dot03(maxColor));

  //speculars.r = highLight / (0.001 + maxComponent(tileTextureAverage)) * dot03(maxColor);
  //speculars.r = pow2(speculars.r) * 20.0;
  //speculars.r = pow5(1.0 - abs(maxLum - 1.0)) / dot03(tileTextureAverage) * clamp01(dot03(maxColor - (tileTextureAverage)) / minComponent(tileTextureAverage) * 10.0);
  speculars.r = pow5(maxLum) / dot03(tileTextureAverage) * clamp01(dot03(maxColor - tileTextureAverage) * maxComponent(tileTextureAverage)) / minComponent(tileTextureAverage) * 2.0;
  //speculars.r *= clamp01(dot03(albedo.rgb * 2.0 - tileTextureAverage.rgb) * 1.0 * maxComponent(tileTextureAverage));
  speculars.r = clamp(speculars.r, 0.001, 0.999);

  //albedo.rgb = maxColor;

  //albedo.rgb = vec3(dot03(maxColor) - minComponent(maxColor));


  //albedo.rgb = tileTextureAverage;
  /*
  vec3 gold = vec3(1.0, 0.782, 0.344);
  gold = gold + (gold - 1.0) * 0.1;

  vec3 goldBlock = gold - tileTextureAverage;
  vec3 goldDetail = gold - albedo.rgb;
  vec3 metalHightLight = clamp01(albedo.rgb - tileTextureAverage);

  //albedo.rgb = abs(goldBlock.rrr + 0.01) * 10.0;

  goldBlock.r = clamp01(abs(dot(goldBlock, goldBlock) + 0.01) * 10.0);

  albedo.rgb = vec3(clamp01(
    sqrt(abs(dot(goldBlock, goldBlock) / minComponent(tileTextureAverage) - 1.1)) * 1.0
    ));
*/
  //if(goldBlock.r > 0.06 && goldBlock.r < 0.16) speculars.rgb = vec3(0.78, 1.0, 0.0);
  //speculars.rg = mix(vec2(0.78, 1.0), speculars.rg, goldBlock.r);

  //albedo.rgb = gold;
  //albedo.rgb = vec3(abs(goldBlock.r - 0.1) * 100.0);
  //albedo.rgb = tileTextureAverage;

  //else if(dot(goldDetail, goldDetail) < 0.032) speculars.rgb = vec3(0.78, 1.0, 0.0);
  //else if(pow2(dot(metalHightLight, metalHightLight)) > 0.08) speculars.rgb = vec3(0.78, 1.0, 0.0);
  //albedo.rgb = metalHightLight;

  //albedo.rgb = albedo.rgb + (tileTextureAverage.rgb - albedo.rgb) * 2.5;

  //albedo.rgb = vec3(abs(dot(gold, gold)) <= 0.04);
  //speculars.rg = mix(vec2(1.0 - dot(vec3(0.3333), abs(tileTextureAverage - albedo.rgb)), 1.0), speculars.rg, step(0.04, dot(gold, gold)));

  int blockID = int(round(id));

  //if(blockID == 35) speculars = vec4(0.23, 0.051, 0.0, 1.0);
  //if(blockID == 235) speculars = vec4(0.846, 0.057, 0.0, 1.0);

  //albedo *= color;

  //albedo.rgb = texture2D(texture, pixel4x4).rgb;
  //avgColor = texture2D(texture, floor(pixel0 * LoDPixel * atlasSize + vec2(2, 2) + 0.001) / vec2(atlasSize) * LoDResolution).rgb;

  //float minColor = maxComponent(albedo.rgb);
  //float maxColor = minComponent(albedo.rgb);
/*
  gold = gold + (gold - 1.0) * 0.25;
  //gold.r = clamp01((dot(gold, gold) / (maxColor) - 0.05) * 20.0);
  if(dot(gold, gold) / minComponent(albedo.rgb) <= 0.1) {speculars.r = 0.72, speculars.g = 1.0;}

  albedo.rgb = gold;
*/
/*
  speculars.r = mix(0.782, speculars.r, gold.r);
  speculars.g = mix(1.0, speculars.g, gold.r);

  albedo.rgb = avgColor / minComponent(avgColor) * 0.1;//vec3(minComponent())
*/
  //vec3 plants = color.rgb;
  //     plants = 1.0 - plants;
  //     plants.r = dot(color.rgb, plants);
  //if(plants.r > 0.5);
  //speculars.r = mix(speculars.r, 0.7, pow(plants.r, 0.5));


  //albedo.rgb = avgColor * color.rgb;

  //float blockID = id;
  //if(-dot(normal, normalize(vP)) < 0.06) blockID = 1.0;

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, id / 255.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalTexture), preShadow, 1.0);
  gl_FragData[3] = speculars;
}

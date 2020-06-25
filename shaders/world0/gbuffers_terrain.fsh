#version 130

#extension GL_ARB_shader_texture_lod : require

//#define winter_mode

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
uniform int blockEntityId;
in float id;
in float spruce_leaves;
in float isBlockEntity;

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

#include "../libs/common.inc"

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
  vec2 textureSize = float(tileSize.z) / vec2(tileSize.xy);

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


vec2 OffsetCoord(in vec2 coord, in vec2 offset, in vec2 textureSize){
	vec2 offsetCoord = coord + mod(offset.xy, textureSize);

	vec2 minCoord = vec2(coord.x - mod(coord.x, textureSize.x), coord.y - mod(coord.y, textureSize.y));
	vec2 maxCoord = minCoord + textureSize;

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

float hash(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.2031);
	p3 += dot(p3, p3.yzx + 19.19);
	return fract((p3.x + p3.y) * p3.z);
}

//The Unreasonable Effectiveness of Quasirandom Sequences | Extreme Learning
float t(in float z){
  if(0.0 <= z && z < 0.5) return 2.0*z;
  if(0.5 <= z && z < 1.0) return 2.0 - 2.0*z;
}

float R2sq(in vec2 coord){
  float a1 = 1.0 / 0.75487766624669276;
  float a2 = 1.0 / 0.569840290998;

  //coord = floor(coord);

  return t(mod(coord.x * a1 + coord.y * a2, 1));
}

const vec2 haltonSequence_2n3[16] = vec2[16](vec2(0.5    , 0.33333),
                                             vec2(0.25   , 0.66666),
                                             vec2(0.75   , 0.11111),
                                             vec2(0.125  , 0.44444),
                                             vec2(0.625  , 0.77777),
                                             vec2(0.375  , 0.22222),
                                             vec2(0.875  , 0.55555),
                                             vec2(0.0625 , 0.88888),
                                             vec2(0.5625 , 0.03703),
                                             vec2(0.3125 , 0.37037),
                                             vec2(0.8125 , 0.7037 ),
                                             vec2(0.1875 , 0.14814),
                                             vec2(0.6875 , 0.48148),
                                             vec2(0.4375 , 0.81481),
                                             vec2(0.9375 , 0.25925),
                                             vec2(0.03125, 0.59259)
                                           );

void main() {
  //if(dot(normal, normalize(vP)) > 0.0) discard;
  mat3 tbnMatrix = mat3(tangent, binormal, normal);

  bool backFace = !gl_FrontFacing;
  //if(backFace) discard;

	float distance = length(vP);

  vec2 uv = texcoord;
       uv = ParallaxMapping(uv, normalize(viewVector * tbnMatrix), distance);

  vec4 albedo = texture2D(texture, uv) * color;

  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;
  vec3 worldPosition = mat3(gbufferModelViewInverse) * vP + cameraPosition - worldNormal * 0.01;

  #ifdef winter_mode
  bool jingle = false;
  vec3 jingleSpecular = vec3(0.0);

  if(spruce_leaves > 0.5 && worldNormal.y < 0.75){
    //albedo.rgb = vec3(1.0);
    vec2 seed = vec2(worldPosition.x, worldPosition.z);
    vec2 seed2 = vec2(cross(worldPosition, worldNormal).y, worldPosition.y);
         seed = mix(seed2, seed, step(0.5, abs(worldNormal.y)));

    vec2 seed2x2 = floor(seed * 8.0);
    ivec2 seed3 = ivec2(mod(seed2x2.x, 16), mod(seed2x2.y, 16));

    float dither = haltonSequence_2n3[seed3.x].x;
    float dither2 = haltonSequence_2n3[seed3.y].y;
    float hashDither = hash(seed2x2);
    float R2Dither = R2sq(seed2x2);

    if(mod(dither + dither2, 1) > 0.96){

      //albedo.rgb = vec3((randomColor) / (1.0 - randomColor), (1.0 - randomColor) / (randomColor), 0.0);
      //albedo.rgb = vec3(1.0, 0.0, 0.0);
      albedo.r = 1.0;
      albedo.g = (hashDither) / (1.0 - hashDither) * 0.344;
      albedo.b = 0.0;
      albedo.a = 1.0;
      jingle = true;
      uv = texcoord;
      jingleSpecular.r = 0.782;
      jingleSpecular.g = 0.02 + 0.95 * step(hashDither, 0.4);
      //jingleSpecular.b = (albedo.g - albedo.r);
    }else if(worldNormal.y > -0.75){
      //float hashDither2 = hash(floor(seed * 16.0));
      vec2 seed1x1 = floor(seed * 16.0);
           seed1x1 = vec2(seed1x1.x - seed1x1.y, seed1x1.y);
      float lights = mod(hash(seed1x1), 1);

      if(lights > 0.99){
        albedo.rgb = vec3(0.204, 0.156, 0.022);
        albedo.a = 1.0;
        jingleSpecular = vec3(0.32, 0.02, 0.0);
        jingle = true;
      }
    }
  }
  #endif

  vec2 coord = nvec3(gbufferProjection * nvec4(vP)).xy * 0.5 + 0.5;
  coord *= vec2(1920.0, 1080.0);
  coord = floor(coord);

  if(albedo.a < 0.2) discard;
  albedo.a = 1.0;

  vec3 normalBase = normal;
  if(backFace) normalBase = -normalBase;

  float selfShadow = step(0.1, dot(normalize(shadowLightPosition), normalBase));
  //if(id == 18) selfShadow = 1.0;


  selfShadow *= ParallaxShadow(uv, normalize(shadowLightPosition * tbnMatrix), distance);
  //float selfShadow = ParallaxShadow(uv, normalize(shadowLightPosition * tbnMatrix), distance);

  vec4 speculars = texture2D(specular, uv);
       //speculars = vec4(vec3(0.0), 1.0);
       speculars.a = 1.0;

  #ifdef Continuum2_Texture_Format
  speculars = vec4(speculars.b, speculars.r, speculars.g, speculars.a);
  #endif

  if(speculars.r + speculars.g + speculars.b == speculars.r * 3.0) speculars.rgb = vec3(speculars.r, 0.0, 0.0);
  speculars.r = clamp(speculars.r, 0.001, 0.999);
  speculars.b = 0.0;
  //speculars.b *= 0.12;

  //#if MC_VERSION > 11202
  //speculars = vec4(0.001, 0.0, 0.0, 1.0);
  //#endif

  vec4 normalTexture = texture2D(normals, uv);

  vec3 surfaceNormal = normalTexture.rgb * 2.0 - 1.0;
  surfaceNormal = normalize(tbnMatrix * surfaceNormal);
  if(backFace) surfaceNormal = -surfaceNormal;

  #if MC_VERSION > 11404
  if(isBlockEntity > 0.5) surfaceNormal.rgb = normalBase;
  #endif

  //vec3 visibleNormal = surfaceNormal - (surfaceNormal - normal) * step(-0.15, dot(normalize(vP), surfaceNormal));
  vec3 visibleNormal = vec3(0.0);
  vec3 blockNormal = normalBase;

  float mask = id / 255.0;

  #ifdef winter_mode
  if(jingle){
    surfaceNormal = normal;
    visibleNormal = normal;
    selfShadow = 1.0;
    speculars.rgb = jingleSpecular;
  }

  vec2 atlas = vec2(textureSize(texture, 0).xy);
  vec2 atlasWeatherTexture = vec2(textureSize(gaux2, 0).xy);

  vec2 tileFix = atlas / tileResolution;

  vec2 texcoordWeather = texcoord - floor(texcoord * tileFix) / tileFix;
       texcoordWeather = texcoordWeather * tileFix / (atlasWeatherTexture / 16.0);

  vec4 snowTexture = vec4(0.0);
       snowTexture.rgb = texture2D(gaux2, texcoordWeather).rgb;
       snowTexture.a = clamp01(pow5(dot(normalize(upPosition), surfaceNormal) + 0.5));

  if(worldNormal.y > 0.5){
    snowTexture.a = 1.0;
  }else if(worldNormal.y > -0.1) {
    vec4 snowTextureSide = texture2D(gaux2, texcoordWeather + vec2(0.5, 0.0));
         snowTextureSide.a *= clamp01(1.0 + dot(surfaceNormal, normalize(upPosition)));
    snowTexture.rgb = mix(snowTexture.rgb, snowTextureSide.rgb, snowTextureSide.a);
    snowTexture.a = max(snowTexture.a, snowTextureSide.a);
  }

  snowTexture.a *= smoothstep(0.6666, 0.8, lmcoord.y);
  snowTexture.a *= 1.0 - smoothstep(0.4666, 0.6666, lmcoord.x);

  albedo.rgb = mix(albedo.rgb, snowTexture.rgb, snowTexture.a);
  vec3 normalFlat = surfaceNormal - normal;
  surfaceNormal -= normalFlat * 0.33 * snowTexture.a;
  normalFlat = visibleNormal - normal;
  visibleNormal -= normalFlat * 0.33 * snowTexture.a;
  speculars.b *= 1.0 - snowTexture.a;
  speculars.g *= 1.0 - snowTexture.a;
  speculars.r = mix(speculars.r, 0.33, snowTexture.a);
  #endif

  //if(speculars.g > 0.5) speculars.r = 0.7;

  bool isLeaves = CalculateMaskID(18.0, round(id));
  if(isLeaves) {
    selfShadow = 1.0;
  }

  //bool isEndPortal = int(round(id)) == 18;
  //if(isLeaves) albedo.rgb = vec3(1.0);

  //if(texture2D(normals, texcoord).x < 0.01) surfaceNormal = blockNormal;

  surfaceNormal.xy = normalEncode(surfaceNormal);
  blockNormal.xy = normalEncode(blockNormal);

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord, mask, speculars.b);
  gl_FragData[2] = vec4(blockNormal.xy, speculars.r, selfShadow);
  gl_FragData[3] = vec4(surfaceNormal.xy, speculars.g, 1.0);
}

#version 130

uniform sampler2D texture;

uniform mat4 gbufferModelViewInverse;

uniform vec3 cameraPosition;

uniform float rainStrength;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 vP;
in vec3 normal;

in vec4 color;

#define Gen_TnR

#include "../libs/biomes.glsl"
//#include "../lib/packing.glsl"
float pack2x8(in vec2 x){
  float pack = dot(floor(x * 255.0), vec2(1.0, 256.0));
        pack /= (1.0 + 256.0) * 255.0;

  return pack;
}

float pack2x8(in float x, in float y){return pack2x8(vec2(x, y));}

vec2 unpack2x8(in float x){
  x *= 65536.0 / 256.0;
  vec2 pack = vec2(fract(x), floor(x));
       pack *= vec2(256.0 / 255.0, 1.0 / 255.0);

  return pack;
}

float unpack2x8X(in float packge){return (256.0 / 255.0) * fract(packge * (65536.0 / 256.0));}
float unpack2x8Y(in float packge){return (1.0 / 255.0) * floor(packge * (65536.0 / 256.0));}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec3 biomeDustColor = vec3(0.0);
  float temperature = 0.0;
  float rainfall = 0.0;
  CalculateBiomeData(temperature, rainfall, biomeDustColor);
  if(rainfall < 0.001) discard;

  vec4 albedo = texture2D(texture, texcoord) * color;

  //vec3 wP = mat3(gbufferModelViewInverse) * vP + cameraPosition;

  albedo.a = albedo.a * albedo.a;
  if(albedo.a < 0.2) discard;
  albedo.a = 1.0;

  float selfShadow = 1.0;
  float emissive = 0.0;
  vec4 lightmap = vec4(pack2x8(lmcoord), selfShadow, emissive, 1.0);

  vec2 encodeNormal = normalEncode(normal);

  float mask = 250.0 / 255.0;
  float specularPackge = pack2x8(vec2(0.01));

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = vec4(albedo.rgb, 1.0);
  gl_FragData[1] = lightmap;
  gl_FragData[2] = vec4(encodeNormal, mask, 1.0);
  gl_FragData[3] = vec4(encodeNormal, specularPackge, 1.0);
}

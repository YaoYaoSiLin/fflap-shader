#define EXPTWO32MONE 4294967295.0f
#define EXPTWO16MONE 65535.0f
//#define packing2x16_constant

uint packUnorm2x8(in vec2 v){
  v = floor(v * 255.0);
  return uint(v.y) << 8 | uint(v.x);
}

vec2 unpackUnorm2x8(in uint v){
  uvec2 x = (uvec2(v) >> uvec2(0, 8)) & uvec2(0xff);
  return vec2(x) / 255.0;
}

uint packUnorm2x16(in vec2 v){
  v = floor(v * 65535.0);
  return uint(v.y) << 16 | uint(v.x);
}

vec2 unpackUnorm2x16(in uint v){
  uvec2 x = (uvec2(v) >> uvec2(0, 16)) & uvec2(0xffff);
  return vec2(x) / EXPTWO16MONE;
}

float pack2x16(in vec2 v) {return float(packUnorm2x16(v)) / EXPTWO32MONE;}
float pack2x16(in float vx, in float vy) {return float(packUnorm2x16(vec2(vx, vy))) / EXPTWO32MONE;}

vec2  unpack2x16(in float v) { return unpackUnorm2x16(uint(v * EXPTWO32MONE));}
float unpack2x16X(in float v) {return unpackUnorm2x16(uint(v * EXPTWO32MONE)).x;}
float unpack2x16Y(in float v) {return unpackUnorm2x16(uint(v * EXPTWO32MONE)).y;}

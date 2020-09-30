#define INCLUDE_COMMON

#define Solid       1
#define Particels   2
#define Deferred    4
#define World_In_The_Translucent 5
#define Translucent 6
#define ScreenReflection 7
#define Composite   8
#define Bloom       9
#define PostProcessing 10
#define Final       11

#define overRange 38.0  //hdr

#define OFF 0
#define Low 1
#define Medium 2
#define High 3

#define Checker_Board 1
#define Render_Scale 2

#define Simple_Filter 0
#define Temporal 1

#define auto 0
#define default 255
#define render_quality MC_RENDER_QUALITY

const float Pi = 3.14159;
const float invPi = 1.0 / Pi;
const float e = 2.71828;
const float playerEyeLevel = 1.62;
const float dynamicLightHight = 1.62;
const float defaultHightLevel = 63.0;
const float defaultHightLightLevel = 1000000.0;

uniform float far;
uniform float near;

float sky = 1.0 - near / far / far;

#define L2Gamma(c)  pow(vec3(c), vec3(2.2))
#define G2Linear(c) pow(vec3(c), vec3(1.0 / 2.2))

#define lerq(from, to, t) (from + (to - from) * t)

float clamp01(float x){
  return clamp(x, 0.0, 1.0);
}

vec3 clamp01(vec3 x){
  return clamp(x, vec3(0.0), vec3(1.0));
}

#define saturate clamp01

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

float linearizeDepth(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}

float getLum(in vec3 color){
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float dot03(in vec3 x){
  return dot(vec3(0.3333), x);
}

vec3 rgb2L(in vec3 color){
  return pow(color, vec3(2.2));
}

vec3 L2rgb(in vec3 color){
  return pow(color, vec3(1.0 / 2.2));
}

float pow2(in float x){
  return x * x;
}

float pow3(in float x){
  return x * x * x;
}

float pow4(in float x){
  return x * x * x * x;
}

float pow5(in float x){
  return x * x * x * x * x;
}

float minComponent( vec3 a ) {
    return min(a.x, min(a.y, a.z) );
}

float maxComponent( vec3 a ) {
    return max(a.x, max(a.y, a.z) );
}

float rescale(float vmin, float vmax, float v){
  return saturate((v-vmin)/(vmax-vmin));
}

vec3 rescale(vec3 vmin, vec3 vmax, vec3 v){
  return saturate((v-vmin)/(vmax-vmin));
}

#define CalculateMaskID(id, x) bool(step(id - 0.5, x) * step(x, id + 0.5))
#define CalculateMaskIDVert(id, E) step(id - 0.5, E) * step(E, id + 0.5) * id
#define CalculateMaskIDVert2(id, id2, E) step(id - 0.5, E) * step(E, id + 0.5) * id + step(id2 - 0.5, E) * step(E, id2 + 0.5) * id


  float gaussianBlurWeights(in float offset){
    float o = 0.83;

    o *= o;
    offset *= offset;

    return (1/(sqrt(2*Pi*o)))*exp(-(offset)/(2*o));
  }

  float gaussianBlurWeights(in vec2 offset){
    float o = 0.83;

    o *= o;
    offset *= offset;

    return (1.0f/(2.0f*Pi*o))*exp(-((offset.x+offset.y)/(2.0f*o)));
  }


vec3 saturation(in vec3 color, in float s){
  float lum = dot03(color);
  return (lum + (color - lum) * s);
}

vec2 RotateDirection(vec2 V, vec2 RotationCosSin) {
    return vec2(V.x*RotationCosSin.x - V.y*RotationCosSin.y,
                V.x*RotationCosSin.y + V.y*RotationCosSin.x);
}

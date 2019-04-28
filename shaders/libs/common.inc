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

const float Pi = 3.1415;
const float playerEyeLevel = 1.62;
const float defaultHightLevel = 63.0;
const float defaultHightLightLevel = 1000000.0;

uniform float far;
uniform float near;

float sky = 1.0 - near / far / far;

float clamp01(float x){
  return clamp(x, 0.0, 1.0);
}

vec3 clamp01(vec3 x){
  return clamp(x, vec3(0.0), vec3(1.0));
}

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

vec3 rgb2L(in vec3 color){
  return pow(color, vec3(2.2));
}

vec3 L2rgb(in vec3 color){
  return pow(color, vec3(1.0 / 2.2));
}

float pow2(in float x){
  return x * x;
}

float pow5(in float x){
  return x * x * x * x * x;
}

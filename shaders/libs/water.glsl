#ifndef Water_Color_Test
  #define disable -255
  #define swamp 6
  #define frozen_ocean_and_river 10
  #define warm_ocean 44
  #define lukewarm_ocean 45
  #define cold_ocean 46

  #if !defined(default)
  #define default 255
  #endif

  #define Water_Color_Test disable //[disable default swamp frozen_ocean_and_river warm_ocean lukewarm_ocean cold_ocean]
#endif

#ifndef INCLUDE_COMMON
  const float Pi = 3.14159;

  float pow2(in float x){
    return x * x;
  }

  float pow3(in float x){
    return x * x * x;
  }

  float pow5(in float x){
    return x * x * x * x * x;
  }

  float getLum(in vec3 color){
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
  }

  float dot03(in vec3 x){
    return dot(vec3(0.3333), x);
  }

  float minComponent( vec3 a ) {
      return min(a.x, min(a.y, a.z) );
  }

  float maxComponent( vec3 a ) {
      return max(a.x, max(a.y, a.z) );
  }

  vec3 saturation(in vec3 color, in float x){
    float lum = dot03(color);
    return (lum + (color - lum) * x);
  }
#endif

#ifndef Gen_Water_Color
vec4 CalculateWaterColor(in vec4 color){
  //color = vec4(color.rgb, 0.05);

  //color.a = (color.r + color.g) / (min(color.r, color.g) + 0.01) * (1.0 - color.b) * 0.25 / Pi;
  color.a = ((1.0 - color.b) + color.g) / maxComponent(color.rgb) * 0.1 + 0.15;
  //color.a = pow2(color.a);

  //color.a = 0.01;
  //color.rgb = saturation(color.rgb, 1.0 + (min(color.r, color.g) - color.b)) / (maxComponent(color.rgb) * 0.2 + 0.8) * 1.1;
  //color.rgb = color.rgb * color.rgb;

  return color;
}
#endif

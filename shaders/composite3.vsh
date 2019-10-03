#version 130

uniform vec3 vanillaWaterColor;

out vec2 texcoord;

out vec4 waterColor;

#include "libs/water.glsl"

void main() {
  texcoord = gl_MultiTexCoord0.st;

  waterColor = vec4(vanillaWaterColor, 0.05);

  #if Water_Color_Test > disable
    #if Water_Color_Test == normal_biomes
      waterColor.rgb = vec3(0.247 , 0.4627, 0.8941);
    #elif Water_Color_Test == swamp
      waterColor.rgb = vec3(0.3803, 0.4823, 0.3921);
    #elif Water_Color_Test == frozen_ocean_and_river
      waterColor.rgb = vec3(0.2235, 0.2196, 0.7882);
    #elif Water_Color_Test == warm_ocean
      waterColor.rgb = vec3(0.2627, 0.8352, 0.9333);
    #elif Water_Color_Test == lukewarm_ocean
      waterColor.rgb = vec3(0.2705, 0.6784, 0.949 );
    #elif Water_Color_Test == cold_ocean
      waterColor.rgb = vec3(0.2392, 0.3411, 0.8392);
    #endif
  #endif

  #if MC_VERSION < 11202 || !defined(MC_VERSION)
  waterColor.rgb = vec3(0.247 , 0.4627, 0.8941);
  #endif

  waterColor = CalculateWaterColor(waterColor);

  gl_Position = ftransform();
}

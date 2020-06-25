#define disable -255
#define plains 1
#define desert 2
#define forest 4
#define taiga 5
#define swamp 6
#define frozen_ocean 10
#define snowy_tundra 12
#define mushroom_fields 14
#define jungle 21
#define birch_forest 27
#define snowy_taiga 30
#define savanna 35
#define badland 37
#define wooded_badlands_plateau	38
#define badlands_plateau	39
#define warm_ocean 44
#define lukewarm_ocean 45
#define cold_ocean 46

#define Water_Color_Test disable //[disable default swamp frozen_ocean warm_ocean lukewarm_ocean cold_ocean]
#define Temperature_and_Rainfall_Test disable //[disable default plains desert forest taiga swamp snowy_tundra mushroom_fields jungle birch_forest snowy_taiga savanna badland wooded_badlands_plateau badlands_plateau]

#ifdef Gen_Water_Color
//biomes watercolor
uniform vec3 vanillaWaterColor;

vec4 CalculateWaterColor(in vec4 color){
  #if Water_Color_Test > disable
    #if Water_Color_Test == default
      color.rgb = vec3(0.247 , 0.4627, 0.8941);
    #elif Water_Color_Test == swamp
      color.rgb = vec3(0.3803, 0.4823, 0.3921);
    #elif Water_Color_Test == frozen_ocean_and_river
      color.rgb = vec3(0.2235, 0.2196, 0.7882);
    #elif Water_Color_Test == warm_ocean
      color.rgb = vec3(0.2627, 0.8352, 0.9333);
    #elif Water_Color_Test == lukewarm_ocean
      color.rgb = vec3(0.2705, 0.6784, 0.949 );
    #elif Water_Color_Test == cold_ocean
      color.rgb = vec3(0.2392, 0.3411, 0.8392);
    #endif
  #endif

  #if MC_VERSION <= 11202 || !defined(MC_VERSION)
  color.rgb = vec3(0.247 , 0.4627, 0.8941);
  #endif

  color.a = (color.r + color.g) / (min(color.r, color.g) + 0.01) * (1.0 - color.b) * 0.5 / Pi;
  //color.rgb = pow(color.rgb, vec3(2.2));

  return color;
}
//end biomes watercolor
#endif

#ifdef Gen_TnR
//biomes weather data
uniform float biomeTemperature;
uniform float biomeRainFall;

uniform vec3 dustColor;

void CalculateBiomeData(inout float temperature, inout float rainfall, inout vec3 biomeDustColor){
  temperature = biomeTemperature;
  rainfall = biomeRainFall;
  biomeDustColor = dustColor;

  #if Temperature_and_Rainfall_Test > disable
    #if Temperature_and_Rainfall_Test == default
      temperature = 0.5;
      rainfall = 0.5;
    #elif Temperature_and_Rainfall_Test == plains
      temperature = 0.8;
      rainfall = 0.4;
    #elif Temperature_and_Rainfall_Test == desert
      temperature = 2.0;
      rainfall = 0.0;
      biomeDustColor = vec3(0.8588, 0.8274, 0.6274);
    #elif Temperature_and_Rainfall_Test == forest
      temperature = 0.7;
      rainfall = 0.8;
    #elif Temperature_and_Rainfall_Test == taiga
      temperature = 0.25;
      rainfall = 0.8;
    #elif Temperature_and_Rainfall_Test == swamp
      temperature = 0.8;
      rainfall = 0.9;
    #elif Temperature_and_Rainfall_Test == snowy_tundra
      temperature = 0.0;
      rainfall = 0.5;
    #elif Temperature_and_Rainfall_Test == jungle
      temperature = 0.95;
      rainfall = 0.9;
    #elif Temperature_and_Rainfall_Test == snowy_taiga
      temperature = -0.5;
      rainfall = 0.4;
    #elif Temperature_and_Rainfall_Test == savanna
      temperature = 1.2;
      rainfall = 0.0;
    #elif Temperature_and_Rainfall_Test == badland
      temperature = 2.0;
      rainfall = 0.0;
      biomeDustColor = vec3(0.749, 0.4039, 0.1294) * 0.75 + vec3(0.596, 0.3686, 0.2666) * 0.25;
    #elif Temperature_and_Rainfall_Test == badlands_plateau
      temperature = 2.0;
      rainfall = 0.0;
      biomeDustColor = vec3(0.596, 0.3686, 0.2666);
    #elif Temperature_and_Rainfall_Test == wooded_badlands_plateau
      temperature = 2.0;
      rainfall = 0.0;
      biomeDustColor = vec3(0.4666, 0.3372, 0.2313) * 0.6 + vec3(0.596, 0.3686, 0.2666) * 0.4;
    #elif Temperature_and_Rainfall_Test == mushroom_fields
      temperature = 0.9;
      rainfall = 1.0;
    #elif Temperature_and_Rainfall_Test == birch_forest
      temperature = 0.6;
      rainfall = 0.6;
    #endif
  #endif

  //rainfall += rainStrength * rainfall;
}
//end biomes weather data
#endif

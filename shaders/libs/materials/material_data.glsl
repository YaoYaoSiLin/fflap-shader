#ifndef INCLUDE_MATERIAL_DATA 
#define INCLUDE_MATERIAL_DATA

#define disable -255
#define plains 1
#define desert 2
#define mountains 3         //t:0.2, r:0.3
#define forest 4
#define taiga 5
#define swamp 6
#define nether_wastes 8
#define frozen_ocean 10
#define snowy_tundra 12
#define mushroom_fields 14
#define jungle 21
#define birch_forest 27
#define snowy_taiga 30
#define wooded_mountains 34 //t:0.2, r:0.3
#define savanna 35
#define badland 37
#define wooded_badlands_plateau	38
#define badlands_plateau	39
#define warm_ocean 44
#define lukewarm_ocean 45
#define cold_ocean 46

//new biomes add in 1.16
#define soul_sand_valley 170
#define crimson_forest 171 
#define warped_forest 172
#define basalt_deltas 173

//new biomes add in 1.17
#define dripstone_caves 174
#define lush_caves 175

//new biomes add in 1.18
#define meadow 176          
#define grove 177
#define snowy_slopes 178
#define lofty_peaks 179
#define snowcapped_peaks 180
#define stony_peaks 181

#define Water_Color_Test disable //[disable default swamp frozen_ocean warm_ocean lukewarm_ocean cold_ocean]
#define Temperature_and_Rainfall_Test disable //[disable default plains desert forest taiga swamp snowy_tundra mushroom_fields jungle birch_forest snowy_taiga savanna badland wooded_badlands_plateau badlands_plateau]

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

    color.a = ((1.0 - color.b) + color.g) / maxComponent(color.rgb) * 0.1 + 0.15;

    return color;
}
#endif
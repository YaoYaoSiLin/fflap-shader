#version 130

uniform vec3 cameraPosition;

uniform vec3 lightVectorWorld;
uniform vec3 sunVectorWorld;
uniform vec3 moonVectorWorld;
uniform vec3 upVectorWorld;

//uniform int worldTime;

out vec2 texcoord;

out float fading;

out vec3 dayLightingColor;
out vec3 moonLightingColor;
out vec3 sunLightingColor;

out vec3 sunLightingColorRaw;
out vec3 skyLightingColorRaw;

#include "/libs/common.inc"
#include "/libs/volumetric/atmospheric_common.glsl"
#include "/libs/lighting/lighting_color.glsl"

void main() {
    gl_Position = ftransform();

    texcoord = gl_MultiTexCoord0.st;

    //float daytime = float(worldTime);
    //fading = saturate(((abs(daytime - 23000.0) - 200.0) / 800.0) * ((abs(daytime - 12800.0) - 200.0) / 800.0));
    //fading = saturate((abs(daytime - 23200) - 50.0) / 100.0) * saturate((abs(daytime - 13200) - 50.0) / 100.0);
    fading = saturate(abs(sunVectorWorld.y) * 10.0 - 0.5);

    sunLightingColorRaw = vec3(0.0);
    skyLightingColorRaw = vec3(0.0);

    dayLightingColor    = vec3(0.0);
    moonLightingColor   = vec3(0.0);
    sunLightingColor    = vec3(0.0);

    vec3 samplePosition = vec3(0.0, planet_radius + 1.0, 0.0);

    //sunLightingColorRaw = CalculateAtmospheric(samplePosition, sunVectorWorld, sunVectorWorld, vec3(5.0, 5.0, 1.0), 0.999) * SunLight;
    //skyLightingColorRaw = CalculateAtmospheric(samplePosition, upVectorWorld, upVectorWorld, vec3(1.0, 1.0, 1.0), 0.76);

    sunLightingColorRaw = CalculateSunLightColor(E, sunVectorWorld, sunVectorWorld, 10.0, 0.76) * SunLight;
    sunLightingColorRaw += CalculateSunLightColor(E, moonVectorWorld, moonVectorWorld, 1.0, 0.76) * MoonLight;

    skyLightingColorRaw = CalculateSkyLightColor(E, vec3(0.0, 1.0, 0.0), sunVectorWorld, vec2(10.0, 1.0), 0.76);
}
#version 130

uniform sampler2D colortex15;

uniform sampler2D depthtex0;

uniform sampler2D depthtex1;    //

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 lightVectorWorld;
uniform vec3 sunVectorWorld;
uniform vec3 moonVectorWorld;

uniform vec2 resolution;
uniform vec2 pixel;

uniform int moonPhase;

uniform float frameTimeCounter;

in vec2 texcoord;

in float fading;
in vec3 skyLightingColorRaw;
in vec3 sunLightingColorRaw;

vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}
// Assume normalized input. Output is on [-1, 1] for each component.
vec2 float32x3_to_oct(in vec3 v) {
    // Project the sphere onto the octahedron, and then onto the xy plane
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}

vec3 oct_to_float32x3(vec2 e) {
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    return normalize(v.xzy);
}

#include "/libs/common.inc"
#include "/libs/dither.glsl"
#include "/libs/noise_common.glsl"
#include "/libs/volumetric/clouds_common.glsl"
#include "/libs/volumetric/atmospheric_common.glsl"
#include "/libs/volumetric/atmospheric_raymarching.glsl"

//vec3 DrawMoon(in vec3)

const float moon_radius = 1734e3;
const float moon_distance = 38440e3;
const float moon_in_one_tile = 9.0;

vec3 DrawMoon(in vec3 L, vec3 direction, float hit_planet) {
    vec2 traceingMoon = RaySphereIntersection(vec3(0.0) - L * moon_distance, direction, vec3(0.0), moon_radius);
    vec2 traceingMoon2 = RaySphereIntersection(vec3(0.0) - L * moon_distance, L, vec3(0.0), moon_radius);

    mat3 lightModelView = mat3(shadowModelView[0].xy, L.x,
                               shadowModelView[1].xy, L.y,
                               shadowModelView[2].xy, L.z);

    vec3 coord3 = lightModelView * direction; 
    vec2 coord2 = coord3.xy / coord3.z;
         coord2 *= max(0.0, traceingMoon2.x) / moon_radius * inversesqrt(moon_in_one_tile); 
         coord2 = coord2 * 0.5 + 0.5;

    float moon = float(moonPhase);
    vec2 chosePhase = vec2(mod(moon, 4), step(3.5, moon));

    vec2 coord = (coord2 + chosePhase) * vec2(0.25, 0.5);

    vec4 moon_texture = texture2D(depthtex1, coord + chosePhase); moon_texture.rgb = decodeGamma(moon_texture.rgb);
    float hit_moon = float(abs(coord2.x - 0.5) < 0.5 && abs(coord2.y - 0.5) < 0.5 && coord3.z > 0.0) * step(hit_planet, 0.0);

    return moon_texture.rgb * (hit_moon * MoonLight * moon_texture.a * 10.0);    
}

#define Enabled_TAA

void main() {
    vec3 color = vec3(0.0);

    vec2 coord = texcoord;

    #ifdef Enabled_TAA
    coord -= jitter;
    #endif

    gl_FragData[0] = vec4(0.0);
    //if(max(abs(coord.y - 0.5), abs(coord.x - 0.5)) > 0.5) return;

    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, texture(depthtex0, coord).x) * 2.0 - 1.0));
    vec3 wP = mat3(gbufferModelViewInverse) * vP;

    vec3 direction = normalize(wP);

    vec2 tracingPlanet = RaySphereIntersection(E, direction, vec3(0.0), planet_radius);
    vec2 tracingAtmospheric = RaySphereIntersection(E, direction, vec3(0.0), atmosphere_radius);
    vec2 farAtmosphericStart = RaySphereIntersection(E, direction, vec3(0.0), planet_radius + 1600.0);

    color = SimplePlanetSurface(mix(skyLightingColorRaw, sunLightingColorRaw * fading, 0.001), vec3(0.0), direction, vec3(0.0), tracingPlanet.x);

    //float sunDiscAngle = rescale(0.9995, 1.0, dot(direction, sunVectorWorld));
    //color += (SunLight * 10.0) * (step(1e-5, sunDiscAngle) * step(tracingPlanet.x, 0.0) * rescale(-0.05, 1.0, sunDiscAngle));

    //color += texture(depthtex1, texcoord).x;

    //vec3 Tr = rayleigh_extinction * exp(-1.0 / rayleigh_distribution);
    //vec3 Tm = mie_extinction * exp(-1.0 / mie_distribution);

    //normalize(vec3(shadowModelView[0].z, shadowModelView[1].z, shadowModelView[2].z))

    //color += DrawMoon(moonVectorWorld, direction, tracingPlanet.x);

    //if(tracingPlanet.x > 0.0 || tracingPlanet.y > 0.0)
    //color = (sunLightingColorRaw) * exp(-(rayleigh_extinction + mie_extinction) * tracingPlanet.x) * (rayleigh_scattering + mie_scattering) * (tracingPlanet.x);

    //float far_start = max(0.0, tracingAtmospheric.x) + (max(tracingPlanet.x, farAtmosphericStart.x) > 0.0 ? 0.0 : max(0.0, farAtmosphericStart.y));

    //float far_start = tracingAtmospheric.x > 0.0 ? tracingAtmospheric.x : max(0.0, farAtmosphericStart.y) * step(tracingPlanet.x, 0.0);
    //float far_end = tracingPlanet.x > 0.0 ? tracingPlanet.x : tracingAtmospheric.y;

    float far_start = tracingAtmospheric.x > 0.0 ? tracingAtmospheric.x : 0.0; 
    float far_end = tracingPlanet.x > 0.0 ? tracingPlanet.x : max(0.0, tracingAtmospheric.y);

    if(farAtmosphericStart.x < 0.0 && farAtmosphericStart.y > 0.0) {
        far_start += tracingPlanet.x > 0.0 ? 0.0 : farAtmosphericStart.y;
    }else{
        far_start = far_start + (far_end - far_start) * 0.5;
    }

    vec3 atmosphere_color = vec3(0.0);
    vec3 transmittance = vec3(1.0);

    CalculateAtmospheric(transmittance, atmosphere_color, E, direction, sunVectorWorld, far_start, far_end, vec3(0.0, 0.0, 0.0));

    color *= transmittance;
    color += atmosphere_color;

    //color = texture2D(colortex15, (float32x3_to_oct(direction) * 0.5 + 0.5) * 0.5).aaa;

    float middle_start = max(0.0, tracingAtmospheric.x);
    float middle_end = tracingPlanet.x > 0.0 ? tracingPlanet.x : farAtmosphericStart.y;
          middle_start = max(middle_start, (middle_end - middle_start) / float(middle_atmospheric_sample_steps));

    if(farAtmosphericStart.x < 0.0 && farAtmosphericStart.y > 0.0){
    //    CalculateClosestAtmospheric(color, E, direction, sunVectorWorld, middle_start, middle_end);
    }

    color = encodeGamma(color);

    if(tracingPlanet.x > 0.0) transmittance = vec3(0.0);

    gl_FragData[0] = vec4(transmittance, 1.0);
    gl_FragData[1] = vec4(color, 1.0);
}
/* DRAWBUFFERS:45 */
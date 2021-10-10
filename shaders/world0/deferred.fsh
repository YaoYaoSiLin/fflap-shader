#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform sampler2D gaux3;
uniform sampler2D colortex15;

uniform sampler2D depthtex0;

uniform sampler2D shadowcolor0;

uniform vec3 lightVectorWorld;
uniform vec3 sunVectorWorld;
uniform vec3 cameraPosition;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;

uniform vec2 resolution;
uniform vec2 pixel;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

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
#include "/libs/volumetric/atmospheric_common.glsl"
#include "/libs/volumetric/clouds_common.glsl"

float CalculateLowerCloudsSamples(in vec3 p) {
    float H = max(0.0, length(vec3(0.0, planet_radius + e.y, 0.0) + p) - planet_radius);
    float height = (H - lower_clouds_height) / lower_clouds_thickness;

    p += e;

    float t = 0.0 * 0.2;

    p += wind_direction * height * lower_clouds_thickness * 0.2;
    p += wind_direction * t * 25.0;
    
    vec2 cellP = p.xz * 0.001;
    float cell  = worley(cellP + vec2(400.0, 400.0));
          cell += worley(cellP * 2.0 + vec2(1600.0, 1600.0)) * 0.5;
          cell += worley(cellP * 4.0 + vec2(4800.0, 4800.0)) * 0.25;
          cell /= 1.75;
          cell  = rescale(0.717, 1.0, cell);

    vec3 d1 = vec3(1.0, 1.0, 0.0) * 0.4;
    vec3 p1 = p * 0.003 + d1 * t;
    float pn0  = noise(p1); p1 += d1 * t;
          pn0 += noise(p1 * 2.0) * 0.5;
          pn0 /= 1.5;
          pn0 = rescale(0.0, 1.0, pow(pn0, 0.5));

    vec3 d2 = vec3(1.0, -1.0, 0.0) * 0.8;
    vec3 p2 = p * 0.024 + d2 * t;
    float pn1  = noise(p2); p2 += d2 * t * 1.0;
          pn1 += noise(p2 * 2.0) * 0.5; 
          pn1 /= 1.5;

    float height_gradient = rescale(0.1, 0.2, height) * remap(height, 0.6, 0.9, 1.0, 0.0);//remap(height, 0.2, 0.3, 1.0, 0.0) * remap(height, 0.0, 0.1, 0.0, 1.0);//saturate(h / 20.0) * saturate((800.0 - h) / 100.0);
    //float anvil = mix(0.0, 1.0, h / 800.0) + (1.0 - abs(h - 400.0) / 400.0);

    float coverage = pow(1.0 - lower_clouds_coverage, saturate(remap(height, 0.08, 0.8, 1.0, mix(1.0, 0.5, lower_clouds_anvil_bias))));

    float density = (cell * 1.0 + pn0 * 0.5 + pn1 * 0.1) / (1.0 + 0.5 + 0.1);
          density = rescale(coverage, 1.0, density);
          density *= step(lower_clouds_height, H) * step(H, lower_clouds_height + lower_clouds_thickness);
          //density *= height_gradient;
          //density = (shape) * 0.5 * step(lower_clouds_height, H) * step(H, lower_clouds_height + lower_clouds_thickness);
          //density = rescale(1.0, 2.0, density + wn1);

    //float density = rescale(0.5, 1.0, (shape + pn0 * 0.5 + pn1 * 0.2) / 1.7);
          //density = rescale(pow(0.1, mix(0.99, 0.3, h / 800.0)), 1.0, density) * height_gradient;
    //density *= height_gradient;

/*
    float density = ((shape + pn0 * 0.5) / 1.5 + pn1 * 0.2) / 1.2;
          density = rescale(0.5, 1.0, density);
          //density = rescale(0.1, 1.0, density);
    //      density = rescale(0.0, 1.0 - 0.3, density - pn1 * 0.3);
          density = rescale(0.5, 1.0, (density + wn1) / 2.0);
          */
    //float density = rescale(0.739, 1.0, (shape + wn1 + pn0 * 0.3 + mix(wn0, pn2, 0.2)) / 3.3); 
          //density = rescale(0.0, 1.0 - 0.2, density - pn1 * 0.2);

    //float density = rescale(1.5, 2.0, (shape + pn0 * 0.4 + pn1 * 0.2) / 1.6 + worley(p.xz * 0.001));
          //density = rescale(0.3, 1.3, density + 0.3 * pn1);

    /*
    float noise1 = rescale(0.5, 1.0, abs(hash(floor(p.xz * 0.0008))));
    float noise2 = rescale(0.5, 1.0, abs(hash(floor(p.xz * 0.0016))));// * step(1600.0, H) * step(H, 2000.0);
    float noise3 = mix(0.0, abs(hash(floor(p * 0.0032))), saturate((H - 2000.0) / 100.0));

    //float density = step(0.1, rescale(0.9, 3.5, noise1 + noise2 * 2.0 + step(H, 2000.0) * 0.5 - noise3));
    float density = step(0.5, noise1);
*/
/*
    float seed = rescale(0.5, 0.51, abs(hash(floor(p.xz * 0.0004)))) * step(1500.0, H) * step(H, 1900.0);
    float noise0 = abs(hash(floor(p.xz * 0.0008)));
    float noise1 = rescale(0.5, 1.0, abs(hash(floor(p.xz * 0.0016))));
    float noise2 = rescale(0.5, 1.0, abs(hash(floor(p * 0.0032)))) * step(1800.0, H);
    //float noise1 = abs(hash(floor(p * 0.0032)));
    float density = rescale(0.9, 1.0, (noise0 - noise2) + max(0.0, seed - noise1));//rescale(0.4, 1.75, seed + noise0 * 0.5 + noise1 * 0.25);//rescale(0.1, 1.0, (seed + noise0 * 0.5 - noise1 * 0.25) / (1.0 + 0.5 - 0.25));
    //float density = rescale(0.7, 2.0, seed + rescale(0.5, 1.0, noise0));
*/
    return density;

    //seed *= step(1500.0, H) * step(H, 2300.0);

    //return rescale(0.5, 0.51, seed) * 0.05;
}

void CalculateLowerClouds(inout vec3 color, inout float alpha, in vec3 direction, in vec3 L, vec2 tracingPlanet, float viewLength, bool isSky, vec2 dither) {
    vec3 intScattering = vec3(0.0);
    vec3 intTransmittance = vec3(1.0);

    vec2 tracingBottom = RaySphereIntersection(E, direction, vec3(0.0), planet_radius + lower_clouds_height);
    vec2 tracingTop = RaySphereIntersection(E, direction, vec3(0.0), planet_radius + lower_clouds_height + lower_clouds_thickness);

    float bottom = tracingBottom.x > 0.0 ? tracingBottom.x : max(0.0, tracingBottom.y);
    float top = tracingTop.x > 0.0 ? tracingTop.x : max(0.0, tracingTop.y);

    float mu = dot(lightVectorWorld, direction);
    float phaseDual = mix(HG(mu, 0.6), HG(mu, 0.99 - silver_spread) * silver_intensity, 0.7);

    int steps = 12;
    float invsteps = 1.0 / float(steps);

    float stepLength = abs(top - bottom) * invsteps * 0.99;

    vec3 transmittance = vec3(1.0);
    vec3 scattering = vec3(0.0);

    float current = min(top, bottom) + stepLength * dither.x;

    vec3 beta = vec3(0.12) * mix(vec3(1.0, 0.782, 0.344), vec3(1.0), vec3(0.7));

    for(int i = 0; i < steps; i++) {
        vec3 p = direction * current;

        float H = max(0.0, length(vec3(0.0, planet_radius + e.y, 0.0) + p) - planet_radius);
        float height = (H - lower_clouds_height) / lower_clouds_thickness;

        if(tracingPlanet.x > 0.0 && tracingPlanet.x < current) break;
        if(viewLength * Altitude_Scale < current && !isSky) break;

        float density = CalculateLowerCloudsSamples(p);

        vec3 sigma_t = mix(lower_clouds_scattering_bottom, lower_clouds_scattering_top, vec3(pow(height, lower_clouds_scattering_remap)));

        vec3 transmittance = exp(-sigma_t * stepLength * density);

        vec3 lightExtinction = vec3(0.0);
        vec2 tracingLight = RaySphereIntersection(E + p, lightVectorWorld, vec3(0.0), planet_radius);

        vec2 tracingLightBottom = RaySphereIntersection(E + p, lightVectorWorld, vec3(0.0), planet_radius + lower_clouds_height);
        vec2 tracingLightTop = RaySphereIntersection(E + p, lightVectorWorld, vec3(0.0), planet_radius + lower_clouds_height + lower_clouds_thickness);

        if(density > 0.0 && tracingLight.x <= 0.0) {
            lightExtinction = vec3(1.0);

            float stepLengthLight = tracingLightBottom.x > 0.0 ? tracingLightBottom.x : max(0.0, tracingLightTop.y) / 6.0 * 0.99;
            vec3 lightSampleposition = p + stepLengthLight * lightVectorWorld * dither.y;

            vec3 density_along_light_ray = vec3(0.0);

            for(int j = 0; j < 6; j++) {
                float mediaSample = CalculateLowerCloudsSamples(lightSampleposition);

                float H = max(0.0, length(vec3(0.0, planet_radius + e.y, 0.0) + lightSampleposition) - planet_radius);
                float height = (H - lower_clouds_height) / lower_clouds_thickness;

                vec3 sigma_t = mix(lower_clouds_scattering_bottom, lower_clouds_scattering_top, vec3(pow(height, lower_clouds_scattering_remap)));

                density_along_light_ray += sigma_t * (mediaSample * stepLengthLight);

                lightSampleposition += stepLengthLight * lightVectorWorld;
            }

            lightExtinction *= (exp(-density_along_light_ray) + exp(-density_along_light_ray * 0.25) * 0.7 + exp(-density_along_light_ray * 0.0625) * 0.05) / (1.0 + 0.7 + 0.05);
            lightExtinction *= max(vec3(1.0 / 4.0), pow(1.0 - exp(-density_along_light_ray * 2.0), vec3(0.5)));
        }

        vec3 luminance = (  lightExtinction * sunLightingColorRaw * fading * phaseDual
                          /*+ sunLightingColorRaw * remap(height, 0.2, 0.3, 0.00005, 0.0)
                          + ambientLighting * remap(height, 0.1, 0.3, 0.0, 0.02)*/) * density;

        intScattering += (luminance - luminance * transmittance) * intTransmittance / sigma_t / max(0.1, density);
        intTransmittance *= transmittance;

        current += stepLength;
    }

    color *= intTransmittance;
    color += intScattering;

    alpha *= sum3(intTransmittance);
}

void main() {
    float notopaque = step(0.9, texture2D(gcolor, texcoord).a);

    float tileMaterial  = round(texture2D(gnormal, texcoord).z * 255.0);
    bool isSky = bool(step(254.5, tileMaterial));

    float depth0 = texture(depthtex0, texcoord).x;

    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth0) * 2.0 - 1.0));
    vec4 wP = (gbufferModelViewInverse) * nvec4(vP);
    vec3 viewDirection = normalize(vP);
    vec3 eyeDirection = -viewDirection;
    float viewLength = length(vP);

    vec3 color = vec3(0.0);

    vec3 direction = oct_to_float32x3(texcoord * 2.0 - 1.0).xzy;
    vec2 tracingPlanet = RaySphereIntersection(E, direction, vec3(0.0), planet_radius);
    vec2 tracingAtmospheric = RaySphereIntersection(E, direction, vec3(0.0), atmosphere_radius);
    vec2 farAtmosphericStart = RaySphereIntersection(E, direction, vec3(0.0), planet_radius + lower_clouds_height);

    vec2 offset = vec2(0.0);

    float dither = 0.5;//GetBlueNoise(depthtex2, (texcoord) * resolution, offset);
    float dither2 = 0.5;//GetBlueNoise(depthtex2, (1.0 - texcoord) * resolution, offset);

    vec3 n = abs(direction);
    vec3 coord3 = //n.x > max(n.y, n.z) ? direction.yzx :
                  //n.y > max(n.x, n.z) ? direction.zxy : 
                  direction;
    //vec3 coord3 = vec3(1.0, 0.0, 0.0) * direction.x + vec3(0.0, 0.0, 1.0) * direction.y + vec3(0.0, 1.0, 0.0) * direction.z;

    float alpha = 1.0;

    CalculateLowerClouds(color, alpha, direction, lightVectorWorld, tracingPlanet, 0.0, true, vec2(dither, dither2));

    //color += scattering * beta * stepLength * phaseDual * sunLightingColorRaw;

    //color = beta * sunLightingColorRaw;

    //color = vec3(0.0);

    //color = encodeGamma(color);

    gl_FragData[0] = vec4(color, alpha);
}
/* RENDERTARGETS: 15 */
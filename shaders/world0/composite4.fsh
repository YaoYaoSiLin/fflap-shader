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
uniform sampler2D shadowtex0;

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

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

uniform float frameTimeCounter;

#include "/libs/common.inc"
#include "/libs/dither.glsl"
#include "/libs/lighting.glsl"
#include "/libs/noise_common.glsl"
#include "/libs/volumetric/atmospheric_common.glsl"
#include "/libs/volumetric/clouds_common.glsl"

vec3 CalculateGeometryNormal(in vec3 worldNormal) {
	vec3 n = abs(worldNormal);
	return n.x > max(n.y, n.z) ? vec3(step(0.0, worldNormal.x) * 2.0 - 1.0, 0.0, 0.0) : 
		   n.y > max(n.x, n.z) ? vec3(0.0, step(0.0, worldNormal.y) * 2.0 - 1.0, 0.0) : 
		   vec3(0.0, 0.0, step(0.0, worldNormal.z) * 2.0 - 1.0);
}

vec3 CalculateLowDetailNormal(in vec3 worldNormal, float level) {
    vec3 n = abs(worldNormal);
    return floor(n / maxComponent(n) * (level) + 1e-5) / level * vec3(step(0.0, worldNormal.x) * 2.0 - 1.0, step(0.0, worldNormal.y) * 2.0 - 1.0, step(0.0, worldNormal.z) * 2.0 - 1.0);
}

#ifdef Enabled_Volumetric_Clouds
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

void CalculateLowerClouds(inout vec3 color, in vec3 direction, in vec3 L, vec2 tracingPlanet, float viewLength, bool isSky, vec2 dither) {
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

        float H = length(vec3(0.0, planet_radius + e.y, 0.0) + p) - planet_radius;
        float height = max(H - lower_clouds_height, 0.0) / lower_clouds_thickness;

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

                float H = length(vec3(0.0, planet_radius + e.y, 0.0) + lightSampleposition) - planet_radius;
                float height = max(H - lower_clouds_height, 0.0) / lower_clouds_thickness;

                vec3 sigma_t = mix(lower_clouds_scattering_bottom, lower_clouds_scattering_top, vec3(pow(height, lower_clouds_scattering_remap)));

                density_along_light_ray += sigma_t * (mediaSample * stepLengthLight);

                lightSampleposition += stepLengthLight * lightVectorWorld;
            }

            lightExtinction *= (exp(-density_along_light_ray) + exp(-density_along_light_ray * 0.25) * 0.7 + exp(-density_along_light_ray * 0.0625) * 0.05) / (1.0 + 0.7 + 0.05);
            lightExtinction *= max(vec3(1.0 / 8.0), pow(1.0 - exp(-density_along_light_ray * 2.0), vec3(0.5)));
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
}
#endif
#include "/libs/volumetric/atmospheric_raymarching.glsl"

vec4 ResolverRSMGI(in vec2 coord, vec2 scale) {
    vec4 result = vec4(0.0);

    vec3 diffuse = vec3(0.0);
    float total = 0.0;

    float depth0 = linearizeDepth(texture(depthtex0, texcoord).x);

    vec3 centerColor = decodeGamma(texture2D(gdepth, coord).rgb);

    coord *= scale;

    for(float i = -2.0; i <= 2.0; i += 1.0) {
    //    for(float j = -2.0; j <= 2.0; j += 1.0) {
        vec2 offset = vec2(i, 0.0);
        vec2 position = min(coord + offset * pixel, scale - pixel);

        float sample_depth = (texture2D(gaux1, position).a);
        float sample_linear_depth = linearizeDepth(sample_depth);
        float depthWeight = saturate(1.0 - abs(depth0 - sample_linear_depth) / depth0 * 40.0);

        vec3 sampleColor = decodeGamma(texture2D(gdepth, position).rgb);
        float luminanceWeight = pow(1.0 - abs(maxComponent(centerColor) - maxComponent(sampleColor)), 2.0);

        float weight = depthWeight * luminanceWeight;

        result += vec4(sampleColor, 1.0) * weight;
    //}
    }

    if(result.a <= 0.0) {
        result = vec4(decodeGamma(texture2D(gdepth, coord).rgb), 1.0);
    }

/*
    for(float i = -2.0; i <= 2.0; i += 1.0) {
    //    for(float j = -2.0; j <= 2.0; j += 1.0) {  
            vec2 offset = vec2(i, 0.0);
            vec2 position = min(coord + offset * pixel, vec2(0.5 - pixel));
            float sample_depth = (texture2D(gaux1, position).a);
            float sample_linear_depth = linearizeDepth(sample_depth);
            //float sdepth = linearizeDepth(texture(depthtex0, coord * 2.0).x);
            float weight = saturate(1.0 - abs(depth0 - sample_linear_depth) / depth0 * 40.0);

            result += vec4(decodeGamma(texture2D(gnormal, position).rgb), sample_depth) * weight;
            total += weight;
    //    }
    }

    if(total > 0.0) {
        result /= total;
    } else {
        result = vec4(decodeGamma(texture2D(gnormal, coord).rgb), texture2D(gaux1, coord).a);
    }   
*/

    return result / result.a;
}

vec2 IntersectCube(in vec3 shapeCenter, in vec3 direction, in vec3 size) {
    vec3 dr = 1.0 / direction;
    vec3 n = shapeCenter * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    if(far > near && far > 0.0) {
        return vec2(near, far);
    }else{
        return vec2(-1.0);
    }
}

vec2 IntersectCube(in vec3 shapeCenter, in vec3 direction, in vec3 size, inout vec3 normal){
    vec3 dr = 1.0 / direction;
    vec3 n = shapeCenter * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    vec3 front = -sign(direction) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    vec3 back = -sign(direction) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);

    normal = front;

    if(far > near && far > 0.0) {
        return vec2(near, far);
    }else{
        return vec2(-1.0);
    }
}

vec3 Diffusion(in float depth, in vec3 t) {
    depth = max(1e-5, depth);

    return exp(-depth * t) / (4.0 * Pi * t * max(1.0, depth));

    //r += 1e-5;
    //return (exp(-r * d) + exp(-r * d * 0.3333)) / (8.0 * Pi * d * max(1.0, r));
}

float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}

float sdBox( vec3 p, vec3 b ) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec3 ResolverColor(in vec2 coord, vec2 scale, vec2 offset) {
    float total = 0.0;
    vec3 resultColor = vec3(0.0);

    vec2 texelCoord = coord * scale + offset;

    float depth = linearizeDepth(texture(depthtex0, coord).x * 2.0 - 1.0);

    for(float i = -2.0; i <= 2.0; i += 1.0) {
        //for(float j = -1.0; j <= 1.0; j += 1.0) {  
            vec2 texelPosition = clamp(texelCoord + vec2(i, 0.0) * pixel, vec2(0.0) + pixel + offset, vec2(0.5, 1.0) - pixel + offset);

            vec4 sampleColor = (texture2D(gdepth, texelPosition));
                 sampleColor.a *= 255.0;

            float sampleDepth = linearizeDepth(texture2D(gaux1, texelPosition).a * 2.0 - 1.0);
            float depth_weight = saturate(1.0 - abs(depth - sampleDepth) / sampleDepth * 40.0);

            //float mask  = round(texture2D(gnormal, fract(texelPosition / scale)).z * 255.0);

            float weight = depth_weight * sampleColor.a;

            total += weight;
            resultColor += decodeGamma(sampleColor.rgb) * weight;
        //}
    }

    if(total > 0.0){
        resultColor /= total;
    }else{
        resultColor = decodeGamma(texture2D(gdepth, texelCoord).rgb);
    }

    //resultColor = encodeGamma(resultColor);

    return resultColor;
}

void main() {
    float notopaque = step(0.9, texture2D(gcolor, texcoord).a);

    vec3 albedo = decodeGamma(texture2D(gcolor, texcoord).rgb);

    vec3 texturedNormal = normalDecode(texture2D(composite, texcoord).xy);

    vec2 specularPackge = unpack2x8(texture2D(composite, texcoord).b);

    float roughness = pow2(1.0 - specularPackge.x);
    float metallic  = specularPackge.y;
    float material  = floor(texture2D(composite, texcoord).a * 255.0);
    float metal = step(isMetallic, metallic);
    vec3 F0 = mix(vec3(max(0.02, metallic)), albedo.rgb, metal);

    float depth0 = texture(depthtex0, texcoord).x;

    float tileMaterial  = round(texture2D(gnormal, texcoord).z * 255.0);
    bool isSky = linearizeDepth(depth0 * 2.0 - 1.0) > 0.9999;

    vec2 coord = texcoord;

    #ifdef Enabled_TAA
    coord -= jitter;
    #endif

    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, depth0) * 2.0 - 1.0));
    vec4 wP = (gbufferModelViewInverse) * nvec4(vP);
    vec3 viewDirection = normalize(vP);
    vec3 eyeDirection = -viewDirection;
    float viewLength = length(vP);

    vec3 color = decodeGamma(texture2D(gaux2, texcoord).rgb);

    vec3 direction = normalize(wP.xyz);
    vec2 tracingPlanet = RaySphereIntersection(E, direction, vec3(0.0), planet_radius);
    vec2 tracingAtmospheric = RaySphereIntersection(E, direction, vec3(0.0), atmosphere_radius);
    vec2 farAtmosphericStart = RaySphereIntersection(E, direction, vec3(0.0), planet_radius + lower_clouds_height);

    vec2 offset = jitter;

    float dither = GetBlueNoise(depthtex2, (texcoord) * resolution, offset);
    float dither2 = GetBlueNoise(depthtex2, (1.0 - texcoord) * resolution, offset);

    vec3 n = abs(direction);
    vec3 coord3 = //n.x > max(n.y, n.z) ? direction.yzx :
                  //n.y > max(n.x, n.z) ? direction.zxy : 
                  direction;
    //vec3 coord3 = vec3(1.0, 0.0, 0.0) * direction.x + vec3(0.0, 0.0, 1.0) * direction.y + vec3(0.0, 1.0, 0.0) * direction.z;

    #ifdef Enabled_Volumetric_Clouds
    CalculateLowerClouds(color, direction, lightVectorWorld, tracingPlanet, viewLength, isSky, vec2(dither, dither2));
    #endif

    vec3 atmosphere_color = vec3(0.0);
    vec3 atmosphere_transmittance = vec3(1.0);

    float middle_start = 0.0;
    float middle_end = tracingPlanet.x > 0.0 ? tracingPlanet.x : max(0.0, farAtmosphericStart.y);
    
    if(farAtmosphericStart.x > 0.0 || farAtmosphericStart.y < 0.0) {
        middle_start = max(0.0, tracingAtmospheric.x);
        middle_end = tracingPlanet.x > 0.0 ? tracingPlanet.x : max(0.0, tracingAtmospheric.y);
        middle_end = middle_start + (middle_end - middle_start) * 0.5;
    }

    if(isSky) {
        CalculateAtmospheric(atmosphere_transmittance, atmosphere_color, E, direction, sunVectorWorld, middle_start, middle_end, vec3(0.0, 0.0, dither));
        color = color * atmosphere_transmittance + atmosphere_color;
    }else{
        vec3 bounce = ResolverColor(texcoord, vec2(0.5, 1.0), vec2(0.0));
             bounce = -bounce / (bounce - 1.0);
        vec3 diffuse = bounce * albedo * sunLightingColorRaw;
             diffuse *= fading * invPi * (1.0 - metallic) * (1.0 - metal) * (1.0 - notopaque);

        color += diffuse;

        vec3 rayDirection = normalize(reflect(viewDirection, texturedNormal));// (smoothness > 0.7 ? visibleNormal : geometryNormal)
        vec3 fr = SpecularLighting(vec4(albedo, 1.0), rayDirection, eyeDirection, texturedNormal, texturedNormal, F0, roughness, metallic, (material < 64.5 ? 0.0 : material), true);

        vec4 centerSample = pow(texture2D(gdepth, texcoord * vec2(0.5, 1.0) + vec2(0.5, 0.0)), vec4(vec3(2.2), 1.0)) * vec4(vec3(1.0), 255.0);

        vec3 specular = centerSample.a > 100.0 ? centerSample.rgb : ResolverColor(texcoord, vec2(0.5, 1.0), vec2(0.5, 0.0));
             specular = -specular / (specular - 1.0);

        color += specular * fr;

        //color = bounce;
    }

        //color = step(linearizeDepth(depth0 * 2.0 - 1.0), 0.9999) * vec3(1.0);
/*
    float cosTheta = dither2;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float r = dither * 2.0 * Pi;
    vec3 d = vec3(cos(r) * sinTheta, sin(r) * sinTheta, cosTheta);
         d.z = 1.0 - sqrt(d.x * d.x + d.y * d.y);

    color = vec3(saturate(dot(normalize(direction + d), lightVectorWorld)));
*/

    vec3 cubeNormal = vec3(0.0);

    vec2 t = RaySphereIntersection(cameraPosition, direction, vec3(-4.5, 79.5, 54.5), 0.5);
    //vec2 t = IntersectCube(cameraPosition - vec3(-4.5, 79.5, 54.5), direction, vec3(0.5), cubeNormal);
    /*
    if(t.y > 0.0 && min(t.y, t.x) < viewLength) {
        //color = vec3(1.0, 0.0, 0.0);
        vec3 position = direction * (t.x > 0.0 ? t.x : max(0.0, t.y));
        //vec3 normal = cubeNormal;
        vec3 normal = position + cameraPosition - vec3(-4.5, 79.5, 54.5);

        vec2 tLight = RaySphereIntersection(position + cameraPosition, lightVectorWorld, vec3(-4.5, 79.5, 54.5), 0.5);
        //vec2 tLight = IntersectCube(position + cameraPosition - vec3(-4.5, 79.5, 54.5), lightVectorWorld, vec3(0.5));

        float ndotl = dot(lightVectorWorld, normal);

        float lDepth = max(0.0, tLight.y);

        vec3 albedo     = vec3(0.333);
        vec3 sigma_s    = vec3(1.0) / vec3(1.0, 0.782, 0.344);
        vec3 sigma_a    = vec3(0.0);
        vec3 sigma_e    = sigma_s + sigma_a;

        vec3 diffuse = albedo * max(0.0, ndotl) * invPi;

        color = Diffusion(lDepth, sigma_e) * albedo;
        color += diffuse;

        //color = exp(-(sigma_a + sigma_s) * lDepth) * albedo * sigma_s;// * HG(dot(direction, lightVectorWorld), 0.9);
        //color = albedo * saturate(ndotl);

        //color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(pow5(ndotl * 0.5 + 0.5))) * vec3(invPi);
        //color = (exp(-d * 1.0) * exp(-d * 1.0 * 0.333)) / (8.0 * Pi * d * 1.0);
    }
    */
    color = encodeGamma(color);

    gl_FragData[0] = vec4(color, 1.0);
}
/* DRAWBUFFERS:5 */
const int clouds_steps          = 12;
const int clouds_light_steps    = 6;

////////////////

#ifndef INCLUDE_CLOUDS_COMMON
#include "/libs/volumetric/clouds_common.glsl"
#endif

vec3 e = vec3(cameraPosition.x * Altitude_Scale, max(0.0, cameraPosition.y - 63.0) * Altitude_Scale, cameraPosition.z * Altitude_Scale);
vec3 E = vec3(0.0, rE + 1.0 + e.y, 0.0);

////////////////

//move to shape
float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
    return dot(point - origin, normal) / dot(direction, normal);
}

//move to material
vec3 BeersLaw(in vec3 beta, in float d){
     return exp(-beta * d);
}

vec3 PowderEffect(in vec3 beta, in float d){
     return 1.0 - exp(-beta * d * 2.0);
}

vec3 BeersPowder(in vec3 beta, in float d) {
     return exp(-beta * d) * (1.0 - exp(-beta * d * 2.0));
}

////////////////

float CalculateCloudsLightDepth(in vec3 rayPosition, in float dither, in bool notparallax){
    float invsteps = 1.0 / float(clouds_light_steps);

    //float overClouds = rayPosition.y > clouds_bottom + clouds_thickness ? -1.0 : 1.0;

    vec2 traceingSun = notparallax ? vec2(intersectPlane(rayPosition + E, lightVectorWorld, vec3(0.0, rE + clouds_bottom + clouds_thickness, 0.0), vec3(0.0, -1.0, 0.0)), 0.0) : RaySphereIntersection(rayPosition + E, lightVectorWorld, vec3(0.0), rE + clouds_bottom + clouds_thickness);

    float lightStepLength = traceingSun.x > 0.0 ? traceingSun.x : traceingSun.y; //do not forget intersection plane when not parallax clouds
    //float lightStepLength = mix(traceingSun.y, traceingSun.x, step(0.0, traceingSun.x));
    float stepLength = lightStepLength * invsteps;

    //float stepheight = ((clouds_bottom + clouds_thickness) - h) * invsteps;

    if(lightStepLength <= 20.0) return 0.0;
     
    vec3 rayStep = lightVectorWorld * stepLength;
    vec3 lightSamplePosition = rayPosition + rayStep * dither + e;

    float lightDepth = 0.0;

    //lightDepth = CalculateCloudsNoise(lightSamplePosition * clouds_scale, lightSamplePosition.y) * dither * stepLength;

    for(int i = 0; i < clouds_light_steps; i++){
        vec3 samplePosition = lightSamplePosition;
        vec3 samplePosition1 = samplePosition * clouds_scale;

        float H = notparallax ? samplePosition.y : max(1.0, length(samplePosition + vec3(0.0, rE, 0.0)) - rE);

        float density = CalculateCloudsNoise(samplePosition1, H);

        lightDepth += density * stepLength;

        lightSamplePosition += rayStep;
    }

    return lightDepth;
}

vec4 CalculateClouds(in vec3 origin, in vec3 L, in float frontLength, vec2 dither, inout float cloudsLength, inout float viewDepth, inout float cloudsDistance){
    //const float steps = 12;
    float invsteps = 1.0 / float(clouds_steps);

    float opticalDepth = 0.0;
    vec3 scattering = vec3(0.0);

    vec3 direction = normalize(origin);

    float mu = dot(direction, L);
    float phaseBack = HG(0.9999, 0.6) * invPi;
    float phaseFront = HG(mu, 0.99);
    float phase = (phaseFront + phaseBack);
     
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, 0.76);

    float phaseDual = mix(HG(mu, 0.2), HG(mu, 0.99), 0.2) + HG(0.9999, 0.4) * 0.1;

    //vec2 offset = jitter + frameTimeCounter * pixel / 64.0 * 32.0;
    //offset = vec2(0.0);

    //vec2 dither = vec2(GetBlueNoise(depthtex2, (texcoord) * resolution * Clouds_Render_Scale, offset),
    //                    GetBlueNoise(depthtex2, (1.0 - texcoord) * resolution * Clouds_Render_Scale, offset));
    //dither = vec2(0.0); 

    vec2 traceingPlanet = RaySphereIntersection(E, direction, vec3(0.0), rE);

    vec2 traceingStart = RaySphereIntersection(E, direction, vec3(0.0), rE + clouds_bottom);
    vec2 traceingEnd = RaySphereIntersection(E, direction, vec3(0.0), rE + clouds_bottom + clouds_thickness);

    //float bottomLength = traceingStart.x > 0.0 ? traceingStart.x : traceingStart.y;
    //float topLength = traceingEnd.x > 0.0 ? traceingEnd.x : traceingEnd.y;
     
    float bottomLength = mix(traceingStart.y, traceingStart.x, step(0.0, traceingStart.x));
    float topLength = mix(traceingEnd.y, traceingEnd.x, step(0.0, traceingEnd.x));
    float startLength = min(bottomLength, topLength);

    float height = startLength == bottomLength ? clouds_bottom : clouds_bottom + clouds_thickness;
    float stepheight = clouds_thickness * invsteps * (height == clouds_bottom ? 1.0 : -1.0);

    bool notparallax = false;

    //cloudsLength = traceingPlanet.x > 0.0 ? traceingPlanet.x : 100000.0;

    //for better quallity when eyes in clouds
    //but less rendering distance
     if(clamp(e.y, clouds_bottom, clouds_bottom + clouds_thickness) == e.y) {
        //return vec4(0.0);

        topLength = 2000.0;      //end point
        bottomLength = 128.0;     //start point

        //stop planet detection
        traceingPlanet.x = 0.0;
        startLength = bottomLength;

        //
        notparallax = true;
    }

    if(traceingPlanet.x * step(bottomLength, topLength) > 0.0 || (bottomLength) < 0.0) return vec4(vec3(0.0), 1.0);

    float stepLength = abs(topLength - bottomLength) * invsteps;

    vec3 rayPosition = direction * (startLength + stepLength * dither.x);
    vec3 rayStep = direction * stepLength;

    float totalDensity = 0.0;

    float hitLength = cloudsDistance;
    float endLength = 0.0;

    //stepLength += dither.x * stepLength;

    vec3 intScattTrans = vec3(1.0);

    for(int i = 0; i < clouds_steps; i++){
        float rayLength = length(rayPosition);

        if(frontLength < rayLength && frontLength > 0.0) break;
          
        vec3 samplePosition = rayPosition + e;
        float H = notparallax ? samplePosition.y : max(1.0, length(samplePosition + vec3(0.0, rE, 0.0)) - rE);

        if(clamp(H, clouds_bottom, clouds_thickness + clouds_bottom) != H && notparallax) break;

        vec3 samplePosition1 = samplePosition * clouds_scale;

        vec3 Tr = bR * exp(-H / Hr * 0.5);
        vec3 Tm = bM * exp(-H / Hm * 0.5);

        float density = CalculateCloudsNoise(samplePosition1, H);
        float lightDepth = CalculateCloudsLightDepth(rayPosition, dither.y, notparallax);

        if(density > 0.0) {
            vec3 extinction_light = BeersLaw(clouds_beta, lightDepth) * (1.0 - exp(-clouds_beta * density * stepLength * 4.0));
                 extinction_light += BeersLaw(clouds_beta * 0.33 * vec3(1.0, 0.844, 0.586), lightDepth) * 0.5;
                 extinction_light += BeersLaw(clouds_beta * 0.11 * vec3(1.0, 0.844, 0.586), lightDepth) * 0.25;
                 extinction_light /= 1.75;
                 //extinction_light += (1.0 - exp(-clouds_beta * density * stepLength * 0.01));
                 extinction_light *= PowderEffect(clouds_beta, max(8.0, lightDepth));

            vec3 sunlight = (extinction_light * sunLightingColorRaw) * (fading * phaseDual);

            vec3 extinction_atmospheric = exp(-(Tr + Tm) * rayLength * 0.5);
            vec3 step_m = (Tm * extinction_atmospheric);
            vec3 step_r = (Tr * extinction_atmospheric);
            vec3 atmospheric = (step_m + step_r) * sum3(sunLightingColorRaw) / (Tr + Tm) * 0.5 * exp(-clouds_beta * lightDepth);

            vec3 transmittance = exp(-clouds_beta * density * stepLength);

            vec3 luminance = sunlight;
                 luminance = (luminance - luminance * transmittance) / clouds_beta;
                 //luminance += atmospheric;

            scattering += intScattTrans * (luminance) * density;

            intScattTrans *= transmittance;
        }

        cloudsDistance = min(rayLength, cloudsDistance);

        hitLength = min(hitLength, rayLength);
        endLength = max(endLength, rayLength);

        opticalDepth += density * stepLength;

        rayPosition += rayStep;
    }

    cloudsLength = startLength;

    float cloudsMiddleLength = (hitLength + endLength) * 0.5;

    viewDepth = nvec3(gbufferProjection * nvec4(mat3(gbufferModelView) * (cloudsMiddleLength * direction))).z * 0.5 + 0.5;

    float transmittance = saturate(exp(-max(clouds_beta.r, max(clouds_beta.g, clouds_beta.b)) * opticalDepth));

    return vec4(scattering, transmittance);
}

////////////////

#ifndef INCLUDE_CLOUDS_RAYMARCHING
#define INCLUDE_CLOUDS_RAYMARCHING
#endif
#define Sun_Light_Quality 1
#define Sky_Light_Quality 1

////////////////

vec3 SimpleLightingColor(in vec3 origin, in vec3 D, in vec3 L, in float density) {
    vec2 tracingPlanet = RaySphereIntersection(E, L, vec3(0.0), rE);
    vec2 tracingAtmospheric = RaySphereIntersection(E, L, vec3(0.0), rA);

    float stepLength = max(0.0, tracingAtmospheric.y);
    float height = max(1e-6, length(E + L * stepLength) - rE);

    vec3 Tr = rayleigh_extinction * exp(-height / rayleigh_distribution * 0.5);
    vec3 Tm = mie_extinction * exp(-height / mie_distribution * 0.5);

    return exp(-(Tr + Tm) * stepLength * density);
}

vec3 SimpleLightingColor(in vec3 origin, in vec3 D, in vec3 L, in float density, in float g) {
    vec2 tracingPlanet = RaySphereIntersection(E, L, vec3(0.0), rE);
    vec2 tracingAtmospheric = RaySphereIntersection(E, L, vec3(0.0), rA);

    float stepLength = max(0.0, tracingAtmospheric.y);
    float height = max(1e-6, length(E + L * stepLength) - rE);

    vec3 Tr = rayleigh_extinction * exp(-height / rayleigh_distribution * 0.5);
    vec3 Tm = mie_extinction * exp(-height / mie_distribution * 0.5);

    float mu = 0.9999;
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, g);

    return exp(-(Tr + Tm) * stepLength * density) * phase_mie * 2.0;
}

vec3 LightingColorSolidPlanet(in vec3 origin, in vec3 direction, in vec3 L, in float density, in float g) {
    vec2 tracingPlanet = RaySphereIntersection(E, L, vec3(0.0), rE);
    vec2 tracingAtmospheric = RaySphereIntersection(E, L, vec3(0.0), rA);

    float stepLength = max(0.0, tracingAtmospheric.y);
    float height = max(1e-6, length(E + L * stepLength) - rE);

    vec3 Tr = rayleigh_extinction * exp(-height / rayleigh_distribution * 0.5);
    vec3 Tm = mie_extinction * exp(-height / mie_distribution * 0.5);

    float mu = 0.9999;
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, g);

    return exp(-(Tr + Tm) * stepLength * density) * step(tracingPlanet.x, 0.0);// * phase_mie * 2.0;
}

vec3 CalculateSunLightColor(in vec3 origin, in vec3 direction, in vec3 L, in float density, in float g){
    return LightingColorSolidPlanet(origin, direction, L, density, g);
}

////////////////

vec3 CalculateSkyLightColor(in vec3 origin, in vec3 direction, in vec3 L, in vec2 density, in float g){
    vec2 tracingA = RaySphereIntersection(origin, direction, vec3(0.0), atmosphere_radius);
    vec2 tracingE = RaySphereIntersection(origin, direction, vec3(0.0), planet_radius);

    float tmin = max(0.0, tracingA.x);
    float tmax = tracingE.x > 0.0 ? tracingE.x : tracingA.y;
    
    int steps = 6;
    float inv_steps = 1.0 / float(steps);

    float stepLength = (tmax - tmin) * inv_steps;
    tmin += stepLength;

    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);

    float opticalDepthR = 0.0;
    float opticalDepthM = 0.0; 

    float mu = dot(direction, L);

    for(int i = 0; i < steps; i++){
        float sampleLength = stepLength * float(i + 1);
        vec3 position = direction * sampleLength + origin;
        float height = length(position) - rE;

        float density_mie = exp(-height / Hm);
        float density_rayleigh = exp(-height / Hr);

        opticalDepthR += stepLength * density_rayleigh;
        opticalDepthM += stepLength * density_mie;

        vec3 attenuation = exp(-(opticalDepthR * rayleigh_extinction + opticalDepthM * mie_extinction));

        #if Sky_Light_Quality == 1
        attenuation *= SimpleLightingColor(position, L, L, density.x) * SunLight + SimpleLightingColor(position, -L, -L, density.y) * MoonLight;
        #endif

        r += attenuation * density_rayleigh;
        m += attenuation * density_mie;
    }

    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, g);

    return (r * rayleigh_scattering * phase_rayleigh + m * mie_scattering * phase_mie) * stepLength;
}
#if 0
vec3 CalculateAtmospheric(in vec3 origin, in vec3 direction, in vec3 L, in vec3 density, in float g){
    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);

    float mu = dot(direction, L);
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu - 1e-5, g);

    //vec3 e = vec3(cameraPosition.x * 1.0, max(1.0, (cameraPosition.y - 63.0) * 1.0), cameraPosition.y * 1.0);
    //vec3 E = vec3(0.0, rE + e.y, 0.0);

    vec2 tracingPlanet = RaySphereIntersection(origin, direction, vec3(0.0), rE);
    vec2 traceingAtmospheric = RaySphereIntersection(origin, direction, vec3(0.0), rA);
    float tmin = max(traceingAtmospheric.x, 0.0);
    float tmax = tracingPlanet.x > 0.0 ? tracingPlanet.x : traceingAtmospheric.y;

    //const int steps = 12;
    //const int light_steps = 6;
    float invsteps = 1.0 / float(far_atmospheric_sample_steps);
    float invLsteps = 1.0 / float(far_atmospheric_light_sample_steps);

    float stepLength = (tmax - tmin) * invsteps;

    vec3 Rtransmittance = vec3(0.0);
    vec3 Mtransmittance = vec3(0.0);
    vec3 Otransmittance = vec3(0.0);

    for(int i = 0; i < far_atmospheric_sample_steps; i++) {
        vec3 samplePosition = direction * (tmin + stepLength * float(1 + i)) + origin;
        float height = max(1.0, length(samplePosition) - rE);

        vec3 LRtransmittance = vec3(0.0);
        vec3 LMtransmittance = vec3(0.0);
        vec3 LOtransmittance = vec3(0.0);

        vec2 traceingSun = RaySphereIntersection(samplePosition, L, vec3(0.0), rA);
        float lightStepLength = traceingSun.y * invLsteps;

        for(int j = 0; j < far_atmospheric_light_sample_steps; j++){
            vec3 lightSamplePosition = samplePosition + L * lightStepLength * float(1 + j);
            float height = max(1.0, length(lightSamplePosition) - rE);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = exp(-height / ozone_distribution);

            LRtransmittance += lightStepLength * density_rayleigh;
            LMtransmittance += lightStepLength * density_mie;
            LOtransmittance += lightStepLength * density_ozone/* * step(32000.0, height)*/;
        }

        float density_rayleigh = exp(-height / rayleigh_distribution);
        float density_mie = exp(-height / mie_distribution);
        float density_ozone = exp(-height / ozone_distribution);

        Rtransmittance += stepLength * density_rayleigh;
        Mtransmittance += stepLength * density_mie;
        Otransmittance += stepLength * density_ozone/* * step(32000.0, height)*/;

        vec3 tau = rayleigh_extinction * (Rtransmittance + LRtransmittance) * density.xxx + mie_extinction * (Mtransmittance + LMtransmittance) * density.yyy + (LOtransmittance + Otransmittance) * ozone_absorption * density.zzz;
        vec3 attenuation = exp(-tau);

        r += (attenuation) * density_rayleigh;
        m += (attenuation) * density_mie;
    }

    return (r * rayleigh_scattering * phase_rayleigh + m * mie_scattering * phase_mie) * stepLength * 4.0;
}
#endif
////////////////

#ifndef LIGHTING_COLOR
#define LIGHTING_COLOR
#endif
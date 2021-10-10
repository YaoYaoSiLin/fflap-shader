#ifndef INCLUDE_ATMOSPHERIC_COMMON
//#include "/libs/volumetric/atmospheric_raymarching.glsl"
#endif

float CloudsOcclusion(in vec3 origin, in vec3 position, in vec3 L, float dither){
    #if defined(High_Quality_Clouds) && defined(Enabled_Volumetric_Clouds)
    vec2 traceingBottom = RaySphereIntersection(position - vec3(0.0, 1.0, 0.0), L, vec3(0.0), planet_radius + lower_clouds_height);
    vec2 traceingTop = RaySphereIntersection(position - vec3(0.0, 1.0, 0.0), L, vec3(0.0), planet_radius + lower_clouds_height + lower_clouds_thickness);
    
    float bottom = traceingBottom.x > 0.0 ? traceingBottom.x : max(0.0, traceingBottom.y);
    float top = traceingTop.x > 0.0 ? traceingTop.x : max(0.0, traceingTop.y);
    if(bottom < 1.0) return 1.0;

    vec3 p = position - origin + L * bottom * dither;

    vec2 cloudsCoord = float32x3_to_oct(p) * 0.5 + 0.5;
         cloudsCoord *= 0.5;

    float clouds_alpha = texture2D(colortex15, cloudsCoord).a;
          clouds_alpha = pow(clouds_alpha, 0.125);

    return clouds_alpha;
    #else
    return 1.0;
    #endif
}

void CalculateAtmospheric(inout vec3 color, inout vec3 atmosphere_color, in vec3 origin, in vec3 direction, in vec3 L, in float tmin, in float tmax, vec3 dither){
    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);

    float mu = dot(direction, L);
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, 0.76);
    float phase_rayleigh2 = (3.0 / 16.0 / Pi) * (1.0 + -mu * -mu);
    float phase_mie2 = HG(-mu, 0.76);

    const int steps = 8;
    const int light_steps = 6;
    float invsteps = 1.0 / float(steps);
    float invLsteps = 1.0 / float(light_steps);

    float stepLength = (tmax - tmin) * invsteps;

    vec3 Rtransmittance = vec3(0.0);
    vec3 Mtransmittance = vec3(0.0);
    vec3 Otransmittance = vec3(0.0);

    for(int i = 0; i < steps; i++) {
        vec3 samplePosition = direction * (tmin + stepLength * (0.5 + float(i))) + origin;
        float height = max(1.0, length(samplePosition) - planet_radius);

        vec3 LRtransmittance = vec3(0.0);
        vec3 LMtransmittance = vec3(0.0);
        vec3 LOtransmittance = vec3(0.0);

        vec2 traceingSun = RaySphereIntersection(samplePosition, L, vec3(0.0), atmosphere_radius);
        float lightStepLength = traceingSun.y * invLsteps;

        for(int j = 0; j < light_steps; j++){
            vec3 lightSamplePosition = samplePosition + L * lightStepLength * (0.5 + float(j));
            float height = max(1.0, length(lightSamplePosition) - planet_radius);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

            LRtransmittance += lightStepLength * density_rayleigh;
            LMtransmittance += lightStepLength * density_mie;
            LOtransmittance += lightStepLength * density_ozone;
        }

        float density_rayleigh = exp(-height / rayleigh_distribution);
        float density_mie = exp(-height / mie_distribution);
        float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

        Rtransmittance += stepLength * density_rayleigh;
        Mtransmittance += stepLength * density_mie;
        Otransmittance += stepLength * density_ozone;

        vec3 tau = rayleigh_extinction * (Rtransmittance + LRtransmittance) + mie_extinction * (Mtransmittance + LMtransmittance)  + (LOtransmittance + Otransmittance) * ozone_absorption;
        vec3 attenuation = exp(-tau) * SunLight;

        attenuation *= CloudsOcclusion(origin, samplePosition, L, 1.0 - dither.z);

        vec2 traceingMoon = RaySphereIntersection(samplePosition, -L, vec3(0.0), atmosphere_radius);

        float lightStepLength2 = traceingMoon.y * invLsteps;

        vec3 LRtransmittance2 = vec3(0.0);
        vec3 LMtransmittance2 = vec3(0.0);
        vec3 LOtransmittance2 = vec3(0.0);

        for(int jj = 0; jj < light_steps; jj++){
            vec3 lightSamplePosition = samplePosition + -L * lightStepLength2 * (0.5 + float(jj));
            float height = max(1.0, length(lightSamplePosition) - planet_radius);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

            LRtransmittance2 += lightStepLength2 * density_rayleigh;
            LMtransmittance2 += lightStepLength2 * density_mie;
            LOtransmittance2 += lightStepLength2 * density_ozone;
        }

        vec3 tau2 = rayleigh_extinction * (Rtransmittance + LRtransmittance2) + mie_extinction * (Mtransmittance + LMtransmittance2)  + (LOtransmittance2 + Otransmittance) * ozone_absorption;
        vec3 attenuation2 = exp(-tau2) * MoonLight;

        attenuation2 *= CloudsOcclusion(origin, samplePosition, -L, 1.0 - dither.z);

        r += (attenuation * phase_rayleigh + attenuation2 * phase_rayleigh2) * (density_rayleigh);
        m += (attenuation * phase_mie + attenuation2 * phase_mie2) * (density_mie);

        //Rtransmittance += stepLength * density_rayleigh;
        //Mtransmittance += stepLength * density_mie;
    }

    color *= exp(-Rtransmittance * rayleigh_extinction - Mtransmittance * mie_extinction);
    atmosphere_color += r * rayleigh_scattering * stepLength;
    atmosphere_color += m * mie_scattering * stepLength;

    //return (r * bR * phase_rayleigh + m * bM * phase_mie) * stepLength * 4.0;
}

#if 0
void CalculateFarAtmospheric(inout vec3 color, inout vec3 atmosphere_color, in vec3 origin, in vec3 direction, in vec3 L, in float tmin, in float tmax){
    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);

    float mu = dot(direction, L);
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, 0.76);
    float phase_rayleigh2 = (3.0 / 16.0 / Pi) * (1.0 + -mu * -mu);
    float phase_mie2 = HG(-mu, 0.76);

    //vec3 e = vec3(cameraPosition.x * 1.0, max(1.0, (cameraPosition.y - 63.0) * 1.0), cameraPosition.y * 1.0);
    //vec3 E = vec3(0.0, rE + e.y, 0.0);

    //const int steps = 12;
    //const int light_steps = 6;
    float invsteps = 1.0 / float(far_atmospheric_sample_steps);
    float invLsteps = 1.0 / float(far_atmospheric_light_sample_steps);

    float stepLength = (tmax - tmin) * invsteps;
    tmin = stepLength;

    vec3 Rtransmittance = vec3(0.0);
    vec3 Mtransmittance = vec3(0.0);
    vec3 Otransmittance = vec3(0.0);

    for(int i = 0; i < far_atmospheric_sample_steps; i++) {
        vec3 samplePosition = direction * (tmin + stepLength * float(i)) + origin;
        float height = max(1.0, length(samplePosition) - rE);

        vec3 LRtransmittance = vec3(0.0);
        vec3 LMtransmittance = vec3(0.0);
        vec3 LOtransmittance = vec3(0.0);

        vec2 traceingSun = RaySphereIntersection(samplePosition, L, vec3(0.0), rA);
        float lightStepLength = traceingSun.y * invLsteps;

        for(int j = 0; j < far_atmospheric_light_sample_steps; j++){
            vec3 lightSamplePosition = samplePosition + L * lightStepLength * float(j);
            float height = max(1.0, length(lightSamplePosition) - rE);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

            LRtransmittance += lightStepLength * density_rayleigh;
            LMtransmittance += lightStepLength * density_mie;
            LOtransmittance += lightStepLength * density_ozone;
        }

        float density_rayleigh = exp(-height / rayleigh_distribution);
        float density_mie = exp(-height / mie_distribution);
        float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

        Rtransmittance += stepLength * density_rayleigh;
        Mtransmittance += stepLength * density_mie;
        Otransmittance += stepLength * density_ozone;

        vec3 tau = rayleigh_extinction * (Rtransmittance + LRtransmittance) + mie_extinction * (Mtransmittance + LMtransmittance)  + (LOtransmittance + Otransmittance) * ozone_absorption;
        vec3 attenuation = exp(-tau) * SunLight;

        //attenuation *= CloudsOcclusion(origin, samplePosition, L, 1.0);

        vec2 traceingMoon = RaySphereIntersection(samplePosition, -L, vec3(0.0), rA);
        float lightStepLength2 = traceingMoon.y * invLsteps;

        vec3 LRtransmittance2 = vec3(0.0);
        vec3 LMtransmittance2 = vec3(0.0);
        vec3 LOtransmittance2 = vec3(0.0);

        for(int jj = 0; jj < far_atmospheric_light_sample_steps; jj++){
            vec3 lightSamplePosition = samplePosition + -L * lightStepLength2 * float(jj);
            float height = max(1.0, length(lightSamplePosition) - rE);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

            LRtransmittance2 += lightStepLength2 * density_rayleigh;
            LMtransmittance2 += lightStepLength2 * density_mie;
            LOtransmittance2 += lightStepLength2 * density_ozone;
        }

        vec3 tau2 = rayleigh_extinction * (Rtransmittance + LRtransmittance2) + mie_extinction * (Mtransmittance + LMtransmittance2)  + (LOtransmittance2 + Otransmittance) * ozone_absorption;
        vec3 attenuation2 = exp(-tau2) * MoonLight;

        r += (attenuation * phase_rayleigh + attenuation2 * phase_rayleigh2) * (density_rayleigh);
        m += (attenuation * phase_mie + attenuation2 * phase_mie2) * (density_mie);

        //Rtransmittance += stepLength * density_rayleigh;
        //Mtransmittance += stepLength * density_mie;
    }

    color *= exp(-Rtransmittance * rayleigh_extinction - Mtransmittance * mie_extinction);
    atmosphere_color += r * rayleigh_scattering * stepLength;
    atmosphere_color += m * mie_scattering * stepLength;

    //return (r * bR * phase_rayleigh + m * bM * phase_mie) * stepLength * 4.0;
}

void CalculateClosestAtmospheric(inout vec3 color, inout vec3 atmosphere_color, in vec3 origin, in vec3 direction, in vec3 L, float tmin, float tmax){
    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);

    float mu = dot(direction, L);
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, 0.76);
    float phase_rayleigh2 = (3.0 / 16.0 / Pi) * (1.0 + -mu * -mu);
    float phase_mie2 = HG(-mu, 0.76);

    vec2 offset = jitter + frameTimeCounter * pixel / 64.0 * 32.0;
    float dither = GetBlueNoise(depthtex2, (texcoord) * resolution * 0.5, offset);
    dither = mix(1.0, dither, 0.5);

    //vec3 e = vec3(cameraPosition.x * 1.0, max(1.0, (cameraPosition.y - 63.0) * 1.0), cameraPosition.y * 1.0);
    //vec3 E = vec3(0.0, rE + e.y, 0.0);

    //vec2 tracingPlanet = RaySphereIntersection(E, direction, vec3(0.0), rE);
    //vec2 traceingAtmospheric = RaySphereIntersection(E, direction, vec3(0.0), rA);

    //float tmin = max(0.0, traceingAtmospheric.x);
    //float tmax = (tracingPlanet.x > 0.0 ? tracingPlanet.x : traceingAtmospheric.y);

    //tmax = (tmax - tmin) / far_atmospheric_sample_steps;

    //const int steps = 8;
    //const int light_steps = 6;
    float invsteps = 1.0 / float(middle_atmospheric_sample_steps);
    float invLsteps = 1.0 / float(middle_atmospheric_light_sample_steps);

    float stepLength = (tmax - tmin) * invsteps;
    //tmin = stepLength;
    
    vec3 Rtransmittance = vec3(0.0);
    vec3 Mtransmittance = vec3(0.0);
    vec3 Otransmittance = vec3(0.0);

    vec3 transmittance = vec3(0.0);

    for(int i = 0; i < middle_atmospheric_sample_steps; i++) {
        vec3 samplePosition = direction * (tmin + stepLength * float(i)) + origin;
        float height = max(1.0, length(samplePosition) - rE);

        vec3 LRtransmittance = vec3(0.0);
        vec3 LMtransmittance = vec3(0.0);
        vec3 LOtransmittance = vec3(0.0);

        vec2 traceingSun = RaySphereIntersection(samplePosition, L, vec3(0.0), rA);
        float lightStepLength = traceingSun.y * invLsteps;

        for(int j = 0; j < middle_atmospheric_light_sample_steps; j++){
            vec3 lightSamplePosition = samplePosition + L * lightStepLength * float(j);
            float height = max(1.0, length(lightSamplePosition) - rE);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

            LRtransmittance += lightStepLength * density_rayleigh;
            LMtransmittance += lightStepLength * density_mie;
            LOtransmittance += lightStepLength * density_ozone;
        }

        float density_rayleigh = exp(-height / rayleigh_distribution);
        float density_mie = exp(-height / mie_distribution);
        float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

        Rtransmittance += stepLength * density_rayleigh;
        Mtransmittance += stepLength * density_mie;
        Otransmittance += stepLength * density_ozone;

        vec3 tau = rayleigh_extinction * (Rtransmittance + LRtransmittance) + mie_extinction * (Mtransmittance + LMtransmittance)  + (LOtransmittance + Otransmittance) * ozone_absorption;
        vec3 attenuation = exp(-tau) * SunLight;

        vec2 traceingMoon = RaySphereIntersection(samplePosition, -L, vec3(0.0), rA);
        float lightStepLength2 = traceingMoon.y * invLsteps;

        vec3 LRtransmittance2 = vec3(0.0);
        vec3 LMtransmittance2 = vec3(0.0);
        vec3 LOtransmittance2 = vec3(0.0);

        for(int jj = 0; jj < far_atmospheric_light_sample_steps; jj++){
            vec3 lightSamplePosition = samplePosition + -L * lightStepLength2 * float(jj);
            float height = max(1.0, length(lightSamplePosition) - rE);

            float density_rayleigh = exp(-height / rayleigh_distribution);
            float density_mie = exp(-height / mie_distribution);
            float density_ozone = max(0.0, 1.0 - abs(height - 25000.0) / 15000.0);

            LRtransmittance2 += lightStepLength2 * density_rayleigh;
            LMtransmittance2 += lightStepLength2 * density_mie;
            LOtransmittance2 += lightStepLength2 * density_ozone;
        }

        vec3 tau2 = rayleigh_extinction * (Rtransmittance + LRtransmittance2) + mie_extinction * (Mtransmittance + LMtransmittance2)  + (LOtransmittance2 + Otransmittance) * ozone_absorption;
        vec3 attenuation2 = exp(-tau2) * MoonLight;

        r += (attenuation * phase_rayleigh + attenuation2 * phase_rayleigh2) * (density_rayleigh * stepLength);
        m += (attenuation * phase_mie + attenuation2 * phase_mie2) * (density_mie * stepLength);
    }

    //color = vec3(0.0);

    color *= exp(-(rayleigh_extinction * Rtransmittance + mie_extinction * Mtransmittance));
    atmosphere_color = r * rayleigh_scattering + m * mie_scattering;

    //return (r * bR * phase_rayleigh + m * bM * phase_mie) * stepLength * 4.0;
}
#endif
vec3 SimplePlanetSurface(in vec3 lightColor, in vec3 origin, in vec3 direction, in vec3 L, float tmin) {
    if(tmin < 0.0) return vec3(0.0);

    float height = 1.0;

    vec3 Tr = rayleigh_extinction * exp(-height / rayleigh_distribution);
    vec3 Te = mie_extinction * exp(-height / mie_distribution);
    vec3 To = vec3(0.0);

    vec3 transmittance = exp(-(Tr + Te) * tmin * 100.0);

    float mu = 0.99;
    float phase_rayleigh = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phase_mie = HG(mu, 0.9);

    return lightColor * transmittance + lightColor * (1.0 - transmittance) * phase_mie;
}

////////////////

#ifndef INCLUDE_ATMOSPHHERIC_RAYMARCHING
#define INCLUDE_ATMOSPHHERIC_RAYMARCHING
#endif
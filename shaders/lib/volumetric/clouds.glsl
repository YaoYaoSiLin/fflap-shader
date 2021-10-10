/*
float GetNearCloudsDensity(in vec3 P){

}
*/

uniform sampler2D colortex8;
uniform sampler2D colortex9;

float noise(in vec2 x){
  return texture(noisetex, fract(x / 64.0)).x;
}

float noise(in vec3 x) {
    x = x.xzy;

    vec3 i = floor(x);
    vec3 f = fract(x);

	f = f*f*(3.0-2.0*f);

	vec2 uv = (i.xy + i.z * vec2(17.0)) + f.xy;
    uv += 0.5;

	vec2 rg = vec2(noise(uv), noise(uv+17.0));

	return mix(rg.x, rg.y, f.z);
}

float voronoiLD(in vec2 p){
    return 1.0 - texture(depthtex2, p / 128.0).g;
}

float voronoi3(in vec2 p){
    return dot(vec3(1.0, 0.5, 0.25) / 1.75, texture2D(colortex8, p / 128.0).rgb);
    //return dot(vec3(1.0, 0.5, 0.25) / 1.75, vec3(texture2D(colortex8, p / 128.0).r, texture2D(colortex8, p / 128.0 * 2.0).r, texture2D(colortex8, p / 128.0 * 4.0).r));
}

float voronoi3LD(in vec2 p){
    return dot(vec3(1.0, 0.5, 0.25) / 1.75, texture2D(colortex9, p / 128.0).rgb);
    //return dot(vec3(1.0, 0.5, 0.25) / 1.75, vec3(texture2D(colortex9, p / 128.0).r, texture2D(colortex9, p / 128.0 * 2.0).r, texture2D(colortex9, p / 128.0 * 4.0).r));
}

vec2 switch2(in vec2 x){
    return x.x < x.y ? x : x.yx;
}

float CalculateCumulusShape(in vec3 P, in float t){
    float density = 0.0;

    vec2 t0 = vec2(t, 0.0) * 0.1 * 1.0;
    vec2 p0 = P.xz * 0.005 + t0;
    float shape = voronoi3(p0);//(voronoi(p0) + voronoi(p0 * 2.0) * 0.5 + voronoiLD(p0 * 4.0) * 0.25) / 1.75;

    vec3 t1 = vec3(t, -t, 0.0) * 0.2 * 1.0;
    vec3 p1 = P * 0.02;
    float noise1 = (noise(p1 + t1) + noise(p1 * 2.0 + t1 * 2.0) * 0.5) / 1.5;

    vec3 t2 = vec3(t, t, 0.0) * 0.06 * 1.0;
    vec3 p2 = P * 0.004;
    float noise2 = (noise(p2 + t2) + noise(p2 * 2.0 + t2 * 2.0) * 0.5) / 1.5;

    density = saturate(dot(vec3(1.0, 0.25, -0.05) / (1.0 + 0.25 + -0.05), vec3(shape, noise2, noise1)));//saturate((shape + noise2 * 0.25) / 1.25 - noise1 * 0.05) / (1.0 - 0.05);

    float H = max(1.0, length(P) - rE);

    float distanceToCenter = distance(H, 1500.0 + 400.0);
    float distanceToBottom = distance(H, 1500.0);
    density *= exp(-distanceToBottom / 3000.0); 
    density = rescale(1.0 - 0.3, 1.0, density); 
    density *= (1.0 - min(1.0, distanceToCenter / 400.0));    

    return saturate(density * 3.0);
}

float CalculateCumulusShapeSimple(in vec3 P, in float t){
    vec2 t0 = vec2(t, 0.0) * 0.1 * 1.0;
    vec2 p0 = P.xz * 0.005 + t0;
    float shape = voronoi3LD(p0);//(voronoiLD(p0) + voronoiLD(p0 * 2.0)) / 1.5;

    vec3 t2 = vec3(t, t, 0.0) * 0.05 * 1.0;
    vec3 p2 = P * 0.004;
    float noise2 = (noise(p2 + t2));

    float density = (shape + noise2 * 0.25) / 1.25;

    float Hl = length(P) - rE;

    float distanceToCenter = distance(Hl, 1500.0 + 400.0);
    float distanceToBottom = distance(Hl, 1500.0);
    density *= exp(-distanceToBottom / 3000.0); 
    density = rescale(1.0 - 0.3, 1.0, density); 
    density *= (1.0 - min(1.0, distanceToCenter / 400.0));

    return saturate(density * 3.0);
}

vec4 CalculateNearVolumetricClouds(in vec3 rayOrigin, in vec3 L, bool sky, in vec2 offset){
    offset += frameTimeCounter * pixel.y * 20.0;

    float dither = GetBlueNoise(depthtex2, (texcoord) * 0.25, resolution.y, offset);
    float dither2 = GetBlueNoise(depthtex2, (1.0 - texcoord) * 0.25, resolution.y, offset);

    vec3 e = vec3(cameraPosition.xz * Altitude_Scale, max(1.0, (eyeAltitude - 63.0) * Altitude_Scale)).xzy;
    vec3 E = vec3(0.0, e.y + rE, 0.0);

    float bottom = 1500.0;
    float top = bottom + 800.0;

    vec3 rayDirection = normalize(rayOrigin);
    float viewLength = length(rayOrigin) * float(!sky);

    vec2 tracingTop = RaySphereIntersection(E, rayDirection, vec3(0.0), rE + top); //tracingTop = tracingTop.x >= 0.0 ? switch2(tracingTop) : tracingTop;
    vec2 tracingBottom = RaySphereIntersection(E, rayDirection, vec3(0.0), rE + bottom); //tracingBottom = tracingBottom.x >= 0.0 ? switch2(tracingBottom) : tracingBottom;
    vec2 tracingEarth = RaySphereIntersection(E, rayDirection, vec3(0.0), rE);
    vec2 tracingAtmosphere = RaySphereIntersection(E, rayDirection, vec3(0.0), rA);

    vec4 hit = tracingTop.y > tracingBottom.y ? vec4(tracingBottom, tracingTop) : vec4(tracingTop, tracingBottom);
    if(hit.y < 0.0) return vec4(0.0);
    if(hit.x <= 0.0 && tracingEarth.x >= 0.0) return vec4(0.0);

    int steps = 12;
    float invsteps = 1.0 / float(steps);
    int lightsteps = 12;
    float invlightsteps = 1.0 / float(lightsteps);

    float start = (tracingBottom.x >= 0.0 ? min(tracingBottom.y, tracingBottom.x) : tracingBottom.y);
    float end = (tracingTop.x >= 0.0 ? min(tracingTop.y, tracingTop.x) : tracingTop.y);
    vec2 package = vec2(start, end);

    start = start < end ? package.x : package.y;
    end = start < end ? package.y : package.x;

    float stepLength = (end - start) * invsteps;

    //float ttop = (tracingTop.x >= 0.0 ? min(tracingTop.y, tracingTop.x) : tracingTop.y);
    //float tbottom = (tracingBottom.x >= 0.0 ? min(tracingBottom.y, tracingBottom.x) : tracingBottom.y);

    //vec2 In = vec2(tracingTop.x, tracingBottom.x);
    //vec2 Out = vec2(tracingTop.y, tracingBottom.y);

    //float start = (tracingBottom.x >= 0.0 ? min(tracingBottom.y, tracingBottom.x) : tracingBottom.y);
    //float end = (tracingTop.x >= 0.0 ? min(tracingTop.y, tracingTop.x) : tracingTop.y);
    //float stepLength = (end - start) * invsteps;

    vec3 rayStep = rayDirection * stepLength;
    vec3 rayPosition = e + rayDirection * start + rayStep * dither + vec3(0.0, rE, 0.0);

    float t = frameTimeCounter * 0.0;

    vec3 clouds = vec3(0.0);
    vec3 atmospheric = vec3(0.0);
    float opticalDepth = 0.0;

    vec3 b = vec3(0.007) * vec3(1.0, 0.884, 0.586);

    float mu = dot(L, rayDirection);
    float phase = HG(mu, 0.8);
    float phaseR = (3.0 / 16.0 / Pi) * (1.0 + mu * mu);
    float phaseM = HG(mu, 0.76);

    float l = start;
    float smax = (tracingEarth.x >= 0.0 ? min(tracingAtmosphere.y, tracingEarth.x) : tracingAtmosphere.y) - max(0.0, tracingAtmosphere.x);

    const float size = 0.5;

    vec3 front = vec3(HG(1.0, 0.9));
    vec3 back = vec3(HG(-1.0, 0.0));

    vec2 p = rayDirection.xz * (start) * 0.0005 + vec2(t, 0.0);
    vec3 dist = (texture2D(noisetex, p).xyz + texture2D(noisetex, p * 2.0).xyz * 0.5) / 1.5 * 2.0 - 1.0;

    rayPosition += dist * stepLength * 0.1;

    for(int i = 0; i < steps; i++){
        if(l > viewLength * 1.0 && viewLength > 0.0) break;
        float s = min(smax, l);

        float density = CalculateCumulusShape(rayPosition, t);

        float H = max(1.0, length(rayPosition) - rE);
        
        float lightDepth = 0.0;
                    
        vec2 lightTracingTop = RaySphereIntersection(rayPosition, L, vec3(0.0), rE + top);
        float lightStepLength = lightTracingTop.y * invlightsteps;

        vec3 lightStep = L * lightStepLength;
        vec3 lightSamplePosition = rayPosition + lightStep * dither2;

        for(int j = 0; j < lightsteps; j++){
            float density = CalculateCumulusShapeSimple(lightSamplePosition, t);

            float Hl = length(lightSamplePosition) - rE;

            lightDepth += density * abs(lightStepLength);
            lightSamplePosition += lightStep;
        }  

        vec3 sunLightExtinction = exp(-b * lightDepth);
             sunLightExtinction *= clamp(exp(-vec3(sum3(b)) * lightDepth), back, front);
             //sunLightExtinction = mix(back, front, sunLightExtinction);
             sunLightExtinction *= clamp(1.0 - exp(-b * lightDepth * 2.0), vec3(0.3), vec3(1.0));
        
        vec3 cloudsVisibility = exp(-b * opticalDepth);

        //#ifdef Clouds_Atmospheric_Scatter
        vec3 Tr = bR * exp(-H / Hr);
        vec3 Tm = bM * exp(-H / Hm);

        vec3 atmosphericExtinction = exp(-(Tr + Tm) * s);
        vec3 atmosphericScatter = (Tr * phaseR + Tm * phaseM) * s;

        atmospheric += atmosphericScatter * density * cloudsVisibility;
        //#endif
        
        vec3 cloudsColor = sunLightExtinction * abs(stepLength) * atmosphericExtinction;

        clouds += cloudsColor * density * cloudsVisibility;

        opticalDepth += density * abs(stepLength) * sum3(atmosphericExtinction);
        rayPosition += rayStep;
        l += abs(stepLength);
    }

    float sunLightFading = saturate(abs(L.y) * 10.0);

    clouds *= sunLightingColorRaw * sum3(b) * fading * (invPi + phase);
    atmospheric *= sunLightingColorRaw * sunLightFading;

    return vec4(clouds, opticalDepth);
}
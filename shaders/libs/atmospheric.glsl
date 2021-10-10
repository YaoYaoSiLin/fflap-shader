#define SunLight 3.0		//
#define SkyLight 1.0		//
#define MoonLight 0.12		//

#define Sea_Level 63.0

#define Altitude_Scale 1.0		//[1.0 3.0 9.0 18.0 27.0 54.0 81.0 163.0 243.0 729.0 2187.0 6561.0]

#define Extended_Earth_Surface_Atmospheric_Density 30.0
#define Extended_Land_Atmospheric_Density 30.0

#ifndef INCLUDE_ATMOSPHERIC
#define INCLUDE_ATMOSPHERIC
#endif

const float rE = 6360e3;
const float rA = 6420e3;
const float Hr = 8000.0;
const float Hm = 1200.0;

const vec3 bM = vec3(2e-6);
const vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6);

/*
const float rE = 6360e3;
const float rA = 6420e3;
const float Hr = 7994;
const float Hm = 1200;

const vec3 bM = vec3(21e-6);
const vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6);
*/
float HG(in float m, in float g){
  return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}

vec2 RaySphereIntersection(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float sphereRadius) {
	rayOrigin -= sphereCenter;

	float a = dot(rayDir, rayDir);
	float b = 2.0 * dot(rayOrigin, rayDir);
	float c = dot(rayOrigin, rayOrigin) - (sphereRadius * sphereRadius);
	float d = b * b - 4 * a * c;

	if (d < 0) return vec2(-1.0);

	d = sqrt(d);
	return vec2(-b - d, -b + d) / (2 * a);
}

float density(in float l, in float h){
	return exp(-l / h);
}

vec2 OpticalDepthRM(in float h, in float l){
	return vec2(exp(-h / Hr),
							exp(-h / Hm)) * l;
}

vec2 OpticalDepthRM(in float h, in float l, vec2 density){
	return vec2(exp(-h / density.x),
							exp(-h / density.y)) * l;
}

vec3 DrawSunScattering(in vec3 m, in float mu, in float g){
	float phase = HG(mu, g);

	return m * phase * bM;
}

vec3 DrawSun(in vec3 L, in vec3 D, in vec3 E, float g, float density){
	vec2 tracingSun = RaySphereIntersection(E, L, vec3(0.0), rA);
	vec2 tracingPlanetA = RaySphereIntersection(E, D, vec3(0.0), rA);

	float sA = max(0.0, -tracingPlanetA.x);
	//if(bool(step(0.0, tL.x))) sA = min(sA, tL.x);

	float s = tracingSun.y - max(0.0, tracingSun.x);
	float h = max(1.0, length(E + D * sA) - rE);

	vec3 tM = bM * exp(-h / Hm / density);
	vec3 tR = bR * exp(-h / Hr / density);
	vec3 tE = tR + tM;

	return vec3(1.0) * saturate(exp(-tE * s)) * saturate(HG(dot(L, D), g));
}

//vec3 CalculateLocalScattering

void CalculateLocalInScattering(in float Dr, in float Dm, in float densityR, in float densityM, out vec3 r, out vec3 m){
	vec3 Tr = bR * Dr;
	vec3 Tm = bM * Dm;

	vec3 extinction = exp(-(Tr + Tm));

	r = extinction * densityR;
	m = extinction * densityM;
}

vec3 CalculateInScattering(in vec3 e, in vec3 d, in vec3 l, in float g, ivec2 steps, in vec3 intensityRMS){
	e.y *= Altitude_Scale;
	e.y = max(e.y, 1.0);
	e.y = e.y + rE;
	
	//float g = 0.76;
	float g2 = g*g;

	float mu = dot(d, l);
	float opmu2 = 1. + mu*mu;

	float phaseR = (3.0 / 16.0 / Pi) * opmu2;
	float phaseM = HG(mu, g);

	vec2 traceingEarth = RaySphereIntersection(vec3(0.0, e.y, 0.0), d, vec3(0.0), rE);
	float isEarth = step(0.0, traceingEarth.x);
	if(bool(isEarth)) {
		e.y = min(rA, e.y);
		traceingEarth = RaySphereIntersection(vec3(0.0, e.y, 0.0), d, vec3(0.0), rE);
	}

	vec2 t = RaySphereIntersection(e, d, vec3(0.0), rA);

	float tmin = max(0.0, t.x);
	float tmax = bool(isEarth) ? traceingEarth.x : t.y;//mix(t.y, traceingEarth.x, isEarth);

	float stepLength = (tmax - tmin); //if(t.x >= 0.0) stepLength = min(t.x, stepLength);
		  stepLength /= float(steps);

	//if(stepLength <= 0.0) return vec3(0.0);

	vec3 rayStep = d * stepLength;

	vec3 r = vec3(0.0);
  	vec3 m = vec3(0.0);

	vec2 opticalDepth;

	for(int i = 1; i <= steps.x; i++){
		vec3 p = e + rayStep * float(i);
		float rayLength = max(1.0, length(p) - rE);

		float hr = exp(-rayLength / Hr);
		float hm = exp(-rayLength / Hm);

		opticalDepth.x += (hr) * stepLength;
		opticalDepth.y += (hm) * stepLength;

		vec2 opticalDepthLight = vec2(0.0);

		vec2 t = RaySphereIntersection(p, l, vec3(0.0), rA);

		//if(t.x <= 0.0){
		float stepLengthL = (t.x > 0.0 ? t.x : t.y - max(0.0, t.x)) / float(steps.y);
		//float stepLengthL = (t.y - max(0.0, t.x)) / float(steps.y);
		//	  stepLengthL = t.x >= 0.0 ? min(stepLengthL, t.x / float(steps)) : stepLengthL;

		vec3 rayStepL = l * stepLengthL;
		
		for(int j = 1; j <= steps.y; j++){
			vec3 pL = p + rayStepL * float(j);

			float rayLengthL = max(1.0, length(pL) - rE);

			opticalDepthLight += OpticalDepthRM(rayLengthL, stepLengthL);
			//opticalDepthLight += vec2(exp(-rayLengthL / Hr), exp(-rayLengthL / Hm)) * stepLengthL;
		}	
		//}
		//vec3 extinction = exp(-(bR * (opticalDepth.x + opticalDepthLight.x) + bM * (opticalDepth.y + opticalDepthLight.y)));

		vec3 stepR = vec3(0.0);
		vec3 stepM = vec3(0.0);
		CalculateLocalInScattering(opticalDepth.x + opticalDepthLight.x, opticalDepth.y + opticalDepthLight.y, hr, hm, stepR, stepM);

		r += (stepR) * stepLength;
		m += (stepM) * stepLength;
	}

	float Rintensity = intensityRMS.x;
	float Mintensity = intensityRMS.y;
	float SunIntensity = intensityRMS.z;	

	vec3 sun = DrawSunScattering(m, mu, 0.999) * (1.0 - isEarth);

	return saturate(r * bR * phaseR * Rintensity + m * bM * phaseM * Mintensity + sun * SunIntensity) * 1.0;
}

vec3 Extinction(in float h, in float s){
	float Dr = exp(-h / Hr);
	float Dm = exp(-h / Hm);

	vec3 Tr = bR * Dr;
	vec3 Tm = bM * Dm;

	return exp(-(Tr + Tm) * s);
}

vec3 Extinction(in vec3 e, in vec3 d){
	e.y = max(1.0, e.y);
	e.y += rE;

	vec2 tE = RaySphereIntersection(e, d, vec3(0.0), rE);
	//if(bool(step(0.0, tE.x))) return vec3(0.0, 0.0, 0.0);

	vec2 t = RaySphereIntersection(e, d, vec3(0.0), rA);

	//if(earth) t.y = min(t.y, tE.x);
	//if(t.x > t.y) t = vec2(t.y, t.x);
	//t.x = max(0.0, t.x);

	float l = t.y;
	vec3 p = e + d * l;
	float h = clamp(p.y - rE, 1.0, 20000.0);

	return Extinction(h, l);
}

vec3 CalculateSimpleInScattering(in vec3 offset, in vec3 v, in vec3 l, in float g){
	float mu = dot(v, l);

	vec3 r = bR * (0.0596831) * (1.0 + mu*mu);
	vec3 m = bM * HG(mu, g);

	return Extinction(offset, l) * (r + m) / (bR + bM);
}

vec3 ApplyEarthSurface(in vec3 color, in vec3 e, in vec3 d, in vec3 l, vec3 lightColor, vec3 surfaceColor){
	e.y *= Altitude_Scale;
	e.y = max(e.y, 1.0);
	e.y += rE;

	vec2 traceingEarth = RaySphereIntersection(e, d, vec3(0.0), rE);
	//float isEarth = step(0.0, traceingEarth.x);

	if(traceingEarth.x < 0.0){
		return color;
	}

	e.y = min(rA, e.y);

	traceingEarth = RaySphereIntersection(e, d, vec3(0.0), rE);
	vec2 traceingAtmospheric = RaySphereIntersection(e, d, vec3(0.0), rA);

	float tmax = traceingEarth.x;
	float tmin = max(traceingAtmospheric.x, 0.0);
	float stepLength = tmax - tmin;

	float H = max(1.0, length(e + d * stepLength) - rE);

	float g = 0.76;
	float g2 = g * g;

	float mu = dot(d, l);
	float opmu2 = 1. + mu*mu;

	float phaseR = (3.0 / 16.0 / Pi) * opmu2;
	float phaseM = HG(mu, g);

	vec3 earthSurfaceColor = vec3(0.0);

	vec3 extinction = exp(-(bR * exp(-H / Hr) + bM * exp(-H / Hm)) * stepLength);
	vec3 surfaceR = bR * extinction * stepLength * phaseR * lightColor;
	vec3 surfaceM = bM * extinction * stepLength * phaseM * lightColor;

	//vec3 surfaceR = lightColor * (1.0 - exp(-stepLength * bR * Extended_Earth_Surface_Atmospheric_Density * exp(-H / Hr))) * phaseR;
	//vec3 surfaceM = lightColor * (1.0 - exp(-stepLength * bM * Extended_Earth_Surface_Atmospheric_Density * exp(-H / Hm))) * phaseM;

	earthSurfaceColor = surfaceColor * mix(0.1, 1.0, fading);
	earthSurfaceColor += surfaceR + surfaceM;
	earthSurfaceColor *= Extinction(H, stepLength);

	return color + earthSurfaceColor * invPi * 0.5;
}

vec3 InScattering(in vec3 e, in vec3 d, in vec3 l, in float mu, in float g){
	e.y = max(e.y, 1.0);
	e.y += rE;

	float g2 = g*g;

	//float mu = dot(d, -l);
	float opmu2 = 1. + mu*mu;

	float phaseR = .0596831 * opmu2;
	float phaseM = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	vec2 tA = RaySphereIntersection(e, d, vec3(0.0), rA);
	if(tA.x > tA.y) tA = vec2(tA.y, tA.x);
	float dist = tA.y;

	vec3 p = e + d * dist;
	vec2 opticalDepth = OpticalDepthRM(max(0.0, length(p) - rE), dist);

	vec2 tL = RaySphereIntersection(p, l, vec3(0.0), rA);
	if(tL.x > tL.y) tL = vec2(tL.y, tL.x);
	float distL = tL.y;

	vec3 pL = p + l * distL;
	vec2 opticalDepthL = OpticalDepthRM(max(0.0, length(pL) - rE), distL);

	vec3 attenuation = exp(-(bR * (opticalDepth.x + opticalDepthL.x) + bM * (opticalDepth.y + opticalDepthL.y)));

	vec3 r = bR * attenuation * phaseR;
	vec3 m = bM * attenuation * phaseM;

	vec3 inscattering = (r + m) / (bR + bM);
	vec3 sunLight = Extinction(e, l);
			 //sunLight *= max(RaySphereIntersection(e, d, vec3(0.0), rE).x, 0.0)

	return inscattering * sunLight;
}

vec3 InScattering(in vec3 e, in vec3 d, in vec3 l, in float h, in float s, in float g, in float mu){
	e.y = max(1.0, e.y);
	e.y += rE;
	//e = vec3(0.0, rE + h, 0.0);

	float g2 = g*g;

	float opmu2 = 1. + mu*mu;

	float phaseR = .0596831 * opmu2;
	float phaseM = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	vec3 scattering;

	vec2 opticalDepth = OpticalDepthRM(h, s);
	scattering = exp(-(bR * opticalDepth.x + bM * opticalDepth.y)) * (phaseR * bR + phaseM * bM) / (bR + bM);

	/*
	vec2 tE = RaySphereIntersection(e, d, vec3(0.0), rE);
	vec2 t = RaySphereIntersection(e, d, vec3(0.0), rA);

	if(tE.x > 0.0) t.y = min(t.y, tE.x);
	if(t.x > t.y)	t = vec2(t.y, t.x);

	vec3 p = e + d * t.y;
	float height = h;//max(0.0, length(p) - rE);

	vec2 tL = RaySphereIntersection(e, d, vec3(0.0), rA);
	if(tL.x > tL.y)	tL = vec2(tL.y, tL.x);

	vec3 pl = p + l * tL.y;
	float heightL = h;//max(0.0, length(pl) - rE);

	vec2 opticalDepth  = vec2(exp(-height / Hr),
													  exp(-height / Hm)) * s;
	vec2 opticalDepthL = vec2(exp(-heightL / Hr),
	 												  exp(-heightL / Hm)) * s;

	vec3 tau = bR * (opticalDepth.x + opticalDepthL.x) + bM * 1.1 * (opticalDepth.y + opticalDepthL.y);
	vec3 attenuation = exp(-tau);
	scattering = attenuation * bR * phaseR * s;
	scattering += attenuation * bM * phaseM * s;
	scattering *= 10.0;
	*/
	// * (bR * phaseR * opticalDepth.x + bM * phaseM * opticalDepth.y) / (bR + bM);
	//scattering = attenuation * bR * phaseR * opticalDepth.x;

	//vec3 Tr = bR * exp(-h / Hr);
	//vec3 Tm = bM * exp(-h / Hm);

	/*
	float ltop = exp(-length(l-d));

	d = e + d * rayLength;

	t = RaySphereIntersection(d, l, vec3(0.0), rA);
	if(t.x > t.y){
		t = vec2(t.y, t.x);
	}

	tmin = max(0.0, t.x);
	tmax = t.y;
	rayLength = (tmax-tmin);

	l = d + l * rayLength;

	float distanceToLight = max(0.0, length(l-d));

	float dR = exp(-h / Hr);
	float dM = exp(-h / Hm);

	scattering = (1.0 - exp(-s * (bR*dR+bM*dM))) * exp(-distanceToLight*(bR+bM)) * (bR*phaseR+bM*phaseM) / (bR+bM);
	//scattering *= exp(-(dR*bR+dM*bM) * s);
	*/

	return (scattering);
}

vec3 SimpleLightColor(in vec3 L, vec3 D, vec3 E, float density, float mu, float g) {
	E += vec3(0.0, rE, 0.0);

	if(RaySphereIntersection(E, L, vec3(0.0), rE).x > 0.0) return vec3(0.0);

    vec2 traceing = RaySphereIntersection(E, L, vec3(0.0), rA);

    float rayLength = traceing.x >= 0.0 ? min(traceing.y, traceing.x) : traceing.y;
    float sunAltitude = max(1.0, length(E + L * rayLength) - rE);

    vec3 Tr = bR * exp(-sunAltitude / Hr * 0.5);
    vec3 Tm = bM * exp(-sunAltitude / Hm * 0.5);

    return max(vec3(0.0), exp(-(Tr + Tm) * rayLength * 0.5 * density)) * saturate(HG(mu, g));
}
#define SunLight 1.0	//[2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0]
#define SkyLight 1.0	//

#ifndef AtmosphericScattering_Steps
	#define AtmosphericScattering_Steps 4
#endif

#ifndef AtmosphericScattering_Stepss
	#define AtmosphericScattering_Stepss 4
#endif

const float rE = 6360e3;
const float rA = 6420e3;
const float Hr = 7994;
const float Hm = 1200;

const vec3 bM = vec3(21e-6);
const vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6);

float escape(in vec3 p, in vec3 d, in float R, out vec2 t) {
	vec3 v = p;
	float b = dot(v, d);
	float c = dot(v, v) - R*R;
	float det2 = b * b - c;
	if (det2 < 0.) return -1.;
	float det = sqrt(det2);
	float t1 = -b - det, t2 = -b + det;
	t = vec2(t1, t2);
	return (t1 >= 0.) ? t1 : t2;
}

vec3 AtmosphericScattering(in vec3 o, in vec3 wP, in vec3 sP, in float mu, in float g, in bool night){
  vec3 cP = vec3(0.0, o.y + rE, 0.0);

	#ifndef Void_Sky
	if(wP.y < 0.0){
		//return vec3(0.0);
		//wP.y = 0.001 / (1.0 - wP.y);
	}
	#endif

	//float g = 0.76;
	//if(night) g = 0.96;
	//g = -g;

	float g2 = g*g;
	//float mu = clamp01(dot(wP, sP));
	float opmu2 = 1. + mu*mu;
	float phaseR = .0596831 * opmu2;
	float phaseM = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	//if(night) phaseM *= 0.03;
	//else phaseM *= 0.75;

	//wP *= max(1.0, sqrt(wP.y * 2.0 * Pi));

  int steps = int(AtmosphericScattering_Steps);
  int stepss = int(AtmosphericScattering_Stepss);

	float b = dot(cP, wP);
	float c = dot(cP, cP) - rE*rE;
	float det2 = b * b - c;

	bool ground = false;

	if(-b - sqrt(det2) > b && det2 > 0.0){
		//return vec3(1.0, 0.0, 0.0);
		//wP.y = length(wP.y) / sqrt(det2) * 2.0 * Pi;
		wP.y = 0.0;
		ground = true;
		//wP.y = sqrt(wP.y * wP.y / det2);
		//wP.y = abs(wP.y);
	}

	//vec2 t0;
	//float L0 = escape(cP, wP, rA + 0.5, t0);

	vec2 t;
	float L = escape(cP, wP, rA, t);
	//L = min(L, L0);

	if(L > 0.0){
  vec3 r = vec3(0.0);
  vec3 m = vec3(0.0);

  float opticalDepthR = 0.0;
  float opticalDepthM = 0.0;

	float len = L / 16.0;
	vec3 direction = wP * len;
	#if AtmosphericScattering_Steps < 8
	vec3 position = cP;
	#else
	vec3 position = cP + direction * 0.5;
	#endif

	float scale = 16.0 / steps - 1.0;

  for(int i = 0; i <= steps; i++){
    float h = length(position) - rE;

		if(ground){
			h = 1000.0;
		}

    float hr = exp(-h / Hr) * len;
    float hm = exp(-h / Hm) * len;

    opticalDepthR += hr;
    opticalDepthM += hm;

    float opticalDepthLightR = 0.0;
    float opticalDepthLightM = 0.0;

		vec2 tL;
		float Ls = escape(position, sP, rA, tL);

		float len2 = (Ls) / 16.0;
		vec3 direction2 = sP * len2;

		#if AtmosphericScattering_Stepss < 8
		vec3 position2 = position;
		#else
		vec3 position2 = position + direction2 * 0.5;
		#endif

		float scale2 = 16.0 / stepss - 1.0;

    if(Ls > 0.0){
      for(int j = 0; j <= stepss; j++){
        float hL = length(position2) - rE;

				if(ground){
					hL = 1000.0;
				}

        opticalDepthLightR += exp(-hL / Hr) * len2;
        opticalDepthLightM += exp(-hL / Hm) * len2;

				position2 += direction2 * (1.0 + float(j) / steps * scale2);
				//if(j == stepss - 1) position2 += direction2 * (15.0 - j);
      }
    }

    vec3 tau = bR * (opticalDepthR + opticalDepthLightR) + 1.1 * bM * (opticalDepthM + opticalDepthLightM);
    vec3 attenuation = exp(-tau);

    r += attenuation * hr;
    m += attenuation * hm;
		position += direction * (1.0 + float(i) / steps * scale);
		//if(i == steps - 1) position += direction * (15.0 - i);
  }

  return (r * bR * phaseR + m * bM * phaseM) * 20.0;
  }
}

vec2 RaySphereIntersection(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float sphereRadius) {
	rayOrigin -= sphereCenter;

	float a = dot(rayDir, rayDir);
	float b = 2.0 * dot(rayOrigin, rayDir);
	float c = dot(rayOrigin, rayOrigin) - (sphereRadius * sphereRadius);
	float d = b * b - 4 * a * c;

	if (d < 0)
	{
		return vec2(-1.0);
	}
	else
	{
		d = sqrt(d);
		return vec2(-b - d, -b + d) / (2 * a);
	}
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

vec3 DrawSun(in vec3 m, in float mu, in float g){
	float g2 = g*g;
	float phase = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	return m * phase * bM;
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
	e.y = max(e.y, 1.0);
	e.y = e.y + rE;

	//float g = 0.76;
	float g2 = g*g;

	float mu = dot(d, l);
	float opmu2 = 1. + mu*mu;

	float phaseR = .0596831 * opmu2;
	float phaseM = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	vec2 tE = RaySphereIntersection(e, d, vec3(0.0), rE);
	bool earth = bool(step(0.0, tE.x));

	vec2 t = RaySphereIntersection(e, d, vec3(0.0), rA);

	if(earth) t.y = min(t.y, tE.x);

	#if 0

	vec2 invsteps = 1.0 / vec2(steps);

	vec3 rayDirection = d * t.y;
	vec3 rayStep = rayDirection * invsteps.x;
	float stepSize = length(rayStep);

	vec3 r = vec3(0.0);
	vec3 m = vec3(0.0);

	vec2 lastDensity = vec2(exp(-1.0 / Hr), exp(-1.0 / Hm));
	vec3 lastLocalInScatteringR;
	vec3 lastLocalInScatteringM;
	Extinction(lastDensity, 1.0, 1.0, lastLocalInScatteringR, lastLocalInScatteringM);

	vec2 particleDensity = vec2(0.0);

	for(int i = 0; i < steps.x; ++i){
		vec3 p = e + rayStep * float(i);

		float h = max(0.0, length(p * vec3(0.0, 1.0, 0.0)) - rE);

		vec2 atmosphereDensity = vec2(exp(-h / Hr), exp(-h / Hm));
		particleDensity += (lastDensity + atmosphereDensity) * 0.5 * stepSize;
		lastDensity = atmosphereDensity;

		vec3 localInScatteringR;
		vec3 localInScatteringM;
		Extinction(lastDensity, particleDensity.x, particleDensity.y, localInScatteringR, localInScatteringM);

		r += atmosphereDensity.x * (localInScatteringR + lastLocalInScatteringR) * 0.5 * stepSize;
		m += atmosphereDensity.y * (localInScatteringM + lastLocalInScatteringM) * 0.5 * stepSize;
	}

	return r * phaseR + m * phaseM;

	#else
	if(t.x > t.y) t = vec2(t.y, t.x);

	float tmin = max(0.0, t.x);
	float tmax = t.y;

	float stepLength = (tmax) / float(steps.x);
	//if(earth){
	//	stepLength = max(0.0, tE.x) / float(steps.x);
	//	tmin = stepLength * 0.5;
	//}

	vec3 rayStep = d * stepLength;
	//e += d * (stepLength * 0.5 + tmin);
	e += rayStep;

	vec3 r = vec3(0.0);
  vec3 m = vec3(0.0);

	vec2 opticalDepth;

	for(int i = 0; i < steps.x; i++){
		vec3 p = e + rayStep * float(i);
		float rayLength = max(0.0, length(p) - rE);

		float hr = exp(-rayLength / Hr);
		float hm = exp(-rayLength / Hm);

		opticalDepth.x += (hr) * stepLength;
		opticalDepth.y += (hm) * stepLength;

		vec2 opticalDepthLight = vec2(0.0);

		vec2 t = RaySphereIntersection(p, l, vec3(0.0), rA);

		float stepLengthL = (t.y) / float(steps.y);
		vec3 rayStepL = l * stepLengthL;
		p += rayStepL;

		for(int j = 0; j < steps.y; j++){
			vec3 pL = p + rayStepL * float(j);

			float rayLengthL = max(0.0, length(pL) - rE);

			opticalDepthLight += OpticalDepthRM(rayLengthL, stepLengthL);
		}

		vec3 stepR;
		vec3 stepM;
		CalculateLocalInScattering(opticalDepth.x + opticalDepthLight.x, opticalDepth.y + opticalDepthLight.y, hr, hm, stepR, stepM);

		r += (stepR) * stepLength;
		m += (stepM) * stepLength;
	}

	float Rintensity = intensityRMS.x;
	float Mintensity = intensityRMS.y;
	float SunIntensity = intensityRMS.z;

	vec3 sun = DrawSun(m, mu, 0.994) * step(tE.x, 0.0) * SunIntensity;

	return (r * bR * phaseR * Rintensity + m * bM * phaseM * Mintensity + sun);
	#endif
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
	bool earth = bool(step(0.0, tE.x));
	//if(earth) return vec3(0.0, 0.0, 0.0);

	vec2 t = RaySphereIntersection(e, d, vec3(0.0), rA);

	//if(earth) t.y = min(t.y, tE.x);
	//if(t.x > t.y) t = vec2(t.y, t.x);
	//t.x = max(0.0, t.x);

	float l = t.y;
	vec3 p = e + d * l;
	float h = clamp(p.y - rE, 1.0, 20000.0);

	return Extinction(h, l);
}

vec3 ApplyEarthSurface(in vec3 color, in vec3 e, in vec3 d, in vec3 l){
	e.y = max(e.y, 1.0);
	e.y += rE;

	vec2 t = RaySphereIntersection(e, d, vec3(0.0), rE);
	float earth = step(0.0, t.x);

	float tmax = max(t.x, t.y);
	float tmin = max(min(t.x, t.y), 0.0);

	float g = 0.76;
	float g2 = g * g;

	float mu = dot(d, l);
	float opmu2 = 1. + mu*mu;

	float phaseR = .0596831 * opmu2;
	float phaseM = (0.25 / Pi) * ((1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	vec3 earthSurfaceColor = vec3(0.0);

	vec3 surfaceR = sunLightingColorRaw * (1.0 - exp(-tmin * bR * 10.0)) * phaseR;
	vec3 surfaceM = sunLightingColorRaw * (1.0 - exp(-tmin * bM * 10.0)) * phaseM;

	earthSurfaceColor = skyLightingColorRaw;
	earthSurfaceColor *= Extinction(e.y - rE, tmin);
	earthSurfaceColor += surfaceR + surfaceM;

	return color + earthSurfaceColor * earth;
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

vec3 CalculateSky(in vec3 viewVector, in vec3 lightVector, in float height, in float dither){
	viewVector = mat3(gbufferModelViewInverse) * viewVector;

	vec3 worldLightPosition = lightVector;
	bool night = worldLightPosition.y < -0.1;

	if(night) worldLightPosition = -worldLightPosition;

	vec3 scattering = AtmosphericScattering(vec3(0.0, height - 63.0, 0.0), viewVector, worldLightPosition, -dot(viewVector, worldLightPosition), -0.76, night);
	scattering = max(vec3(0.0), scattering) * 0.1;

	if(night) scattering *= 0.04 * clamp01((worldLightPosition.y - 0.1) * 3.0);
	vec3 nightSkyTransition = vec3(0.000167, 0.000277, 0.000413) * 0.3;
			 nightSkyTransition += pow5(1.0 - clamp01(viewVector.y)) * vec3(1.049, 0.582, 0.095) * 0.0004;
			 nightSkyTransition *= sqrt(clamp01(viewVector.y));

	scattering += max(vec3(0.0), nightSkyTransition - dot03(scattering) * 1.1);

	return L2Gamma(sqrt(scattering));
}

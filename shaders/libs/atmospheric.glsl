#define SunLight 4.0	//[2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0]
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

	if(-b - sqrt(det2) > b && det2 > 0.0){
		//return vec3(1.0, 0.0, 0.0);
		wP.y = length(wP.y) / sqrt(det2) * 2.0 * Pi;
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

  for(int i = 0; i < steps; i++){
    float h = length(position) - rE;

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
      for(int j = 0; j < stepss; j++){
        float hL = length(position2) - rE;

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

#define cameraAltitudeFixed -1	//[-1 0 1000 2000 3000 4000 5000 8000 9000 10000 12000 14000 16000 18000 20000 24000 28000 36000 48000]

const float rE = 6360e3;
const float rA = 6420e3;
const float Hr = 7994;
const float Hm = 1200;

const vec3 bM = vec3(21e-6);
const vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6);

float escape(in vec3 p, in vec3 d, in float R) {
	vec3 v = p;
	float b = dot(v, d);
	float c = dot(v, v) - R*R;
	float det2 = b * b - c;
	if (det2 < 0.) return -1.;
	float det = sqrt(det2);
	float t1 = -b - det, t2 = -b + det;
	return (t1 >= 0.) ? t1 : t2;
}

vec3 AtmosphericScattering(in vec3 o, in vec3 wP, in vec3 sP, in float dither){
  vec3 cP = vec3(0.0, o.y + rE, 0.0);

  const float Pi = 3.1415;

  float mu = min(dot(wP, sP), 1.0); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
  float phaseR = 3.f / (16.f * Pi) * (1 + mu * mu);
  float g = 0.76f;
  float phaseM = 3.f / (8.f * Pi) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

  int steps = 4;
  int stepss = 2;

  float u = escape(cP, wP, rA) / steps * 0.5;

	if(u > 0.0){
  vec3 r = vec3(0.0);
  vec3 m = vec3(0.0);

  float opticalDepthR = 0.0;
  float opticalDepthM = 0.0;

  for(int i = 0; i < steps; ++i){
    vec3 p = cP + wP * (u * dither);
    float h = length(p) - rE;

    float hr = exp(-h / Hr) * (u);
    float hm = exp(-h / Hm) * (u);

    opticalDepthR += hr;
    opticalDepthM += hm;

    float opticalDepthLightR = 0.0;
    float opticalDepthLightM = 0.0;

    float uS = escape(p, sP, rA) / stepss * 0.5;

    if(uS > 0.0){
      for(int j = 0; j < stepss; ++j){
        vec3 pns = p + sP * (uS * dither);
        float hL = length(pns) - rE;

        opticalDepthLightR += exp(-hL / Hr) * (uS);
        opticalDepthLightM += exp(-hL / Hm) * (uS);
      }
    }

    vec3 tau = bR * (opticalDepthR + opticalDepthLightR) + 1.1 * bM * (opticalDepthM + opticalDepthLightM);
    vec3 attenuation = exp(-tau);

    r += attenuation * hr;
    m += attenuation * hm;
  }

  return (r * bR * phaseR + m * bM * phaseM) * 20.0;
  }
}

vec3 CalculateSky(in vec3 P, in vec3 sP, in float H){
	return vec3(0.0);
}

vec3 CalculateSky(in vec3 viewVector, in vec3 lightVector, in float height, in float dither){
	vec3 scattering = AtmosphericScattering(vec3(0.0, 1000 + (height - 63.0), 0.0), mat3(gbufferModelViewInverse) * viewVector, lightVector, dither);
	scattering = max(scattering, vec3(0.0));

	return scattering * 0.16;
}
/*
float GetFogDensity(in float dist, in bool calcLighting){
	if(calcLighting){
		return clamp01((-cameraHight + 120.0) / 48.0);
	}
}
*/
vec3 CalculateFog(in vec3 color, in float d, in float cameraHight, in float worldHight, in bool isSky, in vec3 sunLightingColor, in vec3 skyLightingColor, in float fading){
	vec4 Wetness = vec4(0.0);
			 Wetness.rgb = (sunLightingColor * fading * 0.5 + skyLightingColor + 1.0) * 0.4;
			 Wetness.a = min(d, 256.0) * 0.002 - 0.008;
		 	 Wetness.a *= pow(clamp01(dot(sunLightingColor, vec3(0.333))), 5.0);
			 Wetness.a *= clamp01((-min(cameraHight, worldHight) + 120.0) / 48.0);
			 Wetness.a = clamp01(Wetness.a * 300.0);

	if(isSky){
		Wetness.a *= 0.78;
	}else{
		Wetness.rgb *= 0.78;

		vec4 atmosphericScattering = vec4(skyLightingColor / (0.5 + dot(skyLightingColor, vec3(0.333))), clamp(d / 500.0, 0.0, 0.78));

		color = mix(color, atmosphericScattering.rgb, atmosphericScattering.a);
	}

	color = mix(color, Wetness.rgb, Wetness.a * 0.48);

	return color;
}

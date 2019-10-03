#define SunLight 2.0	//[2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0]
#define SkyLight 1.0	//

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

	#ifdef Void_Sky
	if(wP.y + (o.y * length(wP) * 0.0000085) < 0.0){
		wP.y = abs(wP.y + o.y * length(wP) * 0.0000085) / (0.001 + abs(6.0 / wP.y)) * 6.0 - 0.001;
	}
	#endif

	float g = 0.76f;
  float mu = min(dot(wP, sP), 1.0); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction

  float phaseR = 3.f / (16.f * Pi) * (1 + mu * mu);
  float phaseM = 3.f / (8.f * Pi) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

  int steps = 4;
  int stepss = 4;

	//if(wP.y * length(wP) * near + o.y + rE < rA) return vec3(1.0);

  float u = escape(cP, wP, rA) / steps * 0.25;

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

    vec3 tau = bR * (opticalDepthR + opticalDepthLightR) + bM * (opticalDepthM + opticalDepthLightM);
    vec3 attenuation = exp(-tau);

    r += attenuation * hr;
    m += attenuation * hm;
  }

  return (r * bR * phaseR + m * bM * phaseM) * 20.0;
  }
}

vec3 CalculateSky(in vec3 viewVector, in vec3 lightVector, in float height, in float dither){
	viewVector = mat3(gbufferModelViewInverse) * viewVector;
	//if(viewVector.y < 0.0) viewVector.y = 0.04 / -viewVector.y;

	vec3 scattering = AtmosphericScattering(vec3(0.0, height - 63.0, 0.0), viewVector, lightVector, dither);
	scattering = max(vec3(0.0), scattering) * 0.16;
	//if(scattering.r + scattering.g + scattering.b < 0.001) scattering = vec3(1.0);

	//if(viewVector.y < 0.0) scattering = vec3(-viewVector.y);

	//scattering = mix(scattering, vec3(1.0), clamp01(-viewVector.y + 0.05));

	return scattering;
}

vec3 CalculateAtmosphericScattering(in vec3 color, in float factor){
	//color = vec3(0.0);
	//color = rgb2L(color);
	color = mix(color, (skyLightingColorRaw + sunLightingColorRaw * 0.5), (clamp01(factor)));
	//color = L2rgb(color);

	return color;
}

vec3 CalculateSun(in vec3 color, in vec3 vP, in vec3 lP, in vec3 lightColor, in float h){
	float p = clamp01(dot(vP, lP));
				p = clamp01(pow5((p - 0.9995) * 2000.0) * 620.0) * clamp01(h + 0.0063);

	color = mix(color, lightColor * 60, p);

	//color -= p * lightColor;
	//color = abs(color);
	//color += p * lightColor;

	return color;
}

#ifndef disable
	#define disable -255
#endif

#ifndef swamp
	#define swqmp 6
#endif

#define default_biome 1
#define desert 2
#define forest 4
#define taiga 5
#define snowy_tundra 12
#define mushroom_fields 14
#define jungle 21
#define birch_forest 27
#define snowy_taiga 30
#define savanna 35
#define badland 37

//#define Temperatrue_Test disable //[default_biome desert forest taiga snowy_tundra jungle snowy_taiga savanna badland]
//#define Rainfall_Test disable		 //[default_biome desert forest taiga snowy_tundra jungle snowy_taiga savanna badland]
#define Temperature_and_Rainfall_Test disable //[disable default_biome desert forest taiga snowy_tundra mushroom_fields jungle birch_forest snowy_taiga savanna badland]

void CalculateFog(inout vec3 color, in float l, in float h, in float temperature, in float rainfall, in vec3 lightDirect){
	vec3 fogColor = sunLightingColorRaw * (0.02 + lightDirect * 0.98) * 2.33 + skyLightingColorRaw * 1.99 + 0.12;
	//sunLightingColorRaw * lightDirect * 2.33 +

	//biome.y *= biome.x;

	//float temperature = 1.0;

	//temperature = 1.0 - clamp01(dot(normalize(shadowLightPosition), normalize(upPosition)));

	float d = 1.0;

	//float h2 = 1.0 - clamp01(exp(-max(0.0, 90.0 - h) / 24.0));
	float wetnessHight = clamp01((90.0 - h) / 24.0);

	float fogFactor = 1.0 - clamp01(exp(-(l * 0.0012 * temperature * rainfall * wetnessHight)));
			  //fogFactor = min(fogFactor, h2);
				//fogFactor *= h2;
			  //fogFactor = fogFactor * (0.8 + 0.2 * clamp01(dot03(fogColor))) * 0.1;

	//fogColor /= overRange;

	float range = 1.0 / (overRange * overRange);
	//color = vec3(0.0);

	//color = vec3(h2 * range * 0.5);

	//color.rgb = mix(color, vec3(1.0), clamp01(fogColor * fogFactor / (overRange * overRange) * 0.005));
	//color.rgb = mix(color.rgb, vec3(1.0), lightDirect * range * 0.012 * sunLightingColorRaw);
	color.rgb = mix(color.rgb, vec3(1.0), clamp01(fogFactor * fogColor * range * 0.12 * temperature * rainfall));
	//color = skyColor;
	//color.rgb = vec3(lightDirect) * range * 0.1;

	//color = mix(color, rgb2L(vec3(1.0 / overRange)), clamp01(fogFactor * fogColor * 0.01));

	//color.rgb = fogColor / overRange;

	//color = mix(color.rgb, sunLightingColorRaw, clamp01(sunLightingColorRaw * lightDirect * 0.0002));

}

/*
float GetFogDensity(in float dist, in bool calcLighting){
	if(calcLighting){
		return clamp01((-cameraHight + 120.0) / 48.0);
	}
}
*/
/*
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
*/

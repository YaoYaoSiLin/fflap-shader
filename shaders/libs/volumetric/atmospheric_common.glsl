
#ifndef INCLUDE_ATMOSPHERIC_COMMON
#define INCLUDE_ATMOSPHERIC_COMMON

const vec3  rayleigh_scattering         = vec3(5.8, 13.5, 33.1) * 1e-6;
const vec3  rayleigh_extinction         = rayleigh_scattering;
const float rayleigh_distribution       = 8000.0;

const vec3  mie_scattering              = vec3(2.0, 2.0, 2.0) * 1e-6;
const vec3  mie_extinction              = mie_scattering * 1.11;
const float mie_distribution            = 1200.0;

const vec3  ozone_scattering            = vec3(0.0);
const vec3  ozone_absorption            = vec3(3.426, 8.298, 0.356) * 0.12 * 10e-7;

const int   far_atmospheric_sample_steps            = 12;
const int   far_atmospheric_light_sample_steps      = 6;

const int   middle_atmospheric_sample_steps         = 8;
const int   middle_atmospheric_light_sample_steps   = 6;

const float planet_radius = 6360e3;
const float atmosphere_radius = 6420e3;

//#define High_Quality_Clouds
//#define Enabled_Volumetric_Clouds

#ifdef High_Quality_Clouds

#endif

#ifndef Altitude_Scale
#define Altitude_Scale 1.0		//[1.0 3.0 9.0 18.0 27.0 54.0 81.0 163.0 243.0 729.0 2187.0 6561.0]
#endif

#ifndef INCLUDE_ATMOSPHERIC
#define SunLight 3.0		//
#define SkyLight 1.0		//
#define MoonLight 0.12		//

const float rE = 6360e3;
const float rA = 6420e3;
const float Hr = 8000.0;
const float Hm = 1200.0;

const vec3 bM = vec3(2e-6);
const vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6);

////////////////

float HG(in float m, in float g){
  return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}

////////////////

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

////////////////

#endif

vec3 e = vec3(cameraPosition.x * Altitude_Scale, max(1.0, (cameraPosition.y - 63.0) * Altitude_Scale), cameraPosition.z * Altitude_Scale);
vec3 E = vec3(0.0, planet_radius + e.y, 0.0);

#endif
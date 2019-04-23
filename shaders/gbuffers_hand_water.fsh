#version 120

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform sampler2D noisetex;

uniform float wetness;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;

uniform mat4 gbufferModelViewInverse;

varying float isWater;
varying float isGlass;
varying float isIce;

varying vec2 uv;
varying vec2 lmcoord;

varying vec3 vP;

varying vec3 normal;
varying vec3 tangent;
varying vec3 binormal;

varying vec4 color;

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}
/*
const float rE = 6360e3;
const float rA = 6420e3;
const float Hr = 7994;
const float Hm = 1200;

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

float rl = 5895 / (171 - 63) * 8.0;

const vec3 bM = vec3(21e-6);
const vec3 bR = vec3(5.8e-6, 13.5e-6, 33.1e-6);

vec3 AtmosphericScattering(in vec3 o, in vec3 wP, in vec3 sP){
  vec3 cP = vec3(0.0, o.y + rE, 0.0);

  const float Pi = 3.1415;

  float mu = min(dot(wP, sP), 1.0); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction
  float phaseR = 3.f / (16.f * Pi) * (1 + mu * mu);
  float g = 0.76f;
  float phaseM = 3.f / (8.f * Pi) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));

  int steps = 4;
  int stepss = 4;

  float u = escape(cP, wP, rA) / steps;

  sP *= 0.333;

  //if(u < 0.0) return vec3(1.0);
  //else return vec3(1.0);

  //if(u2 < cP.y || !sky){
  vec3 r = vec3(0.0);
  vec3 m = vec3(0.0);

  float opticalDepthR = 0.0;
  float opticalDepthM = 0.0;

  float l = 0.0;
  float sl = 0.0;

  for(int i = 0; i < steps; ++i){
    vec3 p = cP + (wP) * (u * 0.5 + l);
    float h = length(p) - rE;

    float hr = exp(-h / Hr) * u;
    float hm = exp(-h / Hm) * u;

    opticalDepthR += hr;
    opticalDepthM += hm;

	l += u;

    float opticalDepthLightR = 0.0;
    float opticalDepthLightM = 0.0;

    float uS = escape(p, sP, rA) / stepss;

    if(uS > 0.0){
      for(int j = 0; j < stepss; ++j){
        vec3 pns = p + sP * (uS * 0.5 + sl);
        float hL = length(pns) - rE;

        opticalDepthLightR += exp(-hL / Hr) * uS;
        opticalDepthLightM += exp(-hL / Hm) * uS;

		    sl += uS;
      }
    }

    vec3 tau = bR * (opticalDepthR + opticalDepthLightR) + 1.1 * bM * (opticalDepthM + opticalDepthLightM);
    vec3 attenuation = exp(-tau);

    r += attenuation * hr;
    m += attenuation * hm;
  }

  //return vec3(u > 100000);

  return (r * bR * phaseR + 1.1 * m * bM * phaseM) * 20.0;
  //}
}
*/
void main() {

  vec4 albedo = texture2D(texture, uv) * color;

  vec3 normalTexture = texture2D(normals, uv).xyz * 2.0 - 1.0;
  mat3 tbnMatrix = mat3(tangent, binormal, normal);
  normalTexture = normalize(tbnMatrix * normalTexture);

  float smoothness = 0.0;
  float metalness = 0.0;

  float r = 1.0;

  if(isGlass > 0.01){
    smoothness = 0.75;
    //metalness = 1.0 - pow(albedo.a, 3.0);
    r = 1.51;
  }

  if(isIce > 0.1){
    smoothness = 0.91;
    r = 1.31;
  }

  if(isWater > 0.1){
    albedo = vec4(0.02, 0.02, 0.02, 0.98);
    smoothness = 0.88;
  }

  //vec2 texcoord = vec3(gbufferProjection * nvec4(vP)).xy * 0.5 + 0.5;

  vec3 color = albedo.rgb * (1.0 - metalness) * lmcoord.y * 0.2;
  color = pow(color, vec3(1.0 + pow(smoothness, 3.0) * 1.0));
  color = mix(color, vec3(0.0), pow(smoothness, 2.0) * 0.14);

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color, albedo.a);
}

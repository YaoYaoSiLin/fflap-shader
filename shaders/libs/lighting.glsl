#ifndef INCLUDE_LIGHTING
  #define INCLUDE_LIGHTING
#endif

#define ShaderLab1.2
#define SEUSv10
#define GrayScale
#define PulchraRevisited_V11

const float isMetallic = 0.9;
const float isMetal = 0.9;
//shader lab format

float SchlickFresnel(in float cosTheta){
	return pow5(1.0 - cosTheta);
}

vec3 SchlickFresnel(in vec3 F0, in float cosTheta){
	return F0 + (1.0 - F0) * SchlickFresnel(cosTheta);
}

vec3 SchlickFresnel(in vec3 F0, in vec3 L, in vec3 v){
	vec3 h = normalize(L + v);
	float vdoth = max(0.0, dot(v, h));

	return F0 + (1.0 - F0) * SchlickFresnel(vdoth);
}

float DistributionTerm( float roughness, float ndoth ) {
  //roughness *= roughness;

	float d	 = ( ndoth * roughness - ndoth ) * ndoth + 1.0;
	return roughness / ( d * d * Pi );
}

float D(in float roughness, in float ndoth){
  return DistributionTerm(roughness, ndoth);
}

float SmithGGX(float cosTheta, float roughness){
  float r2 = roughness * roughness;
  float c2 = cosTheta * cosTheta;

  return (2.0 * cosTheta) / (cosTheta + sqrt(r2 + (1.0 - r2) * c2));
}

float G(in float cosTheta, in float roughness){
  return SmithGGX(cosTheta, roughness);
  //float a = roughness*roughness;
  //float b = cost * cost;

  //float cosTheta2 = cosTheta * cosTheta;
  //float roughness2 = roughness;

  //return 1.0 / (cost + sqrt(roughness2 + cosTheta2 - roughness2 * cosTheta2));
}

float VisibilityTerm(float cosTheta1, float cosTheta2, float roughness){
  return SmithGGX(cosTheta1, roughness) * SmithGGX(cosTheta2, roughness);
}

//float VisibilityTerm( float roughness, float ndotv, float ndotl ) {
//	float gv = ndotl * sqrt( ndotv * ( ndotv - ndotv * roughness ) + roughness );
//	float gl = ndotv * sqrt( ndotl * ( ndotl - ndotl * roughness ) + roughness );
//	return min(1.0, 0.5 / max( gv + gl, 0.00001 ));
//}

float CalculateBRDF(in vec3 viewPosition, in vec3 rayDirection, in vec3 normal, float roughness){
  vec3 h = normalize(rayDirection + viewPosition);

  float ndotv = max(0.0, dot(viewPosition, normal));
  float ndoth = max(0.0, dot(normal, h));
  float ndotl = max(0.0, dot(rayDirection, normal));

  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);

  return max(0.0, d * g);
}

void FDG(inout vec3 f, inout float g, inout float d, in vec3 v, in vec3 l, in vec3 n, in vec3 F0, float roughness){
  vec3 h = normalize(l + v);

  float ndotv = max(0.0, dot(v, n));
  float hdotv = max(0.0, dot(v, h));
  float ndoth = max(0.0, dot(n, h));
  float ndotl = max(0.0, dot(l, n));

  f = SchlickFresnel(F0, hdotv);
  d = DistributionTerm(roughness, ndoth);
  g = G(ndotl, roughness) * G(ndotv, roughness);
}

vec3 LambertLighting(in vec3 k, in float attenuation, in float cosTheta){
	return k / pow2(attenuation) * cosTheta;
}

vec3 LambertLighting(in vec3 L, in vec3 v, in vec3 n, in vec3 albedo, in vec3 F0, in float metallic){
	vec3 h = normalize(L + v);

	float ndoth = max(0.0, dot(n, h));
	float ndotv = max(0.0, dot(v, h));
	float ndotl = max(0.0, dot(L, n));

	float hdotl = max(0.0, dot(L, h));

	if(bool(step(ndotl, 0.1))) return vec3(0.0);

	vec3 f = SchlickFresnel(F0, ndoth); 
	vec3 kS = f;
	vec3 kD = (1.0 - kS) * albedo;

	float rayLength = length(L + v);

	vec3 diffuse = LambertLighting(kD, rayLength, ndotl) * (1.0 - metallic);
	vec3 specular = LambertLighting(kS, rayLength, ndoth);

	return diffuse;
}

//vec3 BRDFSpecular(in vec3)

vec3 DisneyDiffuse(in vec3 l, in vec3 v, in vec3 n, in float a, in vec3 albedo){
  vec3 h = normalize(l + v);

  float hdotl = dot(l, h);

  float ndotl = max(0.0, dot(l, n));
  float ndotv = max(0.0, dot(v, n));

  if(ndotv < 1e-4 || ndotl < 1e-4) return vec3(0.0);

  float FD90 = hdotl * hdotl * a * 2.0 + 0.5;

  float FDV = 1.0 + (FD90 - 1.0) * pow5(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * pow5(ndotl);

  return albedo * invPi * FDL * FDV;
}

vec3 DiffuseLight(in vec3 albedo, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 ntextured, in vec3 F0, in float roughness, in float metallic){
  if(bool(step(isMetallic, metallic))) return vec3(0.0);

  vec3 h = normalize(l + v);

  float ndotl = max(0.0, dot(l, ntextured));
  float ndotv = max(0.0, dot(v, ntextured));

  if(!bool(ndotl)) return vec3(0.0);
  
  float hdotl = max(0.0, dot(l, h));

  float FD90 = hdotl * hdotl * roughness * 2.0 + 0.5;

  float FDV = 1.0 + (FD90 - 1.0) * pow5(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * pow5(ndotl);

  vec3 f = SchlickFresnel(F0, hdotl);

  vec3 diffuse = invPi * albedo.rgb * FDL * FDV;
       diffuse *= 1.0 - metallic;
       diffuse *= 1.0 - f;

  return diffuse;
}

vec3 SpecularLight(in vec3 albedo, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 ntextured, in vec3 F0, in float roughness, in float metallic){
  vec3 h = normalize(l+v);

  float ndotl = max(0.0, dot(l, ntextured));
  if(ndotl < 1e-5) return vec3(0.0, 0.0, 0.0);

  float ndotv = max(0.0, dot(v, nvisible));
  if(ndotv < 1e-5) return vec3(0.0);

  float ndoth = max(0.0, dot(ntextured, h));

  float hdotl = max(0.0, dot(l, h));
  float hdotv = max(0.0, dot(v, h));

  float roughness2 = clamp(roughness * roughness, 1e-5, 1.0 - 1e-5);

  vec3 f = SchlickFresnel(F0, hdotv);
  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * ndotl * ndotv + 1e-5;

  vec3 specular = f * d * g / c * ndotl;

  return specular;
}

vec3 AmbientSpecular(in vec3 albedo, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 ntextured, in vec3 F0, in float roughness, in float metallic){
  vec3 h = normalize(l+v);

  float ndotl = max(0.0, dot(l, ntextured));
  if(ndotl < 1e-5) return vec3(0.0, 0.0, 0.0);

  float ndotv = max(0.0, dot(v, nvisible));
  if(ndotv < 1e-5) return vec3(0.0);

  float ndoth = max(0.0, dot(ntextured, h));

  float hdotl = max(0.0, dot(l, h));
  float hdotv = max(0.0, dot(v, h));

  float roughness2 = clamp(roughness * roughness, 1e-5, 1.0 - 1e-5);

  vec3 f = SchlickFresnel(F0, hdotv);
  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * ndotl * ndotv + 1e-5;

  vec3 specular = f * saturate(d * saturate(g / c) * ndotl);

  return specular;
}

vec3 Sheen(in vec3 tint, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 ntextured, in vec3 F0, in float roughness, in float metallic, in float materials){
  float impermeable = step(materials, 0.5);
  float scatter = step(64.5, materials);
  float metals = step(isMetallic, metallic);
  float porosity = materials / 64.0;

  if(bool(impermeable) || bool(scatter) || bool(metals)) return vec3(0.0);

  vec3 sheenTint = mix(vec3(1.0), tint, 1.0);

  vec3 h = normalize(l + v);

  vec3 sheen = SchlickFresnel(max(0.0, dot(l, h))) * sheenTint * (1.0 - metallic);

  return sheen;
}

vec3 BRDFLighting(in vec3 albedo, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 nfull, in vec3 F0, in float roughness, in float metallic){
  //roughness = mix(1.0, roughness, );

  vec3 h = normalize(l+v);

  float ndotl = max(0.0, dot(l, nfull));
  float ndotv = max(0.0, dot(v, nfull));

  if(ndotl < 1e-5) return vec3(0.0);

  float ndoth = max(0.0, dot(nfull, h));

  float hdotl = max(0.0, dot(l, h));
  float hdotv = max(0.0, dot(v, h));

  float FD90 = hdotl * hdotl * roughness * 2.0 + 0.5;

  float FDV = 1.0 + (FD90 - 1.0) * pow5(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * pow5(ndotl);

  vec3 diffuse = invPi * albedo.rgb * FDL * FDV;
       diffuse *= step(metallic, isMetallic) * (1.0 - metallic);

  ndotl = max(0.0, dot(l, nvisible));
  ndotv = max(0.0, dot(v, nvisible));
  ndoth = max(0.0, dot(nvisible, h));

  if(ndotl < 1e-5) return vec3(0.0);

  roughness *= roughness;

  vec3 f = SchlickFresnel(F0, hdotl);
  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * (1e-5 + ndotl * ndotv);

  vec3 specular = saturate(f * g * d / c);

  vec3 lighting = diffuse + specular;
  //     lighting *= saturate(pow5(dot(nfull, l) + 0.5) * 4.0);

  return lighting;
}

vec3 BRDFLighting(in vec3 albedo, in float attenuation, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 ntextured, in vec3 F0, in float roughness, in float metallic, in float materials){
  float impermeable = step(materials, 0.5);
  float sss = step(64.5, materials);
  float metals = step(isMetallic, metallic);
  float porosity = (1.0 - impermeable) * (1.0 - sss) * (1.0 - metals);

  vec3 tint = albedo;

  float roughness2 = clamp(roughness * roughness, 1e-5, 1.0 - 1e-5);

  vec3 h = normalize(l + v);

  float ndotl = max(0.0, dot(l, ntextured));
  float ndotv = max(0.0, dot(v, nvisible));
  float ndoth = max(0.0, dot(ntextured, h));

  if(bool(step(ndotl, 1e-5) + step(ndotv, 1e-5) + step(ndoth, 1e-5))) return vec3(0.0, 0.0, 0.0);

  float hdotl = max(0.0, dot(l, h));
  float hdotv = max(0.0, dot(v, h));

  vec3 f = SchlickFresnel(F0, hdotl);

  float FD90 =  hdotl * hdotl * roughness * 2.0 + 0.5;

  float FDV = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotl);

  vec3 diffuse = albedo.rgb * mix(vec3(1.0), tint, porosity * (1.0 - ndotl));
       diffuse *= invPi * FDL * FDV * ndotl / attenuation;
       diffuse *= 1.0 - metallic;
       diffuse *= 1.0 - metals;

  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * ndotl * ndotv;

  vec3 specular = mix(vec3(1.0), tint, porosity * SchlickFresnel(hdotl)) * f * (d * (g / c) * ndotl / attenuation);

  vec3 sheen = tint;
       sheen *= SchlickFresnel(hdotl) * porosity * saturate(ndotl) / attenuation;

  return diffuse + specular + sheen;
}

vec3 BRDFShading(in vec3 L, in vec4 albedo, in vec3 l, in vec3 e, in vec3 nvisible, in vec3 ntextured, in vec3 F0, in float roughness, in float metallic, in float materials){
  float impermeable = step(materials, 0.5);
  float sss = step(64.5, materials) * step(albedo.a, 0.99);
  float metals = step(isMetallic, metallic);
  float porosity = (1.0 - impermeable) * (1.0 - sss) * (1.0 - metals);

  float sigma_s = albedo.a * max(0.05, 1.0 - (materials - 65.0) / 190.0) * 20.0;

  vec3 tint = albedo.rgb;

  vec3 h = normalize(l + e);

  float ndotl = max(0.0, dot(l, ntextured));
  float ndotv = max(0.0, dot(e, nvisible));
  float ndoth = max(0.0, dot(ntextured, h));

  if(bool(step(ndotl, 1e-5) + step(ndotv, 1e-5) + step(ndoth, 1e-5))) return vec3(0.0, 0.0, 0.0);

  float hdotl = max(0.0, dot(l, h));
  float hdotv = max(0.0, dot(e, h));

  vec3 kS = SchlickFresnel(F0, hdotl);
  vec3 kD = 1.0 - kS;

  float FD90 =  hdotl * hdotl * roughness * 2.0 + 0.5;

  float FDV = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotl);

  vec3 diffuse = albedo.rgb * invPi * FDL * FDV * L * ndotl;
       diffuse *= 1.0 - metallic;
       diffuse *= 1.0 - metals;
       diffuse *= max(1.0 - sss, sigma_s * 0.01);
       diffuse *= kD;
       diffuse *= mix(vec3(1.0), tint, porosity * (1.0 - ndotl));

  vec3 f = kS;
  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * ndotl * ndotv;

  vec3 specular = saturate(f * min(1.0, g / c) * d * ndotl * L);
       specular *= mix(vec3(1.0), tint, porosity * SchlickFresnel(hdotl));

  vec3 sheen = SchlickFresnel(hdotl) * tint * porosity * 1.0 * saturate(L * ndotl);

  return diffuse + specular + sheen;
}

vec3 IBLReflection(in vec3 L, in vec3 v, in vec3 n, in vec3 F0, float roughness, float metallic){
	vec3 f = vec3(0.0);
	float g = 0.0;
  float d = 0.0;
	float c = max(1e-5, 4.0 * max(0.0, dot(n, L)) * max(0.0, dot(n, v)));

	FDG(f, g, d, v, L, n, F0, roughness);

  float cosTheta = max(0.0, dot(L, n));

  return f * saturate(d * min(1.0, g / c) * cosTheta);
}

float ApplyBRDFBias(in float a){
  //default : 0.7
  return mix(a, 0.0, 0.9);
}

vec4 ImportanceSampleGGX(in vec2 E, in float roughness){
  //roughness *= roughness;
  roughness = clamp(roughness, 1e-5, 1.0 - 1e-5);

  float Phi = E.x * 2.0 * Pi;
  float CosTheta = sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  vec3 H = vec3(cos(Phi) * SinTheta, sin(Phi) * SinTheta, CosTheta);
  float D = DistributionTerm(roughness, abs(CosTheta)) * CosTheta;

  return vec4(H, D);
}

vec3 DiffuseLighting(in vec4 albedo, in vec3 L, in vec3 E, in vec3 n0, in vec3 n1, in vec3 F0, in float roughness, in float metallic, in float material){
  float impermeable = step(material, 0.5);
  float sss = step(64.5, material);
  float metals = step(isMetallic, metallic);
  float porosity = (1.0 - impermeable) * (1.0 - sss) * (1.0 - metals);

  vec3 h = normalize(L + E);

  float hdotl = max(0.0, dot(L, h));

  float ndotv = max(0.0, dot(n0, E));
  float ndotl = max(0.0, dot(n0, L));

  //if(min(ndotl, ndotv) <= 0.0) return vec3(0.0);

  //ndotl = mix(1.0, ndotl, rescale(0.05, 0.1, ndotl));
  //ndotv = mix(1.0, ndotv, rescale(0.05, 0.1, ndotv));

  vec3 kS = SchlickFresnel(F0, hdotl);
  vec3 kD = 1.0 - kS;

  float FD90 =  hdotl * hdotl * roughness * 2.0 + 0.5;
  float FDV = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotl);

  vec3 diffuse = (albedo.rgb * kD) * (invPi * FDL * FDV * ndotl);
       diffuse *= 1.0 - metallic;
       diffuse *= 1.0 - metals;
       diffuse *= sss > 0.5 ? albedo.a : 1.0;
       diffuse *= mix(vec3(1.0), albedo.rgb, porosity * SchlickFresnel(ndotl));

  return diffuse;
}

vec3 SpecularLighting(in vec3 albedo, in vec3 L, in vec3 E, in vec3 ntextured, in vec3 nvisible, in vec3 F0, in float roughness, in float metallic, in float material){
  float impermeable = step(material, 0.5);
  float sss = step(64.5, material);
  float metals = step(isMetallic, metallic);
  float porosity = (1.0 - impermeable) * (1.0 - sss) * (1.0 - metals);

  vec3 H = normalize(L + E);

  float ndotl = max(0.0, dot(ntextured, L));
  float ndotv = max(0.0, dot(nvisible, E));
  float ndoth = max(0.0, dot(ntextured, H));
  float hdotl = max(0.0, dot(H, L));

  //ndotl = mix(1.0, ndotl, rescale(0.05, 0.1, ndotl));
  ndotv = mix(1.0, ndotv, rescale(0.05, 0.1, ndotv));
  //ndoth = mix(1.0, ndoth, rescale(0.05, 0.1, ndoth));

  if(ndotl <= 0.0) return vec3(0.0);

  vec3 f = SchlickFresnel(F0, hdotl);

  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * ndotl * ndotv + 1e-5;

  vec3 specular = f * (d * (g / c) * ndotl);
       specular *= mix(vec3(1.0), albedo, porosity * SchlickFresnel(hdotl));

  vec3 sheen = SchlickFresnel(hdotl) * albedo * porosity * 1.0 * ndotl;

  return specular + sheen;
}

vec3 SpecularLighting(in vec4 albedo, in vec3 L, in vec3 E, in vec3 ntextured, in vec3 nvisible, in vec3 F0, in float roughness, in float metallic, in float material, bool low_light){
  roughness = clamp(roughness, 1e-5, 1.0 - 1e-5);

  float impermeable = step(material, 0.5);
  float sss = step(64.5, material);
  float metals = step(isMetallic, metallic);
  float porosity = (1.0 - impermeable) * (1.0 - sss) * (1.0 - metals);

  float clearcoat = metallic;

  vec3 H = normalize(L + E);

  float ndotl = max(0.0, dot(ntextured, L));
  float ndotv = max(0.0, dot(nvisible, E));
  float ndoth = max(0.0, dot(ntextured, H));
  float hdotl = max(0.0, dot(H, L));

  if(low_light) {
    //ndotl = mix(1.0, ndotl, rescale(0.05, 0.1, ndotl));
    ndotv = mix(1.0, ndotv, rescale(0.05, 0.1, ndotv));
    //ndoth = mix(1.0, ndoth, rescale(0.05, 0.1, ndoth));
  }

  float angle = SchlickFresnel(hdotl);

  if(min(ndotl, ndotv) <= 0.0) return vec3(0.0, 0.0, 0.0);

  vec3 f = SchlickFresnel(F0, hdotl);

  float d = DistributionTerm(roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * ndotl * ndotv + 1e-5;

  vec3 specular = f * (low_light ? min(1.0, d) * min(1.0, g * d / c * ndotl) : (g * d / c * ndotl));
       specular *= mix(vec3(1.0), albedo.rgb, vec3(porosity * angle));

  //vec3 specular = vec3(d * g / c * ndotl * max(metals, clearcoat));
  //     specular *= mix(vec3(1.0), albedo.rgb, vec3(porosity * angle));
  //     specular = low_light ? f * min(vec3(1.0), specular) : f * specular;

  //vec3 specular = vec3(d * (g / c) * ndotl);
  //     specular *= mix(vec3(1.0), vec3(min(1.0, albedo.a)), vec3(1.0 - angle) * sss);
  //     specular = low_light ? f * min(vec3(1.0), specular * mix(vec3(1.0), albedo.rgb, vec3(porosity * angle))) : f * specular;

  vec3 sheen = albedo.rgb * (angle * porosity * ndotl * clearcoat);

  return specular + sheen;
}
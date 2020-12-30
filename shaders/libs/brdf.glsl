vec3 F(vec3 F0, float cosTheta){
 return F0 + (1.0 - F0) * cosTheta;
}

vec3 F(vec3 F0, vec3 V, in vec3 N){
  float cosTheta = pow5(1.0 - saturate(dot(V, N)));

 return F(F0, cosTheta);
}

float DistributionTerm( float roughness, float ndoth ) {
  roughness = clamp(roughness, 0.0001, 0.9999);

	float d	 = ( ndoth * roughness - ndoth ) * ndoth + 1.0;
	return roughness / ( d * d * Pi );
}

float G(in float cost, in float roughness){
  float a = roughness*roughness;
  float b = cost * cost;

  return 1.0 / (cost + sqrt(a + b - a * b));
}

float VisibilityTerm( float roughness, float ndotv, float ndotl )
{
	float gv = ndotl * sqrt( ndotv * ( ndotv - ndotv * roughness ) + roughness );
	float gl = ndotv * sqrt( ndotl * ( ndotl - ndotl * roughness ) + roughness );
	return min(1.0, 0.5 / max( gv + gl, 0.00001 ));
}

float CalculateBRDF(in vec3 viewPosition, in vec3 rayDirection, in vec3 normal, float roughness){
  vec3 h = normalize(rayDirection + viewPosition);

  float ndotv = clamp01(dot(viewPosition, normal));
  float ndoth = clamp01(dot(normal, h));
  float ndotl = clamp01(dot(rayDirection, normal));

  float d = DistributionTerm(roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);

  return max(0.0, d * g);
}

void FDG(inout vec3 f, inout float g, inout float d, in vec3 rayOrigin, in vec3 rayDirection, in vec3 n, in vec3 F0, float roughness){
  vec3 m = normalize(rayDirection + rayOrigin);

  float ndotv = clamp01(dot(rayOrigin, m));
  float ndoth = clamp01(dot(n, m));
  float ndotl = clamp01(dot(rayDirection, n));

  f = F(F0, pow5(1.0 - ndotv));
  d = DistributionTerm(roughness * roughness, ndoth);
  g = G(ndotl, roughness) * G(ndotv, roughness);
}

#define HightLightNerf 0.996

float ApplyBRDFBias(in float a){
  return mix(a, 0.0, 0.7);
}

#if CalculateHightLight == 1

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
  if(bool(step(0.5, metallic))) return vec3(0.0);

  vec3 h = normalize(l + v);

  float ndotl = max(0.0, dot(l, ntextured));
  float ndotv = max(0.0, dot(v, ntextured));

  if(!bool(ndotl)) return vec3(0.0);
  
  float hdotl = max(0.0, dot(l, h));

  float FD90 = hdotl * hdotl * roughness * 2.0 + 0.5;

  float FDV = 1.0 + (FD90 - 1.0) * pow5(ndotv);
  float FDL = 1.0 + (FD90 - 1.0) * pow5(ndotl);

  vec3 f = F(F0, max(0.0, pow5(1.0 - hdotl)));

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

  float hdotl = max(0.0, dot(l, h));
  float ndoth = max(0.0, dot(ntextured, h));

  vec3 f = F(F0, pow5(1.0 - hdotl));
  float d = DistributionTerm(roughness * roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * (1e-5 + ndotl * ndotv);

  vec3 specular = saturate(f * g * d / c * ndotl);

  return specular;
}

vec3 BRDFLighting(in vec3 albedo, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 nfull, in vec3 F0, in float roughness, in float metallic){
  roughness = mix(1.0, roughness, HightLightNerf);

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
       diffuse *= step(metallic, 0.5) * (1.0 - metallic);

  ndotl = max(0.0, dot(l, nvisible));
  ndotv = max(0.0, dot(v, nvisible));
  ndoth = max(0.0, dot(nvisible, h));

  if(ndotl < 1e-5) return vec3(0.0);

  vec3 f = F(F0, pow5(1.0 - hdotl));
  float d = DistributionTerm(roughness * roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * (1e-2 + ndotl * ndotv);

  vec3 specular = saturate(f * g * d / c * ndotl);

  vec3 lighting = diffuse + specular;
       lighting *= saturate(pow5(dot(nfull, l) + 0.5) * 4.0);

  return lighting;
}

#endif

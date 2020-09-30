vec3 F(vec3 F0, float cosTheta){
 return F0 + (1.0 - F0) * cosTheta;
}

float DistributionTerm( float roughness, float ndoth )
{
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
  g = VisibilityTerm(d, ndotv, ndotl);
}

#define BRDF_Bias 0.7

#define HightLightNerf 0.996


float ApplyBRDFBias(in float a){
  return mix(a, 0.0, BRDF_Bias);
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


vec3 BRDFLighting(in vec3 albedo, in vec3 l, in vec3 v, in vec3 nvisible, in vec3 nfull, in vec3 F0, in float roughness, in float metallic){
  roughness = ApplyBRDFBias(roughness);
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

  vec3 diffuse = albedo.rgb * invPi * FDL * FDV;
       diffuse *= (1.0 - step(0.5, metallic));

  ndotl = max(0.0, dot(l, nvisible));
  ndotv = max(0.0, dot(v, nvisible));
  ndoth = max(0.0, dot(nvisible, h));

  if(ndotl < 1e-5) return vec3(0.0);

  vec3 f = F(F0, pow5(1.0 - hdotl));
  float d = DistributionTerm(roughness * roughness, ndoth);
  float g = G(ndotl, roughness) * G(ndotv, roughness);
  float c = 4.0 * (1e-2 + ndotl * ndotv);

  vec3 lighting = diffuse + f * g * d / c;
       lighting *= saturate(pow5(dot(nfull, l) + 0.5) * 4.0);

  return lighting;
}

#endif

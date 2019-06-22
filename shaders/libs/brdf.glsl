vec3 F(vec3 albedo, float cosTheta){
 return albedo + (1.0 - albedo) * cosTheta;
}

float DistributionTerm( float roughness, float ndoth )
{
	float d	 = ( ndoth * roughness - ndoth ) * ndoth + 1.0;
	return roughness / ( d * d * Pi );
}

float VisibilityTerm( float roughness, float ndotv, float ndotl )
{
	float gv = ndotl * sqrt( ndotv * ( ndotv - ndotv * roughness ) + roughness );
	float gl = ndotv * sqrt( ndotl * ( ndotl - ndotl * roughness ) + roughness );
	return min(1.0, 0.5 / max( gv + gl, 0.00001 ));
}

#if CalculateHightLight == 1
vec3 BRDF(in vec3 albedo, in vec3 L, in vec3 viewVector, in vec3 normal, in float roughness, in float metallic, in vec3 F0){
  roughness *= roughness;
  roughness = max(roughness, 0.0001);

  vec3 h = normalize(L + viewVector);

  float ndotv = 1.0 - clamp01(dot(viewVector, normal));
  float vdoth = 1.0 - clamp01(dot(viewVector, h));
  float ndoth = clamp01(dot(normal, h));
  float ndotl = clamp01(dot(L, normal));

  vec3  f = F(F0, pow5(vdoth));
  float d = DistributionTerm(roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);
  float c = 4.0 * clamp01(dot(viewVector, normal)) * clamp01(dot(L, normal)) + 1.0;

  float FD90 = 0.5 + 2.0 * roughness * ndoth * ndoth;
  float FdV = 1.0 + (FD90 - 1.0) * pow5(clamp01(dot(viewVector, normal)));
  float FdL = 1.0 + (FD90 - 1.0) * pow5(1.0 - clamp01(dot(L, normal)));

  return ((albedo / Pi * FdV * FdL) * (1.0 - metallic)
         + clamp01(f * d * g) / c * 4.0
         ) * (ndotl);
}
#endif

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

float CalculateBRDF(in vec3 viewPosition, in vec3 rayDirection, in vec3 normal, float roughness){
  //roughness *= roughness;
  //roughness = max(roughness, 0.0001);

  vec3 h = normalize(rayDirection + viewPosition);

  float ndotv = 1.0 - clamp01(dot(viewPosition, normal));
  float ndoth = clamp01(dot(normal, h));
  float ndotl = clamp01(dot(rayDirection, normal));

  float d = DistributionTerm(roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);

  return max(0.0, d * g);
}

#if CalculateHightLight == 1
vec3 BRDF(in vec3 albedo, in vec3 L, in vec3 viewPosition, in vec3 visibleNormal, in vec3 lightVisibleNormal, in float roughness, in float metallic, in vec3 F0){
  roughness *= roughness;
  roughness = max(roughness, 0.0001);

  vec3 h = normalize(L + viewPosition);

  //if(dot(viewPosition, visibleNormal) < 0.15) visibleNormal = ;

  float vdoth = pow5(1.0 - clamp01(dot(viewPosition, h)));
  float ndotl = clamp01(dot(lightVisibleNormal, L));
  float ndoth = clamp01(dot(visibleNormal, h));
  float ndotv = 1.0 - clamp01(dot(viewPosition, visibleNormal));

  float c = 4.0 * clamp01(dot(viewPosition, visibleNormal)) * clamp01(dot(L, lightVisibleNormal)) + 0.0001;
  //if(dot(viewPosition, visibleNormal) < 0.05) c = 1.0;

  float FD90 = 0.5 + 2.0 * roughness * ndoth * ndoth;
  float FdV = 1.0 + (FD90 - 1.0) * pow5(clamp01(dot(viewPosition, visibleNormal)));
  float FdL = 1.0 + (FD90 - 1.0) * pow5(1.0 - clamp01(dot(L, lightVisibleNormal)));

  vec3 f = F(F0, vdoth);
  float d = DistributionTerm(roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);

  vec3 diffuse = (albedo / Pi * FdV * FdL) * (1.0 - metallic);
  vec3 highLight = max(vec3(0.0), f * max(0.0, g * d) / c) * step(0.01, dot(visibleNormal, viewPosition));

  return (diffuse + highLight) * min(1.0, pow3(ndotl * 3.0));
}

#endif

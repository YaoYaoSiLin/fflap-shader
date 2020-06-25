vec3 F(vec3 F0, float cosTheta){
 return F0 + (1.0 - F0) * cosTheta;
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
  vec3 h = normalize(rayDirection + viewPosition);

  float ndotv = 1.0 - clamp01(dot(viewPosition, normal));
  float ndoth = clamp01(dot(normal, h));
  float ndotl = clamp01(dot(rayDirection, normal));

  float d = DistributionTerm(roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);

  return max(0.0, d * g);
}

void FDG(inout vec3 f, inout float g, inout float d, in vec3 rayOrigin, in vec3 rayDirection, in vec3 n, in vec3 m, in vec3 F0, float roughness){
  float ndotv = 1.0 - clamp01(dot(rayOrigin, m));
  float ndoth = clamp01(dot(n, m));
  float ndotl = clamp01(dot(rayDirection, n));

  f = F(F0, pow5(ndotv));
  d = DistributionTerm(roughness * roughness, ndoth);
  g = VisibilityTerm(d, ndotv, ndotl);
}

//vec3 CalculateDiffuse

#if CalculateHightLight == 1
vec3 BRDF(in vec3 albedo, in vec3 L, in vec3 viewPosition, in vec3 visibleNormal, in vec3 surfaceNormal, in float roughness, in float metallic, in vec3 F0){
  //roughness *= roughness;
  //roughness = max(roughness, 0.0001);

  vec3 h = normalize(L + viewPosition);

  float vdoth = pow5(1.0 - clamp01(dot(viewPosition, h)));
  float ndotl = clamp01(dot(surfaceNormal, L));
  float ndoth = clamp01(dot(visibleNormal, h));
  float ndotv = 1.0 - clamp01(dot(viewPosition, visibleNormal));

  float c = 4.0 * (1.0 - ndotv) * ndotl;

  float FD90 = clamp01(0.5 + 2.0 * roughness * ndoth * ndoth);
  float FdV = 1.0 + (FD90 - 1.0) * vdoth;
  float FdL = 1.0 + (FD90 - 1.0) * pow5(ndotl);

  vec3 f = F(F0, vdoth);
  float d = DistributionTerm(roughness * roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);

  vec3 diffuse = albedo / Pi * FdV * FdL * (1.0 - metallic);
  vec3 highLight = f * max(0.0, g * d) / max(1.0, c);

  vec3 rL = normalize(reflect(-viewPosition, visibleNormal));

  vec3 specularityM = normalize(rL + viewPosition);

  float vdotsh = pow5(1.0 - clamp01(dot(specularityM, viewPosition)));

  vec3 specularityF = F(F0, vdotsh);
  float specularityBrdf = CalculateBRDF(viewPosition, rL, visibleNormal, roughness);
        //specularityBrdf = min(1.0, specularityBrdf * specularityBrdf);
        //specularityBrdf = specularityBrdf * specularityBrdf * specularityBrdf;

  vec3 specularity = clamp01(highLight / f * vdoth * 8.0) * min(vec3(1.0), specularityBrdf * specularityBrdf * specularityBrdf * specularityF);
  specularity *= 0.01 / (vdoth + 0.001);
  specularity += highLight;
  //specularity *= step(1.0 - dot(viewPosition, surfaceNormal), 0.9);
  //specularity = min(vec3(1.0), specularity);

  return (diffuse + specularity) * min((ndotl * ndotl * 32.0), 1.0);
}

#endif

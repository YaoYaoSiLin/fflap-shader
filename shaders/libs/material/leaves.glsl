vec3 LeavesShading(vec3 L, vec3 v, vec3 n, vec3 albedo, float material){
    vec3 scattering = pow(albedo, vec3(0.95));

    vec3 sigma_s = vec3((1.0 - (material - 65.0) / 190.0) * 0.1);
    vec3 transmittance = exp(-sigma_s * 0.01);

	float mu = dot(L, -v);
    float phase = mix(HG(mu, 0.8), HG(mu, 0.2), 0.95);
    
    return scattering * transmittance / sigma_s * 0.01 * phase * invPi;
}

#ifndef INCLUDE_LEAVES
#define INCLUDE_LEAVES
#endif
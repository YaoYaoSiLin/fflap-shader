#ifndef INCULDE_NOISE_COMMON
#define INCULDE_NOISE_COMMON

uniform sampler2D noisetex;
uniform sampler2D colortex9;

const int noiseTextureResolution = 64;
const float worleyNoiseTextureResolution = 64.0;

float worley(in vec2 x){
    return 1.0 - texture(colortex9, x / worleyNoiseTextureResolution).x;
}    

float noise(in vec2 x){
    return texture(noisetex, x / noiseTextureResolution).x;
}

float noise(in vec3 x) {
    x = x.xzy;

    vec3 i = floor(x);
    vec3 f = fract(x);

	f = f*f*(3.0-2.0*f);

	vec2 uv = (i.xy + i.z * vec2(17.0)) + f.xy;
    uv += 0.5;

	vec2 rg = vec2(noise(uv), noise(uv+17.0));

	return mix(rg.x, rg.y, f.z);
}

#endif
#ifndef INCLUDE_SSS
    #define INCLUDE_SSS
#endif

#ifndef INCLUDE_ATMOSPHERIC
float HG(in float m, in float g){
  return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}
#endif

bool intersectCube(in vec3 origin, in vec3 direction, in vec3 size, out float near, out float far){
    vec3 dr = 1.0 / direction;
    vec3 n = origin * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    near = max(pin.x, max(pin.y, pin.z));
    far = min(pout.x, min(pout.y, pout.z));

	// check for hit
	return near < far && far > 0.0;
}
/*
bool intersectCube(in vec3 origin, in vec3 direction, in vec3 size, out float near, out float far, inout mat3 normal){
    vec3 dr = 1.0 / direction;
    vec3 n = origin * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    near = max(pin.x, max(pin.y, pin.z));
    far = min(pout.x, min(pout.y, pout.z));

    vec3 front = -sign(direction) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    vec3 back = -sign(direction) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);

    normal = mat3(normal[0], front, back);

	// check for hit
	return near < far && far > 0.0;
}
*/
bool intersectCube(in vec3 origin, in vec3 direction, in vec3 size, out float near, out float far, inout vec3 normal, bool bfront){
    vec3 dr = 1.0 / direction;
    vec3 n = origin * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    near = max(pin.x, max(pin.y, pin.z));
    far = min(pout.x, min(pout.y, pout.z));

    //vec3 front = -sign(direction) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    //vec3 back = -sign(direction) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);

    vec3 front = vec3(0.0);
    vec3 back = vec3(0.0);

    if(near == pin.x){
        front = direction.x < 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(-1.0, 0.0, 0.0);
    }else if(near == pin.y) {
        front = direction.y < 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, -1.0, 0.0);
    }else if(near == pin.z){
        front = direction.z < 0.0 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
    }

    if(far == pout.x){
        back = direction.x < 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(-1.0, 0.0, 0.0);
    }else if(far == pout.y){
        back = direction.y < 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, -1.0, 0.0);
    }else if(far == pout.z){
        back = direction.z < 0.0 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
    }

    normal = bfront ? front : back;

	// check for hit
	return near < far && far > 0.0;
}

vec3 LeavesShading(vec3 L, vec3 v, vec3 n, vec3 albedo, float material){
    vec3 h = normalize(L + v);

	float mu = dot(L, -v);
	float ndotl = dot(L, n);

    float sigma_s = max(0.05, (1.0 - (material - 65.0) / 190.0) * 30.0);

    float extinction = exp(-sigma_s * 0.01);

    return pow(albedo, vec3(0.95)) * (extinction * sigma_s * invPi * 0.05 * max(0.0, -ndotl * 0.5 + 0.5) * (HG(mu, 0.6) + HG(-0.9999, 0.2)));
}

vec3 FullCubeShaing(vec3 L, vec3 v, vec3 n, vec3 n1, vec3 albedo, vec3 F0, float material, float metal, vec3 sun, vec4 wP){
    //if(material < 64.5) return vec3(0.0);

    float sigma_s = max(0.05, (1.0 - (material - 65.0) / 190.0) * 30.0);

    float farPoint = -1.0;
    float nearPoint = -1.0;

	vec3 bias = vec3(0.0);

    vec3 worldNormal = n1;

    if(dot(worldNormal, vec3(0.0, 1.0, 0.0)) > 1.0 - 1e-5){
		bias = vec3(0.0, 1.0, 0.0);
	}else if(dot(worldNormal, vec3(0.0, -1.0, 0.0)) > 1.0 - 1e-5){
		bias = vec3(0.0, -1.0, 0.0);
	}else if(dot(worldNormal, vec3(1.0, 0.0, 0.0)) > 1.0 - 1e-5){
		bias = vec3(1.0, 0.0, 0.0);
	}else if(dot(worldNormal, vec3(-1.0, 0.0, 0.0)) > 1.0 - 1e-5){
		bias = vec3(-1.0, 0.0, 0.0);
	}else if(dot(worldNormal, vec3(0.0, 0.0, 1.0)) > 1.0 - 1e-5){
		bias = vec3(0.0, 0.0, 1.0);
	}else if(dot(worldNormal, vec3(0.0, 0.0, -1.0)) > 1.0 - 1e-5){
		bias = vec3(0.0, 0.0, -1.0);
	}

    vec3 cubePosition = floor(wP.xyz + cameraPosition - bias * 0.999) + 0.5;

    bool hit = intersectCube(cameraPosition - cubePosition, v, vec3(0.5), nearPoint, farPoint);
    float viewLength = length(wP.xyz);

    float ndotl = dot(L, n);

    float dither = GetBlueNoise(depthtex2, texcoord, resolution.y, jitter);

    //float phase = HG(dot(v, L), 0.9);

    if(hit){
        int steps = 4;
        float invsteps = 1.0 / float(steps);

        float stepLength = 0.0625;

        vec3 rayStart = v * max(0.0, nearPoint);
        vec3 rayStep = v * stepLength;
        vec3 rayPosition = rayStart + dither * rayStep;

        vec3 s = vec3(0.0);

        vec3 Fo = albedo;
        vec3 Fi = albedo;

        float mu = dot(L, v);
        float phaseFront = HG(mu, 0.96) * 0.1;
        float phaseBack =  HG(0.9999, 0.4);
        float phase = max(phaseFront, phaseBack);
        
        for(int i = 0; i < steps; i++){
            float rayLength = length(rayPosition);
            if(farPoint < rayLength - 0.05) break;

            float opricalDepth = length(rayPosition - rayStart);

            vec3 shadowCoord = wP2sP(vec4(rayPosition, wP.w)); shadowCoord.z -= 2.0 / 2048.0;
            float visibility = step(shadowCoord.z, texture(shadowtex0, shadowCoord.xy).x);
 
            float f = 0.0; float n = 0.0; vec3 surface = vec3(0.0);
            bool hit1 = intersectCube(cameraPosition - cubePosition + rayPosition, L, vec3(0.5), n, f, surface, false);
            float lightLength = hit1 ? f - max(0.0, n) : 0.0;

            vec3 extinction = vec3(exp(-(opricalDepth + lightLength) * sigma_s));

            s += extinction * visibility;
            
            rayPosition += rayStep;
        }

        return s * invPi * sigma_s * stepLength * phase * (Fi * Fo);
    }

    return vec3(0.0);
/*
            mat3 stepSurfaceNormal = mat3(vec3(0.0), vec3(0.0), vec3(0.0));
            float f = 0.0; float n = 0.0;
            bool hit1 = intersectCube(cameraPosition - cubePosition + rayPosition, v, vec3(0.503), n, f, stepSurfaceNormal);

            float opticalDepth = stepLength * (0.5 + float(i)) / 17.0 * 16.0;
                  opticalDepth += f - max(0.0, n);

            vec3 extinction = exp(-opticalDepth * sc);

            s += extinction * visibility / opticalDepth;
*/
            /*if(hit1){
                float stepLength = f - max(0.0, n);
f
                opticalDepth += stepLength;

                vec3 start = rayPosition;
                vec3 dirction = L * stepLength * invsteps;
                vec3 test = start;

                for(int j = 0; j < steps; j++){
                    
                    test += direction;
                }
            }*/
            //rayLength += stepLength;
        //}


        /*
        float lightNear = -1.0;
        float lightFar = -1.0; mat3 normal2 = mat3(vec3(0.0), vec3(0.0), vec3(0.0));
        bool hit1 = intersectCube(cameraPosition + v * (nearPoint > 0.0 ? nearPoint : farPoint) - cubePosition, L, vec3(0.51), lightNear, lightFar, normal2);

        vec3 enter = dot(normal[1], L) > 0.0 ? -n : -normal2[2];

        float rayLength = farPoint - max(0.0, nearPoint);
        float lightLength = hit1 ? lightFar - max(0.0, lightNear) : 0.0;

        vec3 extinction = saturate(exp(-(lightLength) * sc));
        vec3 extinction2 = saturate(exp(-rayLength * sc));

        return SchlickFresnel(vec3(0.0), 1.0 - saturate(dot(enter, L))) * albedo;// * extinction * invPi;
        */
        /*
        float rayLength = farPoint - max(0.0, nearPoint);
        float lightLength = hit1 ? lightFar - max(0.0, lightNear) : 0.0;

        vec3 extinction = saturate(exp(-(lightLength) * sc));
        vec3 extinction2 = saturate(exp(-rayLength * sc));

        vec3 e = -v;
        vec3 enter = dot(L, normal[1]) > 0.0 ? -n : -normal[2];

        vec3 Fi = SchlickFresnel(vec3(0.0), 1.0 - max(0.0, dot(L, normalize(L + -n)))) * albedo;
        vec3 Fo = albedo;

        float mu = dot(L, v);
        float Pf = min(8.0, HG(mu, 0.99));
        float Pb = HG(0.9999, 0.7);

        return Fi * Fo * extinction * mix(albedo, vec3(1.0), extinction) * (Pf + Pb * invPi);
        */
    //}

    //return vec3(0.0);
}
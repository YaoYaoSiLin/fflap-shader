/*
10^6/10^2
*/
//#define Enabled_Soft_Shadow         //16

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;

float shadowPixel = 1.0 / shadowMapResolution;


vec3 moonColor = vec3(0.713, 0.807, 0.815);

float CalculateSunLightFading(in vec3 wP, in vec3 sP){
  float h = playerEyeLevel + defaultHightLevel;
  return clamp01(dot(sP * defaultHightLevel, vec3(0.0, h, 0.0)) / (h * defaultHightLevel) * 10.0);
}

vec3 wP2sP(in vec4 wP){
	vec4 sP = (wP);
       sP = shadowModelView * sP;
       sP = shadowProjection * sP;
       sP /= sP.w;
       sP.xy /= mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95;
       sP.z /= max(far / shadowDistance, 1.0);
       sP = sP * 0.5 + 0.5;

	return sP.xyz;
}

vec3 CalculateFogLighting(in vec3 color, in float cameraHight, in float worldHight, in vec3 sunLightingColor, in vec3 skyLightingColor, in float fading){
	vec4 Wetness = vec4(0.0);
			 Wetness.rgb = (sunLightingColor * fading * 0.5 + skyLightingColor + 1.0) * 0.4;
			 Wetness.a = 0.504;
			 Wetness.a *= pow(clamp01(dot(sunLightingColor, vec3(0.333))), 5.0);
			 Wetness.a *= clamp01((-min(cameraHight, worldHight) + 120.0) / 48.0);
			 Wetness.a = clamp01(Wetness.a * 300.0) * 0.48;

	//color = mix(color, Wetness.rgb * 0.78, Wetness.a);
	//color *= pow(1.0 - Wetness.a, 2.0);

	return color;
}

vec4 CalculateShading(in sampler2D tex, in vec4 wP){
  float d = length(wP.xyz);

  if(d > shadowDistance) return vec4(0.0);

  float diffthresh = shadowPixel * (d / far) * 1.0 + 0.00012;

  vec3 shading = vec3(0.0);

  vec3 shadowPosition = wP2sP(wP);

  const float bias_pix = 0.0003;
	vec2 bias_offcenter = abs(shadowPosition.xy * 2.0 - 1.0);
  diffthresh = length(bias_offcenter) * bias_pix * 0.0 + shadowPixel * (d / far) * 1.0 + 0.0004;
  //diffthresh *= 2.5;

  shading = vec3(1.0);

  if(floor(shadowPosition.xyz) == vec3(0.0)){
    shading = vec3(float(texture2D(tex, shadowPosition.xy).z + diffthresh > shadowPosition.z));
    #if CalculateShadingColor == 1

    vec4 colorShading = texture2D(shadowcolor1, shadowPosition.xy);
    //shading = mix(shading, colorShading.rgb, shading.x * colorShading.a);
    //shading *= texture2D(shadowcolor1, shadowPosition.xy).rgb;
    shading = mix(vec3(1.0), colorShading.rgb, colorShading.a) * shading;

    #elif CalculateShadingColor == 2

    shading = vec3(texture2D(shadowtex1, shadowPosition.xy).z + diffthresh > shadowPosition.z);
    vec4 colorShading = texture2D(shadowcolor1, shadowPosition.xy);
    //shading = mix(shading, colorShading.rgb, (1.0 - shading.x) * float(texture2D(shadowtex1, shadowPosition.xy).z + diffthresh > shadowPosition.z) * colorShading.a);
    shading = mix(vec3(1.0), colorShading.rgb, colorShading.a * (1.0 - float(texture2D(shadowtex0, shadowPosition.xy).z + diffthresh > shadowPosition.z)) * (shading.x)) * shading;

    #endif
  }

  //shading = mix(vec3(1.0), shading, clamp01((-d + shadowDistance - 16.0) / 32.0));
  //shading *= texture2D(colortex2, texcoord).a;

  return vec4(shading, clamp01((-d + shadowDistance - 16.0) / 32.0));
}

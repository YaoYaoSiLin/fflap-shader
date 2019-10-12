/*
10^6/10^2
*/

//#define OFF 0

#define Simple 1
#define PCSS 2

#define Enabled_Soft_Shadow OFF               //[OFF Simple PCSS]

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution     = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance		  		= 140.0;
const bool  generateShadowMipmap    = false;
const bool  shadowHardwareFiltering = true;

uniform sampler2DShadow shadowtex0;
uniform sampler2DShadow shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 shadowProjection;
uniform mat4 shadowModelView;

float shadowPixel = 1.0 / shadowMapResolution;

vec3 moonColor = vec3(0.713, 0.807, 0.815);

float CalculateSunLightFading(in vec3 wP, in vec3 sP){
  float h = playerEyeLevel + defaultHightLevel;
  return clamp01(dot(sP * defaultHightLevel, vec3(0.0, h, 0.0)) / (h * defaultHightLevel) * 10.0);
}

vec3 wP2sP(in vec4 wP, out float bias){
	vec4 sP = (wP);
       sP = shadowModelView * sP;
       sP = shadowProjection * sP;
       sP /= sP.w;

  bias = 1.0 / (mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95);

  sP.xy *= bias;
  sP.z /= max(far / shadowDistance, 1.0);
  sP = sP * 0.5 + 0.5;

	return sP.xyz;
}
/*
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
*/
/*
float texture2DShadow(in sampler2D sampler, in vec3 coord){
  float pixel = shadowPixel * 0.25;
  coord.xy = round(coord.xy * shadowMapResolution) * shadowPixel;

  vec4 smoothShadow = vec4(step(coord.z, texture2D(sampler, coord.xy + vec2( pixel,  pixel)).x),
                           step(coord.z, texture2D(sampler, coord.xy + vec2( pixel, -pixel)).x),
                           step(coord.z, texture2D(sampler, coord.xy + vec2(-pixel,  pixel)).x),
                           step(coord.z, texture2D(sampler, coord.xy + vec2(-pixel, -pixel)).x));

  return dot(vec4(0.25), smoothShadow);
}
*/
const vec2 TRLB[4] = vec2[4](vec2(1.0, 1.0),
                             vec2(-1.0, 1.0),
                             vec2(1.0, -1.0),
                             vec2(-1.0, -1.0)
                            );

#ifdef Enabled_ScreenSpace_Shadow
float ScreenSpaceShadow(in vec3 lightPosition, in vec3 vP){
  int steps = 8;
  float istep = 1.0 / steps;

  float sss = 0.0;

  float thickness = 0.0625 + 0.05 * (1024.0 / shadowMapResolution);

  float dither = R2sq(texcoord * resolution - vec2(frameCounter * 0.05) - jittering);

  vec3 lightVector = normalize(lightPosition) * istep;
       lightVector = lightVector / shadowMapResolution * 700.0;// * max(3.0, length(vP.xyz));
  vec3 newVector = vP.xyz - lightVector * vP.z * shadowMapResolution / 4096.0 * 0.125;

  //lightVector *= 4.0;
  newVector -= lightVector * dither;

  for(int i = 0; i < steps; i++){
    newVector += lightVector;

    vec3 testPoint = newVector;

    vec3 testCoord = nvec3(gbufferProjection * vec4(testPoint, 1.0)).xyz * 0.5 + 0.5;
    if(floor(testCoord.xy) != vec2(0.0)) break;

    vec3 samplePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(testCoord.xy, texture(depthtex0, testCoord.xy).x) * 2.0 - 1.0));

    if(samplePosition.z > testPoint.z && samplePosition.z < testPoint.z + thickness) sss = 1.0;
  }

  return 1.0 - sss;
}
#endif

vec4 CalculateShading(in sampler2DShadow mainShadowTex, in sampler2DShadow secShadowTex, in vec4 wP, in float preShadeShadow){
  float d = length(wP.xyz);

  if(d > shadowDistance) return vec4(0.0);

  //float diffthresh = shadowPixel * (d / far) * 1.0 + 0.00012;

  vec3 shading = vec3(0.0);

  float bias = 0.0;

  vec3 shadowPosition = wP2sP(wP, bias);
  bias *= 0.125;

  const float bias_pix = 0.0003;
	vec2 bias_offcenter = abs(shadowPosition.xy * 2.0 - 1.0);
  //diffthresh = shadowPixel * ((d / far) * 2.0 + length(texcoord * 2.0 - 1.0) + 1.0);
  //float diffthresh = shadowPixel + (1.0 - preShadeShadow) * shadowPixel * 0.25 * d * shadowPixel;
  float diffthresh = 1.0 + preShadeShadow * (1.0 + d * 0.04);
        diffthresh *= shadowPixel;

  shading = vec3(1.0);

  if(floor(shadowPosition.xyz) == vec3(0.0)){
    float dither = R2sq(texcoord * resolution - (vec2(frameCounter) * 0.5 - jittering) * 0.05) * 2.0 * Pi;
    mat2 rotate = mat2(cos(dither), -sin(dither), sin(dither), cos(dither));

    #if Enabled_Soft_Shadow == OFF
      //shading = vec3(shadow2D(mainShadowTex, shadowPosition - vec3(0.0, 0.0, diffthresh)).x);
      shading = vec3(shadow2D(mainShadowTex, shadowPosition - vec3(0.0, 0.0, diffthresh)).x);
    #elif Enabled_Soft_Shadow == Simple
      shading = vec3(0.0);

      for(float i = -1.0; i <= 1.0; i += 1.0){
        for(float j = -1.0; j <= 1.0; j += 1.0){
          shading += vec3(shadow2D(mainShadowTex, shadowPosition - vec3(shadowPixel * vec2(i, j) * rotate, diffthresh)).x);
        }
      }

      shading /= 9.0;
    #elif Enabled_Soft_Shadow == PCSS
    shading = vec3(0.0);

    float blocker = 0.0;
    float minShadowDepth = 1.0;
    float receiver = shadowPosition.z - diffthresh * 1.0;

    int blockerCount = 0;

    //for(float i = -1.0; i <= 1.0; i += 1.0){
      //for(float j = -1.0; j <= 1.0; j += 1.0){
    for(int i = 0; i < 4; i++){
      vec2 blockerSearch = shadowPosition.xy + shadowPixel * TRLB[i] * 4.0 * rotate;

      //float shadowMapDepth = 1.0 - texture2D(shadowcolor0, blockerSearch).z;

      vec4 depthGather = textureGather(shadowcolor0, blockerSearch);
      //float shadowMapDepth = max(depthGather.x, maxComponent(depthGather.yzw));
      float shadowMapDepth = dot(depthGather, vec4(0.25));

      //float depthT = texture2D(shadowcolor0, blockerSearch + vec2(0.0, 1.0) * shadowPixel * bias).x;
      //float depthL = texture2D(shadowcolor0, blockerSearch + vec2(-1.0, 0.0) * shadowPixel * bias).x;
      //float depthR = texture2D(shadowcolor0, blockerSearch + vec2(1.0, 0.0) * shadowPixel * bias).x;
      //float depthB = texture2D(shadowcolor0, blockerSearch + vec2(0.0, -1.0) * shadowPixel * bias).x;

      //float shadowMapDepth = max(depthT, max(depthL, max(depthR, depthB)));
      //float shadowMapDepth = (depthT + depthL + depthR + depthB) * 0.25;

      minShadowDepth = min(minShadowDepth, shadowMapDepth);
      blocker += minShadowDepth;

      //blocker += min(receiver, depthMapDepth);

      //if(receiver >= shadowMapDepth){
      //  blocker += shadowMapDepth;
        blockerCount++;
      //}
      //}
    //}
    }

    blocker /= blockerCount;
    //blocker = min(receiver, texture2D(shadowcolor0, shadowPosition.xy).z);

    float penumbra = (receiver - blocker) / blocker;

    //shading = vec3(penumbra * 1.0);

    float radius = clamp(penumbra, 0.0625, 2.0) * 8.0;
          //radius *= 0.129;

    //shadowPosition.xy += R2sq2[int(mod(frameCounter, 16))] * shadowPixel * 0.125;

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 coord = shadowPosition.xy + (vec2(i, j) - vec2(j, i) * dither * 0.05) * rotate * bias * shadowPixel;
        shading += vec3(shadow2D(mainShadowTex, vec3(coord, shadowPosition.z - diffthresh)).x);
      }
    }

    shading /= 9.0;
    //shading = vec3(penumbra);
    //shading.b = shadow2D(mainShadowTex, shadowPosition - vec3(0.0, 0.0, diffthresh)).x;

    //if(blockerCount < 1) shading = vec3(1.0);
    //else if(blockerCount == 4) shading = vec3(0.0);

    #endif

    #ifdef Enabled_ScreenSpace_Shadow
      vec3 vP = mat3(gbufferModelView) * wP.xyz;
      shading *= (ScreenSpaceShadow(shadowLightPosition, vP));
    #endif

    #if CalculateShadingColor == 1
    vec4 colorShading = texture2D(shadowcolor1, shadowPosition.xy);
    //shading = mix(vec3(1.0), colorShading.rgb, step(0.1, colorShading.a)) * shading;

    //shading = colorShading.rgb;

    //shading = vec3(scattering);

    //shading = vec3(colorShading.a - 0.9) * 10.0;
    #endif
//    shading = shadowPosition;

    /*
    shading = vec3(float(texture2D(tex, shadowPosition.xy).z + diffthresh > shadowPosition.z));

    if(colorshading){
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
    */
  }

  //shading = mix(vec3(1.0), shading, clamp01((-d + shadowDistance - 16.0) / 32.0));
  //shading *= texture2D(colortex2, texcoord).a;

  return vec4(shading, clamp01((-d + shadowDistance - 16.0) / 32.0));
}

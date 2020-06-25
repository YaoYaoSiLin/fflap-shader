#version 130

#define SSAO_Steps 16    //[8 16 24]

#define gdepth colortex1
#define composite colortex3
#define gnormal colortex2
#define gaux2 colortex5

uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;

uniform sampler2D shadowtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 shadowModelViewInverse;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

vec2 resolution = vec2(viewWidth, viewHeight);

uniform int frameCounter;

in vec2 texcoord;

#include "../libs/common.inc"
#include "../libs/jittering.glsl"
#include "../libs/dither.glsl"

float GetDepth(in vec2 coord){
  float depth = texture(depthtex0, coord).x;
  float depthParticle = texture(gaux2, coord).x;

  if(0.0 < depthParticle && depthParticle < depth) depth = depthParticle;
  return depth;
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

#define SHADOW_MAP_BIAS 0.9

const int   shadowMapResolution = 2048;   //[512 768 1024 1536 2048 3072 4096]
const float shadowDistance = 140.0;

float shadowPixel = 1.0 / float(shadowMapResolution);

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

vec3 wP2sP(in vec4 wP, out float bias){
	vec4 sP = shadowModelView * wP;
       sP = shadowProjection * sP;
       sP /= sP.w;

  bias = 1.0 / (mix(1.0, length(sP.xy), SHADOW_MAP_BIAS) / 0.95);

  sP.xy *= bias;
  sP.z /= max(far / shadowDistance, 1.0);
  sP = sP * 0.5 + 0.5;

	return sP.xyz;
}

vec2 GetCoord(in vec2 coord){
  //vec2 resolution = vec2(viewWidth, viewHeight);

  //coord = round(coord * resolution * 0.25) / resolution * 4.0;
  coord = coord * 2.0 - 1.0;
  float bias = 1.0 / (mix(1.0, length(coord), SHADOW_MAP_BIAS) / 0.95);
  coord = coord * bias * 0.5 + 0.5;

  float dither = bayer_16x16(texcoord, resolution);

  return coord + dither * 0.5 * shadowPixel;
}

void main() {
  #if 1
  vec3 color = vec3(0.0);

  if(texcoord.x < 0.5 && texcoord.y < 0.5){
    vec2 coord = GetCoord(texcoord*2.0);

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 direction = vec2(i, j) * shadowPixel;

        color += texture2D(shadowcolor1, coord + direction).rgb;
      }
    }

    color *= 1.0 / 9.0;
  }
  if(texcoord.x > 0.5 && texcoord.y < 0.5){
    vec2 coord = GetCoord(texcoord*2.0-vec2(1.0,0.0));
    //color = normalDecode(texture2D(shadowcolor0, coord).gb);
    color = texture2D(shadowcolor0, coord).rgb * 2.0 - 1.0;
    color = mat3(shadowModelView) * color;
    color = color * 0.5 + 0.5;
  }
  if(texcoord.x < 0.5 && texcoord.y > 0.5){
    vec2 coord = GetCoord(texcoord*2.0-vec2(0.0,1.0));
    color.r = texture(shadowtex0, coord).r;
  }
  //if()

  //float depth = GetDepth(texcoord);

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color, 0.0);
  #else
  vec2 resolution = vec2(viewWidth, viewHeight);
  vec2 pixel = 1.0 / resolution;

  float renderScale = 0.5;

  vec2 coord = (texcoord);

  vec3 color = vec3(0.0);
  float alpha = 0.0;
  float depth = GetDepth(coord);

  vec3 normalSurface = normalDecode(texture2D(composite, coord).xy);
  vec3 normal = normalSurface;

  int mask = int(round(texture2D(gdepth, coord).z * 255.0));
  bool sky = texture2D(gdepth, coord).z > 0.99999;

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0);
       vP /= vP.w;
  float viewLength = length(vP.xyz);
  vec4 wP = gbufferModelViewInverse * vP;

  vec2 jitteringDither = frameCounter * 0.05 - jittering;
  float dither = R2sq(coord * resolution * renderScale + jitteringDither * renderScale);

  float ao = 0.0;
  vec3 shadowMapColor = vec3(0.0);

  if(!sky){
    vP.xyz -= vP.xyz * (-vP.z * 0.001);

    float thickness = 3.0;
    float t2 = 0.0625 * thickness * 6.0;

    float isteps = 1.0 / float(SSAO_Steps);

    for(int i = 0; i < SSAO_Steps; i++){
      float r = (i + dither) * isteps * 2.0 * Pi;
      vec2 offset = vec2(cos(r), sin(r)) / viewLength / vec2(aspectRatio, 1.0);

      vec2 aoCoord = clamp(coord + offset * 0.0375 * thickness, pixel, 1.0 - pixel);

      vec3 vP2 = nvec3(gbufferProjectionInverse * nvec4(vec3(aoCoord, GetDepth(aoCoord)) * 2.0 - 1.0));

      vec3 lightPosition = (vP2 - vP.xyz);
      float lightLength = length(lightPosition);

      float light = clamp01(dot(normalize(lightPosition), normal) / (lightLength + 0.001));

      lightLength -= clamp01(1.0 / dot(normalize(lightPosition), normal));
      ao += 1.0 - light * clamp01(t2 - lightLength);
    }

    ao *= isteps;

    //RSM

    float bias = 0.0;
// + (frameCounter * 0.05 - jittering) * 0.4)
    float dither = R2sq(texcoord * resolution*0.5);
    //float bayer32Dither = bayer_32x32(floor(texcoord*0.5*resolution+frameCounter*0.025), vec2(1.0));
    mat2 rotate = mat2(cos(dither), -sin(dither), sin(dither), cos(dither));

    int rsmsteps = 16;
    float invrsmsteps = 1.0 / float(rsmsteps);

    //wP.xyz = wP2sP(wP, bias);
    //vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;
    //wP.xyz -= worldNormal * 0.5;

    vec3 lightPosition2 = wP2sP(wP, bias);



/*
    //for(int i = 0; i < rsmsteps; i++){
    //  float r = (float(i) + dither) * invrsmsteps * 2.0 * Pi * 2.0;
    //  vec2 offset = vec2(sin(r), cos(r));
    for(float i = -1.0; i <= 1.0; i += 0.5){
      for(float j = -1.0; j <= 1.0; j += 0.5){
        vec2 offset = rotate * vec2(i, j);
        offset *= abs(offset);
        offset *= shadowPixel * 2048.0;

        vec4 worldPosition = wP;
        worldPosition.xy += offset;

        vec3 lightPosition = wP2sP(worldPosition, bias);

        //vec3 lightPosition = lightPosition2;
             //lightPosition.xy += offset * shadowPixel * Pi * 2.0 * bias / max(6.28, viewLength) * 10.0;

        float shadow = shadow2D(shadowtex0, lightPosition.xyz - vec3(0.0, 0.0, shadowPixel)).x;
              shadow *= step(0.9, shadow);

        //float shadowMapDepth = texture(shadowcolor0, lightPosition.xy).x;
        //float shadow = (lightPosition.z - shadowPixel - shadowMapDepth) / shadowMapDepth;
        //      shadow = 1.0;

        vec3 shadowMapNormal = normalDecode(texture2D(shadowcolor0, lightPosition.xy).yz);

        vec3 halfVector = mat3(gbufferModelView) * vec3(offset, 0.0);
        vec3 nhalfVector = normalize(halfVector);

        float ndotl = max((dot(shadowMapNormal, -nhalfVector) - 0.1) / 0.9, 0.0);
              ndotl *= max(dot(normalSurface, nhalfVector), 0.0);
              ndotl = pow(ndotl, 0.2);
              ndotl /= pow(length(halfVector), 4.0) + 0.0001;

        shadowMapColor += texture2D(shadowcolor1, lightPosition.xy).rgb*shadow*clamp01(ndotl);
    //  }
    }
    }
*/
    //shadowMapColor *= 1.0/25.0;
    //shadowMapColor *= invrsmsteps;


  }else{
    ao = 0.0;
  }

  color = vec3(ao, 0.0, 0.0);
  color = shadowMapColor;

  gl_FragData[0] = vec4(color, depth);
  #endif
}

#version 130

#define TAA_Color_Sampler_Size 0.55   	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]
#define TAA_Depth_Sampler_Size 0.55   	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]

const bool gaux4Clear = false;

uniform sampler2D gdepth;
uniform sampler2D gaux2;
uniform sampler2D gaux4;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

#define lastColorSampler gaux4
#define colorSampler gaux2

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform float viewWidth;
uniform float viewHeight;

uniform int frameCounter;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "libs/common.inc"
#include "libs/jittering.glsl"

vec3 encodePalYuv(vec3 rgb) {
    rgb = rgb2L(rgb);

    return vec3(
        dot(rgb, vec3(0.299, 0.587, 0.114)),
        dot(rgb, vec3(-0.14713, -0.28886, 0.436)),
        dot(rgb, vec3(0.615, -0.51499, -0.10001))
    );
}

vec3 decodePalYuv(vec3 yuv) {
    vec3 rgb = vec3(
        dot(yuv, vec3(1., 0., 1.13983)),
        dot(yuv, vec3(1., -0.39465, -0.58060)),
        dot(yuv, vec3(1., 2.03211, 0.))
    );

    return L2rgb(rgb);
}

vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    // note: only clips towards aabb center (but fast!)
    vec3 center  = 0.5 * (maximum + minimum);
    vec3 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    vec3 offset = color - center;

    vec3 ts = abs(extents / (offset + 0.0001));
    float t = clamp(minComponent(ts), 0.0, 1.0);
    return center + offset * t;
}

void main(){
  vec2 unJitterUV = texcoord - haltonSequence_2n3[int(mod(frameCounter, 16))] * pixel * 0.6;

  vec3 color = encodePalYuv(texture2D(gaux2, unJitterUV).rgb);

  vec3 maxColor = vec3(-1.0);
  vec3 minColor = vec3(1.0);

  vec3 sharpen = vec3(0.0);

  float depth = 1.0;

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec3 colortemp = encodePalYuv(texture2D(gaux2, unJitterUV + vec2(i, j) * pixel * TAA_Color_Sampler_Size).rgb);

      maxColor = max(maxColor, colortemp);
      minColor = min(minColor, colortemp);

      sharpen += colortemp;

      depth = min(depth, texture2D(depthtex0, texcoord + vec2(i, j) * pixel).x);
    }
  }

  sharpen = (sharpen - color) * 0.125;

  //maxColor = encodePalYuv(maxColor);
  //minColor = encodePalYuv(minColor);

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0);
       vP /= vP.w;

  vec4 pvP = vP;
       pvP = gbufferModelViewInverse * pvP;
       pvP.xyz += cameraPosition - previousCameraPosition;
       pvP = gbufferPreviousModelView * pvP;
       pvP = gbufferPreviousProjection * pvP;
       pvP /= pvP.w;

  vec2 previousCoord = pvP.xy * 0.5 + 0.5;
       previousCoord = texcoord - previousCoord;
       previousCoord = -previousCoord + texcoord;

  float blendWeight = sqrt(dot(0.5 - abs(fract(previousCoord * resolution) - 0.5), vec2(1.0))) * float(floor(previousCoord) == vec2(0.0)) * 0.9;

  float mixRate = texture2D(gaux4, previousCoord).a * float(floor(previousCoord) == vec2(0.0));

  vec3 lastColor = encodePalYuv(texture2D(gaux4, previousCoord).rgb);
       lastColor = mix(color, lastColor, mixRate * 0.995);

  vec2 uvNear = ((texcoord - 0.5) - (unJitterUV - 0.5)) * resolution - ((previousCoord - 0.5) - (unJitterUV - 0.5)) * resolution;

  float weightStatic = length(((texcoord - 0.5) - (unJitterUV - 0.5)) * resolution * 2.0);
  float weightMotion = length(((previousCoord - 0.5) - (unJitterUV - 0.5)) * resolution * 2.0);

  float maxWeight = clamp01(max(weightMotion, weightStatic) * 0.5);
  float minWeight = 1.0 - clamp01(length(((previousCoord - 0.5) - (unJitterUV - 0.5)) * 2.0) * 32.0);

  float colorDiff = dot(vec3(0.3333), abs(texture2D(gaux2, texcoord).rgb - texture2D(gaux2, unJitterUV).rgb));

  vec3 lastColorNear = mix(color, lastColor, clamp01(64.0 * colorDiff + 0.1) * 0.9 * minWeight);

  if(int(round(texture2D(gdepth, texcoord).z * 255.0)) != 254){
    //maxColor = max(maxColor, lastColorNear);
    //minColor = min(minColor, lastColorNear);
  }

  vec3 antialiased = clipToAABB(lastColor, minColor, maxColor);
       //antialiased += (antialiased - sharpen) * 0.025 * weight * 50;
       //antialiased = mix(antialiased, color, 0.001 + 0.009 * lastColorNear);
       antialiased += (antialiased - sharpen) * 0.0033 * lastColorNear;
       //antialiased = mix(color, antialiased, blendWeight);

  antialiased = decodePalYuv(antialiased);
  //antialiased += (antialiased - decodePalYuv(sharpen)) * 0.25 * weight * 2.0;

  //float lum = getLum(texture2D(gaux2, vec2(0.5)).rgb);

/* DRAWBUFFERS:57 */
  gl_FragData[0] = vec4(antialiased, 1.0);
  gl_FragData[1] = vec4(antialiased, 1.0);
}

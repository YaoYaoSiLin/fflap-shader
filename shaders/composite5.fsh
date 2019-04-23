#version 130

#define TAA_Color_Sampler_Size 0.55   	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]
#define TAA_Depth_Sampler_Size 0.55   	//[0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9]

const bool gaux2Clear = false;
const bool gaux4Clear = false;

uniform sampler2D gdepth;
uniform sampler2D gaux2;
uniform sampler2D gaux4;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

#define lastColorSampler gaux4
#define colorSampler gaux2

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform float viewWidth;
uniform float viewHeight;
uniform float far;
uniform float near;
uniform int frameCounter;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "libs/jittering.glsl"

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

float getLum(in vec3 color){
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 rgb2L(in vec3 color){
  return pow(color, vec3(2.2));
}

vec3 L2rgb(in vec3 color){
  return pow(color, vec3(1.0 / 2.2));
}

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

float minComponent( vec3 a )
{
    return min(a.x, min(a.y, a.z) );
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
  //if(int(texture2D(gdepth, texcoord).z * 255) == 254) discard;

  vec3 color = encodePalYuv(texture2D(gaux2, texcoord).rgb);
  vec3 minColor = vec3(1.0);
  vec3 maxColor = vec3(-1.0);
  vec3 avgColor = vec3(0.0);

  float depth = 1.0;

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      float depthtemp = texture2D(depthtex0, texcoord + vec2(i, j) * pixel).x;

      depth = min(depth, depthtemp);

      vec3 c = encodePalYuv(texture2D(gaux2, texcoord + vec2(i, j) * pixel * TAA_Color_Sampler_Size).rgb);

      minColor = min(minColor, c);
      maxColor = max(maxColor, c);
      avgColor += c;
    }
  }

  avgColor -= color;
  avgColor *= 0.125;

  vec4 pvP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0);
       pvP /= pvP.w;
       pvP = gbufferModelViewInverse * pvP;
       pvP.xyz += cameraPosition - previousCameraPosition;
       pvP = gbufferPreviousModelView * pvP;
       pvP = gbufferPreviousProjection * pvP;
       pvP /= pvP.w;

  vec2 previousCoord = pvP.xy * 0.5 + 0.5;
       previousCoord = texcoord - previousCoord;
       previousCoord = -previousCoord + texcoord;
       //previousCoord = texcoord;

  vec3 lastColor = encodePalYuv(texture2D(lastColorSampler, previousCoord).rgb);
       //lastColor = mix(color, lastColor, lastMixRate);
       //lastColor = mix(lastColor, color, 0.005);
       //if(texcoord.x < 0.5)lastColor += (lastColor - avgColor) * 0.25;

  if(floor(previousCoord) == vec2(0.0)){
    //vec3 lastImege = mix(lastColor, color, (cla));

    //minColor = min(lastImege, minColor);
    //maxColor = max(lastImege, maxColor);
  }

  vec3 antialiased = clipToAABB(lastColor, minColor, maxColor);
       //antialiased = mix(antialiased, lastColor, 0.05);
       //antialiased = mix(antialiased, color, 0.005);
       //if(texcoord.x < 0.5)antialiased += (antialiased - avgColor) * 0.0025;

  bool isHand = int(round(texture2D(gdepth, texcoord).z * 255.0)) == 254.0;

  if(isHand) antialiased = color;
  antialiased = decodePalYuv(antialiased);

  //if(lastMixRate )


  //color = decodePalYuv(color);

/* DRAWBUFFERS:57 */
  gl_FragData[0] = vec4(antialiased, 1.0);
  gl_FragData[1] = vec4(antialiased, 1.0);
}

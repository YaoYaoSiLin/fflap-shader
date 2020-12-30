#version 130

uniform sampler2D gcolor;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform usampler2D gaux3;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;

uniform float viewWidth;
uniform float viewHeight;

uniform int frameCounter;

in vec2 texcoord;

in vec3 sunLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "../libs/common.inc"
#include "../lib/packing.glsl"


vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 CalculateIndirect(in sampler2D sampler, in vec2 uv){
  vec4 indirect = vec4(0.0);

  const float radius = 1.0;

  //uv = floor(uv * resolution) * pixel;

  float depth = linearizeDepth(texture(depthtex0, uv).x);
  vec3 normal = normalDecode(texture2D(composite, uv).xy);

  for(float i = -radius; i <= radius; i += 1.0){
    for(float j = -radius; j <= radius; j += 1.0){
      vec2 coord = uv + vec2(i, j) * pixel * 4.0;

      vec3 sampleColor = decodeGamma(texture2D(gaux1, coord).rgb);
      float sampleDepth = linearizeDepth(texture(depthtex0, coord).x);
      vec3 sampleNormal = normalDecode(texture2D(composite, coord).xy);

      vec3 tN = sampleNormal - normal;
      float distN = max(0.0, -dot(sampleNormal, normal));
      float weightN = max(0.0, dot(sampleNormal, normal));

      float distD = pow2((sampleDepth - depth) / 0.00005);
      float weightD = min(1.0, exp(-distD));

      float weight = weightD * weightN;

      indirect += vec4(sampleColor, 1.0) * weight;
    }
  }

  indirect.rgb /= indirect.a;
  //indirect /= pow2(radius * 2.0 + 1.0);

  return indirect.rgb;
}

void main(){
  vec3 color = texture2D(gaux2, texcoord).rgb;
  
  color = decodeGamma(color) * decodeHDR;

  vec3 albedo = decodeGamma(texture2D(gcolor, texcoord).rgb);
  
  //vec3 indirect = decodeGamma(texture2D(gaux1, texcoord).rgb);//decodeGamma(vec3(unpackUnorm2x16(texture(gaux3, texcoord).r).y,
									//		             unpackUnorm2x16(texture(gaux3, texcoord).g).y,
									//		             unpackUnorm2x16(texture(gaux3, texcoord).b).y));

  vec3 indirect = decodeGamma(texture2D(gaux1, texcoord).rgb);//CalculateIndirect(gaux1, texcoord);

  if(texture(depthtex0, texcoord).x < 0.9999)
  //color = indirect;
  color += invPi * indirect * albedo * sunLightingColorRaw * 3.0;
  
  color = encodeGamma(color * encodeHDR);
  
  /* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color, 1.0);
}

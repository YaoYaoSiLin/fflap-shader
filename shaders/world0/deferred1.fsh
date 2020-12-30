#version 130

uniform sampler2D gaux2;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;

uniform float viewWidth;
uniform float viewHeight;
uniform float near;
uniform float far;

uniform int frameCounter;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define pow2(x) (x * x)

float linearizeDepth(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}

float BlurAmbientOcclusion(){
  float ao = 0.0;

  float radius = 2.0;

  float centerDepth = linearizeDepth(texture(depthtex0, texcoord).x);

  float totalWeight = 0.0;

  for(float i = -radius; i <= radius; i += 1.0){
  //  for(float j = -radius; j <= radius; j += 1.0){
      vec2 direction = vec2(i, 0.0) * pixel * 1.0;

      vec2 uv = texcoord + direction;
      if(floor(uv) != vec2(0.0)) continue;

      float currentDepth = linearizeDepth(texture(depthtex0, uv).x);
      if(currentDepth >= 0.9999) continue;

      //float weight = step(0.0, currentDepth - centerDepth);
      float depthDistance = pow2(max(0.0, currentDepth - centerDepth));

      float weight = exp(-depthDistance * 4000.0 * 1.338);

      float aoSample = pow(texture(gaux2, uv * 0.5).x, 2.2);

      ao += aoSample * weight;
      totalWeight += weight;
  //  }
  }

  ao /= totalWeight;
  //ao = max(0.0, 1.0 - ao * 3.0);

  return pow(ao, 1.0 / 2.2);
}

void main(){
  vec4 color = vec4(vec3(0.0), 1.0);

  color.r = BlurAmbientOcclusion();

  /* DRAWBUFFERS:5 */
  gl_FragData[0] = color;
}

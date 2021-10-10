#define Continuum2_Texture_Format

#define Enabled_ScreenSpaceReflection
  #define HightQualityReflection 8    //[-1 0 4 8 12 16 20 24 28 32]
  #define Half_Scale_Reflection
  #define SSR_Start 0.1               //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.19]
  #define SSR_Steps 20                //[15 20 25 30 35 40]

vec2 P2C(in vec3 P){
  return nvec3(gbufferProjection * nvec4(P)).xy * 0.5 + 0.5;
}

vec3 P2UV(in vec4 P){
  return nvec3(gbufferProjection * P).xyz * 0.5 + 0.5;
}

vec3 P2UV(in vec3 P){
  return P2UV(nvec4(P)).xyz;
}

#if defined(Enabled_ScreenSpaceReflection)
//float view2LinearizeDepth(in vec3 v){
//  return linearizeDepth(nvec3(gbufferProjection * nvec4(v)).z * 0.5 + 0.5);
//}

#define SSR_3D 0
#define SSR_2D 1

#define SSR_Type SSR_3D

vec4 rayMarching(in vec3 rayOrigin, in vec3 rayDirection, vec3 normal){
  vec2 hitUV = vec2(0.0);

  int count;

  int steps = 20;
  float invsteps = 1.0 / float(steps);

  //rayDirection *= 20.0 * invsteps;

  float rayEnd = 400.0;
  float rayStep = pow(rayEnd, invsteps);

  vec3 testPosition = rayOrigin;
  vec3 direction = rayDirection;// * mix(dither, 1.0, 0.7071) * 0.5;

  vec3 viewDirection = normalize(rayOrigin);

  for(int i = 0; i < steps; i++){
    testPosition += direction;

    vec2 uv = P2C(testPosition);
    //if(abs(uv.x - 0.5) > 0.5 || abs(uv.y - 0.5) > 0.5) break;
    if(max(abs(uv.x - 0.5), abs(uv.y - 0.5))  > 0.5) break;

    float sampleDepth = texture2D(depthtex0, uv).x;
    vec3 samplePosition = nvec3(gbufferProjectionInverse * vec4(vec3(uv, sampleDepth) * 2.0 - 1.0, 1.0));

    float testDepth = linearizeDepth(P2UV(testPosition).z);
    float frontDepth = linearizeDepth(sampleDepth);
    float difference = testDepth - frontDepth;

    vec3 h = normalize(normalize(testPosition) - viewDirection);

    if(testDepth > frontDepth){
      float backDepth = (1.0 / far + frontDepth + testDepth) * far * 0.000125 + near;
      //(1.0 + frontDepth * 256.0) * (1.0 / 2048.0)

      float toSurface = (testDepth - frontDepth) / frontDepth;

      if(toSurface < backDepth){
        return vec4(testPosition, 1.0);
      }

      testPosition -= direction;
      direction *= 0.05;
    }else{
      direction *= rayStep;
    }
  }

  return vec4(vec3(0.0), 0.0);
}

vec4 ScreenSpaceReflection(in vec3 viewVector, in vec3 rayDirection, inout vec3 hitPosition, in float dither, vec3 normal){
  vec4 color = vec4(0.0);

  vec3 testPoint = viewVector;

  rayDirection *= mix(dither, 1.0, 0.7071) * 0.5;
  //testPoint += rayDirection * dither * 4.0;

  vec4 rayPosition = rayMarching(testPoint, rayDirection, normal);

  vec2 coord = nvec3(gbufferProjection * rayPosition).xy * 0.5 + 0.5;

  if(rayPosition.w > 0.0) {
    hitPosition = rayPosition.xyz;

    color = texture2D(gaux2, coord.xy);
    color.rgb = decodeGamma(color.rgb) * decodeHDR;
    /*
    vec2 specularPackge = unpack2x8(texture(composite, coord.xy).b);
  	float smoothness = specularPackge.x; float metallic = specularPackge.y;
    vec2 lightmapPackge = unpack2x8(texture(gdepth, coord.xy).x);
    float torchLightMap = lightmapPackge.x; float skyLightMap = lightmapPackge.y;

    if(bool(step(isMetallic, metallic))) color.rgb = decodeGamma(texture2D(gcolor, coord.xy).rgb) * skyLightingColorRaw * step(0.7, skyLightMap);

    color.a = 1.0;
    */
  }

  return color;
}
#endif

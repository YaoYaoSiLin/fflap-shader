#define Continuum2_Texture_Format

#define Enabled_ScreenSpaceReflection
  #define HightQualityReflection 8    //[-1 0 4 8 12 16 20 24 28 32]
  #define Half_Scale_Reflection
  #define SSR_Start 0.1               //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.19]
  #define SSR_Steps 20                //[15 20 25 30 35 40]

vec2 P2C(in vec3 P){
  return nvec3(gbufferProjection * nvec4(P)).xy * 0.5 + 0.5;
}

vec3 P2UV(in vec3 P){
  return nvec3(gbufferProjection * nvec4(P)).xyz * 0.5 + 0.5;
}

#if defined(Enabled_ScreenSpaceReflection)
//float view2LinearizeDepth(in vec3 v){
//  return linearizeDepth(nvec3(gbufferProjection * nvec4(v)).z * 0.5 + 0.5);
//}

#define SSR_3D 0
#define SSR_2D 1

#define SSR_Type SSR_3D

vec4 rayMarching(in vec3 rayDirection, in vec3 viewPosition, out vec3 hitPosition){
  vec2 hitUV = vec2(0.0);

  int count;

  int steps = 20;
  float invsteps = 1.0 / float(steps);

  float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale);

  //rayDirection *= 0.5 + dither * 0.5;
  //rayDirection *= mix(dither, 0.5, 0.7);

  float rayEnd = 600.0;
  float rayStep = pow(rayEnd, invsteps);

  vec3 testPosition = viewPosition;

  for(int i = 0; i < steps; i++){
    testPosition += rayDirection;

    vec2 uv = P2C(testPosition);
    if(floor(uv) != vec2(0.0)) break;

    float sampleDepth = (texture2D(depthtex0, uv).x);
    vec3 samplePosition = nvec3(gbufferProjectionInverse * vec4(vec3(uv, sampleDepth) * 2.0 - 1.0, 1.0));

    float testDepth = linearizeDepth(P2UV(testPosition).z);
    float forntDepth = linearizeDepth(sampleDepth);

    if(testDepth > forntDepth){
      float backDepth = linearizeDepth(sampleDepth + 0.001);

      if(testDepth - forntDepth < (1.0 + forntDepth * 256.0 * float(count + 1)) * (1.0 / 2048.0)){
        hitUV = uv;
        hitPosition = testPosition;
        count++;
        //if(count > 1) break;
      }

      testPosition -= rayDirection;
      rayDirection *= 0.05;
    }else{
      rayDirection *= rayStep;
    }
  }
  return vec4(hitUV, 0.0, 0.0);
}

vec4 raytrace(in vec3 viewVector, in vec3 rayDirection, out vec3 hitPosition, in float dither){
  vec4 color = vec4(0.0);

  vec3 testPoint = viewVector;

  rayDirection *= mix(dither, 0.5, 0.7);

  vec4 ray = rayMarching(rayDirection, testPoint, hitPosition);

  if(ray.x > 0.0 && ray.y > 0.0) {
    /*
    ray.xy = round(ray.xy * resolution) * pixel;

    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 4; j++){
        vec2 offset = vec2(i, j) - 1.5;
        float weight = gaussianBlurWeights(offset+0.001);

        color += texture2D(gaux2, ray.xy + offset * pixel);
       }
    }

    color /= 16.0;
*/
    color = texture2D(gaux2, ray.xy);

    color.a = 1.0;
  }

  return color;
}
#endif

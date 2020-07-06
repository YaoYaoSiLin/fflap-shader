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

vec4 rayMarching(in vec3 rayDirection, in vec3 viewPosition, out vec3 hitPosition, in float roughness, in float mipMap){
  vec2 hitUV = vec2(0.0);

  int count;

  //viewPosition -= rayDirection * 0.02 * dither;

  int steps = 20;
  float invsteps = 1.0 / float(steps);

  float dither = R2sq(texcoord * resolution);

  //rayDirection *= invsteps;
  //rayDirection *= lerq(dither, 1.0, 0.9);
  rayDirection *= 0.5;
  //rayDirection *= mix(dither, 1.0, 0.95);

  float depth = texture(depthtex0, texcoord).x;

  float rayEnd = 768.0;
  float rayStep = pow(rayEnd, invsteps);

  for(int i = 0; i < steps; i++){
    viewPosition += rayDirection;

    vec2 uv = P2C(viewPosition);
    if(floor(uv) != vec2(0.0)) break;

    float sampleDepth = (texture2D(depthtex0, uv).x);
    vec3 samplePosition = nvec3(gbufferProjectionInverse * vec4(vec3(uv, sampleDepth) * 2.0 - 1.0, 1.0));

    float testDepth = linearizeDepth(P2UV(viewPosition).z);
    float forntDepth = linearizeDepth(sampleDepth);

    if(testDepth > forntDepth){
      //if(testDepth - linearlizeSampleDepth < (exp2(8.0) / 2048.0) / float(count + 1)){
        float backDepth = linearizeDepth(sampleDepth + 0.001);

      if(testDepth - forntDepth < (1.0 + forntDepth * 511.0 / float(count + 1)) * (1.0 / 2048.0)){
        hitUV = uv;
        hitPosition = samplePosition;
        count++;
        if(count > 1) break;
      }


      viewPosition -= rayDirection;
      rayDirection *= 0.01;
    }else{
      rayDirection *= rayStep;
    }
  }
  return vec4(hitUV, 0.0, 0.0);
}

vec4 raytrace(in vec3 viewVector, in vec3 rayDirection, out vec3 hitPosition, in float roughness, in float mipMap){
  vec4 color = vec4(0.0);

  vec3 testPoint = viewVector;

  //rayDirection *= 0.5;

  vec4 ray = rayMarching(rayDirection, testPoint, hitPosition, roughness, mipMap);

  //float radius = mipMap * length(ray.xy - texcoord);
  //      radius = clamp(log2(radius * resolution.x), 0.0, 6.2831);
  /*
  float dither = R2sq(texcoord * resolution * SSR_Rendering_Scale);

  float steps = 8.0 / SSR_Rendering_Scale;
  float invsteps = 1.0 / steps;
  float frameIndex = mod(float(frameCounter), steps);

  float r = (frameIndex * invsteps + dither) * Pi * 2.0;
  vec2 offset = vec2(cos(r), sin(r)) * 0.01;
  */

  if(ray.x > 0.0 && ray.y > 0.0) {
    //color = texture2DLod(gaux2, ray.xy + jittering * pixel * SSR_Rendering_Scale * 0.0, radius);

    #if Surface_Quality == Low
    color = texture2DLod(gaux2, ray.xy, 0.0);
    hitPosition.z = texture2DLod(depthtex0, ray.xy, 0.0).x;
    #else
    //hitPosition.xy = ray.xy;
    color = texture2DLod(gaux2, ray.xy, 0.0);
    //color.a = 1.0;
    #endif
    /*
    for(int i = 0; i < int(steps); i++){
      float r = (i * invsteps + dither) * 2.0 * Pi;
      hitPosition.z += texture(depthtex0, ray.xy + offset * radius * 0.0025).x;
    }

    hitPosition.z *= invsteps;
    */
  }

  return color;
}
#endif

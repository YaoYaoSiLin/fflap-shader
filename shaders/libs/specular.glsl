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

  for(int i = 0; i < 20; i++){
    viewPosition += rayDirection;

    vec2 uv = P2C(viewPosition);
    if(floor(uv) != vec2(0.0)) break;

    float sampleDepth = texture2D(depthtex0, uv).x;
    vec3 samplePosition = nvec3(gbufferProjectionInverse * vec4(vec3(uv, sampleDepth) * 2.0 - 1.0, 1.0));

    float testDepth = P2UV(viewPosition).z;

    float delta = testDepth - sampleDepth;

    if(0.0 < delta){
      //(float(1.0 + i) / 40.0)
      count++;
      if(count == 2 && length(viewPosition - samplePosition) < length(viewPosition) * (float(1 + i) / 40.0)){
        hitUV = uv;
        hitPosition = vec3(vec2(0.0), sampleDepth);
        break;
      }

      viewPosition -= rayDirection;
      rayDirection *= 0.01;
    }else{
      //rayDirection *= 1.0 + (40.0 / float(SSR_Steps)) * 0.2 + 2.0 * 2.0 * 2.0 * 0.02;
      rayDirection *= 1.5 + min(0.5, length(viewPosition) / 200.0);
    }
  }

  return vec4(hitUV, 0.0, 0.0);
}

vec4 raytrace(in vec3 viewVector, in vec3 rayDirection, out vec3 hitPosition, in float roughness, in float mipMap){
  vec4 color = vec4(0.0);

  vec3 testPoint = viewVector;

  rayDirection *= 0.5;

  vec4 ray = rayMarching(rayDirection, testPoint, hitPosition, roughness, mipMap);

  float radius = mipMap * length(ray.xy - texcoord);
        radius = clamp(log2(radius * resolution.x), 0.0, 6.2831);
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
    color = texture2DLod(gaux2, ray.xy, radius);
    hitPosition.z = texture2DLod(depthtex0, ray.xy, radius).x;
    #else
    color = texture2DLod(gaux2, ray.xy, 0.0);
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

#define Continuum2_Texture_Format

#define Default 0  //
#define Blur  1  //

#define Metal_Block_Reflection_Smoothness Default //[Default Blur]

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

  float dither = bayer_32x32(texcoord, resolution);

  #if SSR_Type == SSR_2D
  rayDirection = normalize(rayDirection);
  //viewPosition -= normalize(rayDirection - viewPosition) * 0.1;
  float depth = texture2D(depthtex0, texcoord).x;

  float rayLength = viewPosition.z + rayDirection.z * 256.0 > -near ?
                    (-near - viewPosition.z) / rayDirection.z : 256.0;

  vec3 rayEnd = viewPosition + rayDirection * rayLength;

  vec4 h0 = (gbufferProjection * nvec4(viewPosition));
  vec4 h1 = (gbufferProjection * nvec4(rayEnd));

  float k0 = 1.0 / h0.w,  k1 = 1.0 / h1.w;
  vec2 p0 = h0.xy * k0, p1 = h1.xy * k1;
  vec3 q0 = viewPosition * k0, q1 = rayEnd * k1;

  p1 += abs(dot(p1 - p0, p1 - p0)) < 0.0001 ? pixel : vec2(0.0);

  vec2 delta = (p1 - p0) * resolution;
  float sampleScaler = clamp01(1.0 - pow5(-dot(normalize(viewPosition), normalize(rayDirection - viewPosition))));
  float step = min(1.0 / abs(delta.x), 1.0 / abs(delta.y)) * 8.0 * (1.0 + sampleScaler * 1.0);

  float interpolationCounter = step;

  vec4 pqk = vec4(p0, q0.z, k0);
  vec4 dpqk = vec4(p1 - p0, q1.z - q0.z, k1 - k0) * step;

  pqk += bayer_16x16(texcoord, resolution) * dpqk * 0.001;

  float prevZMaxEstimate = viewPosition.z;

  bool intersected = false;

  //dpqk.xy *= 20.0;
  //hitUV = (pqk.xy + dpqk.xy - dpqk.xy * 0.5) * 0.5 + 0.5;


  for(int i = 0; i < 20; i++){
    if(interpolationCounter > 1.0 || intersected) break;
    interpolationCounter += step;

    pqk += dpqk;

    float rayZMin = prevZMaxEstimate;
    float rayZMax = pqk.z / pqk.w;

    if(rayZMin > rayZMax){
      float t = rayZMin;
      rayZMin = rayZMax;
      rayZMax = t;
    }

    float depth = texture2D(depthtex0, (pqk.xy - dpqk.xy * 0.5) * 0.5 + 0.5).x;
          depth = nvec3(gbufferProjectionInverse * vec4(depth, depth, depth, 1.0)).z;

    if(rayZMin < depth + 0.003 * 16.0 && rayZMax > depth + 0.003 * 16.0){
      hitUV = (pqk.xy - dpqk.xy * 0.5) * 0.5 + 0.5;
      intersected = true;
    }else{
      prevZMaxEstimate = rayZMax;
    }
  }
  #endif

  #if SSR_Type == SSR_3D
  int count;

  viewPosition -= rayDirection * 0.02 * dither;

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
        hitPosition = viewPosition;
        break;
      }

      viewPosition -= rayDirection;
      rayDirection *= 0.01;
    }else{
      //rayDirection *= 1.0 + (40.0 / float(SSR_Steps)) * 0.2 + 2.0 * 2.0 * 2.0 * 0.02;
      rayDirection *= 1.5 + min(0.5, length(viewPosition) / 200.0);
    }
  }
  #endif

  return vec4(hitUV, 0.0, 0.0);
}

vec4 raytrace(in vec3 viewVector, in vec3 rayDirection, out vec3 hitPosition){
  #if (Stage == ScreenReflection && defined(Half_Scale_Reflection)) || (Stage == Composite && !defined(Half_Scale_Reflection))
  //if(roughness < 0.01) return vec4(0.0);
  //else{

  int maxf = int(SSR_Steps / 10);
  if(maxf < 2) maxf = 2;

  vec4 color = vec4(0.0);

  //float f = pow5(1.0 - clamp01(-dot(normalize(viewVector), normal))) * max(256.0, far) * 0.0625 * 0.5;

  vec3 testPoint = viewVector;

  //rayDirection *= max(0.05, (length(viewVector)) / (256.0 * 256.0) * 2049.0);
  rayDirection *= 0.5;//dot(normalize(viewVector), normal)
  //rayDirection *= 1.0 + length(normal);

  int sr = 0;
  int count = 0;

  //float ndotv = 1.0 - clamp01(dot(-normalize(viewVector), normal));

  //dither = dither * 2.0 - 1.0;
  //dither *= 0.005;

  //float dither = bayer_16x16(texcoord + haltonSequence_2n3[int(mod(frameCounter, 16))] * pixel, resolution) - 0.5;

  //rayDirection *= 0.0625 + (mod(frameCounter, 16));

  vec4 ray = rayMarching(rayDirection, testPoint, hitPosition);
  if(ray.x > 0.0 && ray.y > 0.0) {
    color = texture2D(gaux2, ray.xy);
    //color.a *= 1.0 - pow5(length(ray.xy * 2.0 - 1.0));
  }
  //color.a = 1.0;

  //color /= min(1.0, float(count));

  return color;
  //}
  #else
  return vec4(0.0);
  #endif
}
#endif

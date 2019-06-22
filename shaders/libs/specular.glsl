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

#if defined(Enabled_ScreenSpaceReflection)
//float view2LinearizeDepth(in vec3 v){
//  return linearizeDepth(nvec3(gbufferProjection * nvec4(v)).z * 0.5 + 0.5);
//}

vec4 raytrace(in vec3 viewVector, in vec3 rayDirection, in vec3 normal, in float dither, in float ra){
  #if (Stage == ScreenReflection && defined(Half_Scale_Reflection)) || (Stage == Composite && !defined(Half_Scale_Reflection))
  //if(roughness < 0.01) return vec4(0.0);
  //else{

  float dist = length(viewVector);

  int maxf = int(SSR_Steps / 10);
  if(maxf < 2) maxf = 2;

  vec4 color = vec4(0.0);

  float f = pow5(1.0 - clamp01(-dot(normalize(viewVector), normal))) * max(256.0, far) * 0.0625 * 0.5;

  vec3 testPoint = viewVector + normal * f * 0.05;

  //rayDirection *= max(0.05, (length(viewVector)) / (256.0 * 256.0) * 2049.0);
  rayDirection *= 0.5 + f;//dot(normalize(viewVector), normal)
  //rayDirection *= 1.0 + length(normal);

  int sr = 0;
  int count = 0;

  float ndotv = 1.0 - clamp01(dot(-normalize(viewVector), normal));

  //dither = dither * 2.0 - 1.0;
  //dither *= 0.005;

  //float dither = bayer_16x16(texcoord + haltonSequence_2n3[int(mod(frameCounter, 16))] * pixel, resolution) - 0.5;

  for(int i = 0; i < SSR_Steps; i++){
    testPoint += rayDirection + rayDirection * dither;

    vec2 uv = P2C(testPoint);
    if(floor(uv) != vec2(0.0)) break;

    vec3 samplePosition = nvec3(gbufferProjectionInverse * nvec4(vec3(uv, texture2D(depthOpaque, uv).x) * 2.0 - 1.0));

    if(length(testPoint) > length(samplePosition)){
      sr++;
      if(sr == maxf){
        //if(texture2D(depthOpaque, uv).x > sky) break;
        if(length(testPoint - samplePosition) > length(testPoint) * (float(1 + i) / float(SSR_Steps) / float(maxf))) break;

        #if HightQualityReflection > 3
          vec4 colorIndex = vec4(0.0);

          for(int i = 0; i < int(HightQualityReflection); i++){
            float r = (1.0 + float(i) * 2.0 * 3.14159) / float(HightQualityReflection) + dither;
                  //r *= ra;

            vec3 n = vec3(cos(r), sin(r), 1.0);
                 //n.xy *= pow(roughness, 0.5);
                 n.xy *= ra;

            colorIndex += texture2D(reflectionSampler, uv + n.xy);

            #ifdef skyReflectionSampler
            vec3 skyReflection = texture2D(skyReflectionSampler, uv + n.xy).rgb;
            colorIndex.rgb += skyReflection.rgb;
            #endif
          }

          colorIndex /= HightQualityReflection;
          color = colorIndex;
        #else
          color = texture2DLod(reflectionSampler, uv, float(HightQualityReflection + 1) * min(5.0, ra));

          #ifdef skyReflectionSampler
          vec4 skyReflection = texture2DLod(skyReflectionSampler, uv, float(HightQualityReflection + 1) * min(4.0, ra));
          color.rgb += skyReflection.rgb * skyReflection.a;
          #endif
        #endif
        //count++;

        //color.a = step(texture2D(depthOpaque, uv).x, sky);

        //color.a = 1.0;
        //color.a = floor(color.a);
/*
        #ifdef skyReflectionSampler
        vec4 skyReflection = texture2DLod(skyReflectionSampler, uv, 0.0);
        color.rgb += skyReflection.rgb * skyReflection.a;
        #endif
*/
        //if(color.a > 0.003) color.a = 1.0;

        //color.rgb *= clamp01(g * d);
        //color.a *= clamp01(g * d);

        break;
      }

      testPoint -= rayDirection;
      rayDirection *= 0.005;
    }else{
      rayDirection *= 1.0 + (40.0 / float(SSR_Steps)) * 0.2 + maxf * maxf * maxf * 0.02;// + 0.03 * float(maxf) / 4.0;
      //0 + (16.0 / float(SSR_Steps)) * maxf * 0.13 + (float(SSR_Steps) * 0.001 * maxf)
    }
  }

  //color /= min(1.0, float(count));

  return color;
  //}
  #else
  return vec4(0.0);
  #endif
}
/*
vec4 CalculateScreenReflection(in vec3 vP, in vec3 reflectP, in vec3 normal, in float roughness, in float blockIDs){
  #if HightQualityReflection == 0 || Stage == Translucent
    return raytrace(vP, reflectP, normal, roughness);
  #else
    //float blockIDs = texture2D(gdepth, texcoord).z * 255.0;

    vec4 ssR = vec4(0.0);

    float distance = length(vP.xyz);

    if(blockIDs != 18.0 && blockIDs != 31.0){
      float dither = bayer_32x32(texcoord, resolution) * 0.3;

      vec3 t = normalize(cross(normalize(upPosition), normal));
      vec3 b = cross(normal, t);
      mat3 tbn = mat3(t, b, normal);

      vec3 nvP = normalize(vP.xyz);

      for(int i = 0; i < HightQualityReflection; i++){
        float r = ((float(i) + 1.0 + dither) * 2.0 * 3.141592) / HightQualityReflection + dither;

        vec3 normalTexture = vec3(cos(r), sin(r), 1.0);
             normalTexture.xy = normalTexture.xy * roughness / distance * 0.037 * 8.0;
             //normalTexture.xy = clamp(normalTexture.xy * 0.037 * roughness * 1.0, vec2(-1.0), vec2(1.0));
             normalTexture.xy = sqrt(normalTexture.xy * normalTexture.xy) * sign(normalTexture.xy);
             normalTexture.z = sqrt(abs(1.0 - dot(normalTexture.xy, normalTexture.xy)));
             normalTexture = normalize(tbn * normalTexture);
             //if(length(normalTexture) > 1.0) normalTexture = normal;

        vec4 t = raytrace(vP.xyz, reflect(nvP, normalTexture), normalTexture, roughness);
        ssR += t;
      }

      ssR /= HightQualityReflection;
    }

    return ssR;
  #endif
}
*/
#endif

#if (Stage <= Composite && Stage >= ScreenReflection)
//calculate solid blocks reflection
vec4 CalculateSpecularReflection(inout vec4 color, in vec3 skyReflection, in vec3 vP, in vec3 normal, in float smoothness, in float metallic, in vec3 F0){
  vec3 nvP = normalize(vP.xyz);
  vec3 reflectP = normalize(reflect(nvP, normal));

  vec3 h = normalize(reflectP - nvP);

  float vdoth = 1.0 - clamp01(dot(-nvP, h));
  float ndoth = clamp01(dot(h, normal));
  float ndotv = 1.0 - clamp01(dot(-nvP, normal));
  float ndotl = clamp01(dot(reflectP, normal));

  float roughness = 1.0 - smoothness;
        roughness = clamp(roughness * roughness, 0.00003, 0.99996);

  float d = DistributionTerm(roughness, clamp01(dot(normalize(reflectP -nvP), normal)));
  float g = VisibilityTerm(d, ndotv, ndotl);
  float specularity = pow(1.0 - g, SpecularityReflectionPower);

  vec3 f = F(F0, pow(vdoth, 5.0));

  //vec4 ssR = CalculateScreenReflection(vP, reflectP, normal, roughness, (texture2D(gdepth, texcoord).z * 255.0));
  vec4 ssR = vec4(0.0);

  #ifdef Enabled_ScreenSpaceReflection
  //ssR = raytrace(vP + normal * length(normal) * 0.07, reflectP, normal, roughness);
  ssR.rgb *= f;
  //skyReflection /= max(f, vec3(0.02));
  //skyReflection = vec3(0.0);
  #endif

  vec3 reflection = mix(skyReflection, ssR.rgb, ssR.a);

  vec4 colorWithReflection = color;

  //specularity = 1.0;
  //colorWithReflection.rgb = vec3(0.0);

  if(colorWithReflection.a > 0.0){
    colorWithReflection.rgb *= clamp01(1.0 - f * specularity);
    colorWithReflection.rgb += reflection.rgb * specularity;
  }else{
    colorWithReflection.rgb = reflection.rgb;
  }
  colorWithReflection.a = ssR.a;

  return colorWithReflection;
}
#endif
//end solid blocks reflection
//#elif Stage == Translucent
//calculate translucent blocks reflection
/*
void CalculateSpecularReflection(inout vec4 color, inout vec3 reflection, in vec3 vP, in vec3 normal, in float smoothness, in float metallic, in vec3 F0){
  vec3 nvP = normalize(vP.xyz);
  vec3 reflectP = normalize(reflect(nvP, normal));

  float ndotv = 1.0 - clamp01(dot(-nvP, normal));
  float ndotl = clamp01(dot(reflectP, normal));

  float roughness = 1.0 - smoothness;
        roughness *= roughness;

  float d = DistributionTerm(roughness, clamp01(dot(normalize(reflectP -nvP), normal)));
  float specularity = pow(1.0 - VisibilityTerm(d, ndotv, ndotl), specularityPow);//pow(smoothness, specularityPow);

  vec3 h = normalize(reflectP - nvP);

  float vdoth = 1.0 - clamp01(dot(-nvP, h));
  float ndoth = clamp01(dot(h, normal));

  vec3 f = F(F0, pow(vdoth, 5.0));

  //vec4 ssR = CalculateScreenReflection(vP, reflectP, normal, roughness, 0.0);
  //reflection = mix(reflection, ssR.rgb, ssR.a);

  color.rgb = mix(color.rgb, reflection, f * specularity);
  //color.a   = max(color.a, dot(mix(color.aaa, vec3(specularity), f * specularity), vec3(1.0)) / 3.0 + dot(reflection * f * specularity, vec3(1.0)) / 3.0);
  color.a = clamp(mix(color.a, 1.0, dot(specularity * f, vec3(1.0)) / 3.0) + dot(color.rgb, vec3(1.0) / 3.0), color.a, 1.0);
  //color.a = dot(mix(vec3(color.a), vec3(max(color.a, specularity)), f), vec3(1.0)) / 3.0 + dot(color.rgb, vec3(1.0)) / 3.0;
  //color.a = max(color.a, specularity * length(f) + length(color.rgb));
}
//end translucent blocks reflection
#endif
*/

#ifndef Gaussian_Blur
  float gaussianBlurWeights(in float offset){
    float o = 0.83;

    o *= o;
    offset *= offset;

    return (1/(sqrt(2*Pi*o)))*exp(-(offset)/(2*o));
  }

  float gaussianBlurWeights(in vec2 offset){
    float o = 0.83;

    o *= o;
    offset *= offset;

    return (1.0f/(2.0f*Pi*o))*exp(-((offset.x+offset.y)/(2.0f*o)));
  }
#endif

#if defined(Enabled_TAA) || Reflection_Filter == Temporal_AA
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

  vec4 GetColor(in sampler2D colorSample, in vec2 coord){
    #ifdef CheckerBoard_Rendering
    return GetCheckerBoardColor(colorSample, coord);
    #else
    return texture2D(colorSample, coord);
    #endif
  }

  vec3 CalculateTAA(in sampler2D color_tex, in sampler2D last_color, in float scale, in float blend, in float sharpenFactor){
    vec2 unJitterUV = texcoord - jittering * scale * pixel;

    vec3 color = texture2D(color_tex, unJitterUV).rgb;
         #ifdef simpleTonemap
         color = min(vec3(1.0), color * overRange);
         #endif
         color = encodePalYuv(color);

    vec3 maxColor = vec3(-1.0);
    vec3 minColor = vec3(1.0);
    vec3 sharpen  = vec3(0.0);

    float depth_near = 1.0;

    #define TAA_Color_Blend_Scale 1.0

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec3 color_temp = texture2D(color_tex, unJitterUV + vec2(i, j) * pixel).rgb;
             #ifdef simpleTonemap
             color_temp = min(vec3(1.0), color_temp * overRange);
             #endif
             color_temp = encodePalYuv(color_temp);

        maxColor = max(maxColor, color_temp);
        minColor = min(minColor, color_temp);

        //if(i != 0.0 && j != 0.0) sharpen += color_temp;

        #if aaDepthComp == 3
        depth_near = min(depth_near, texture2D(aaDepthTexture, unJitterUV + vec2(i, j) * pixel).a);
        #else
        depth_near = min(depth_near, texture2D(aaDepthTexture, unJitterUV + vec2(i, j) * pixel).x);
        #endif
      }
    }

    //sharpen -= color;
    //sharpen *= 0.125;

    vec4 vP = gbufferProjectionInverse * nvec4(vec3(unJitterUV, depth_near) * 2.0 - 1.0);
         vP /= vP.w;

    vec4 pvP = vP;
         pvP = gbufferModelViewInverse * pvP;
         pvP.xyz += cameraPosition - previousCameraPosition;
         pvP = gbufferPreviousModelView * pvP;
         pvP = gbufferPreviousProjection * pvP;
         pvP /= pvP.w;

    float depthFix = min(1.0, length(vP.xyz));
    if(int(round(texture2D(gdepth, texcoord).z * 255.0)) == 254.0) depthFix *= MC_HAND_DEPTH * MC_HAND_DEPTH;

    vec2 previousCoord = (pvP.xy * 0.5 + 0.5);
    previousCoord = (unJitterUV - previousCoord) * depthFix;

    float motionScale = length(-previousCoord * resolution);
          motionScale = min(motionScale, 0.5) * 2.0;

    previousCoord = -previousCoord + texcoord;

    blend += blend * 0.3 * motionScale * sharpenFactor;

    float mixRate = texture2D(last_color, vec2(0.5)).a * float(floor(previousCoord) == vec2(0.0));

    vec3 lastColor = encodePalYuv(texture2D(last_color, previousCoord).rgb);
         lastColor = mix(color, lastColor, mixRate * (1.0 - blend));

    maxColor += (color - maxColor) * motionScale * 0.02 * sharpenFactor;
    minColor += (color - minColor) * motionScale * 0.02 * sharpenFactor;

    //#if aaDepthComp == 3
    //minColor = mix(minColor, lastColor, mixRate * 0.9);
    //maxColor = mix(maxColor, lastColor, mixRate * 0.9);
    //#endif

    vec3 antialiased = clipToAABB(lastColor, minColor, maxColor);
         //antialiased += (antialiased - sharpen) * 0.00025 * sharpenFactor * motionScale;
         //antialiased = mix(color, antialiased, 1.0 - blend);

    return decodePalYuv(antialiased);
  }
#endif

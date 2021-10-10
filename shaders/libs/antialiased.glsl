#define TAA_Sharpen 50  //[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]

#define Less_Edge 0.05  //[0.05 0.1 0.15 0.2 0.25]
#define Base_Motion_Sharpen 0.05  //

//#define Very_Sharpe_Image
//#define TAA_No_Clip

#define Yuv 0
#define YCoCg 1
#define Color_Format YCoCg //[Yuv YCoCg]

#ifdef Enabled_TAA
  vec3 encodePalYuv(vec3 rgb) {
      rgb = decodeGamma(rgb);

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

      return encodeGamma(rgb);
  }

  // https://software.intel.com/en-us/node/503873
	vec3 RGB_YCoCg(vec3 c)
	{
		// Y = R/4 + G/2 + B/4
		// Co = R/2 - B/2
		// Cg = -R/4 + G/2 - B/4
    //return c;

    //c = decodeGamma(c);

		return vec3(
			 c.x/4.0 + c.y/2.0 + c.z/4.0,
			 c.x/2.0 - c.z/2.0,
			-c.x/4.0 + c.y/2.0 - c.z/4.0
		);
	}

	vec3 YCoCg_RGB(vec3 c)
	{
		// R = Y + Co - Cg
		// G = Y + Cg
		// B = Y - Co - Cg
    //return c;

    c = saturate(vec3(
			c.x + c.y - c.z,
			c.x + c.z,
			c.x - c.y - c.z
		));

    return c;
		//return encodeGamma(c);
	}

  #if Color_Format == Yuv
  #define encode encodePalYuv
  #define decode decodePalYuv
  #else
  #define encode RGB_YCoCg
  #define decode YCoCg_RGB
  #endif

  vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    vec3 p_clip = 0.5 * (maximum + minimum);
    vec3 e_clip = 0.5 * (maximum - minimum);

    vec3 v_clip = color - p_clip;
    vec3 v_unit = v_clip.xyz / e_clip;
    vec3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0) return p_clip + v_clip / ma_unit;
    
    return color;// point inside aabb
  }

  vec3 GetClosest(in vec2 coord){
    vec3 closest = vec3(0.0, 0.0, far);

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 neighborhood = vec2(i, j) * pixel;
        float neighbor = (texture(depthtex1, coord + neighborhood).x);

        if(neighbor < closest.z){
          closest.z = neighbor;
          closest.xy = neighborhood;
        }
      }
    }

    closest.xy += coord;
    //closest.z = mix(closest.z, texture(depthtex1, closest.xy).x, 0.5);

    return vec3(closest.xy, closest.z);
  }

  uniform vec2 previousJitter;

  vec2 GetMotionVector(in vec3 coord){
    vec4 view = gbufferProjectionInverse * nvec4(coord * 2.0 - 1.0);
         view /= view.w;
         view = gbufferModelViewInverse * view;
         view.xyz += cameraPosition - previousCameraPosition;
         view = gbufferPreviousModelView * view;
         view = gbufferPreviousProjection * view;
         view /= view.w;
         view.xy = view.xy * 0.5 + 0.5;

    vec2 velocity = coord.xy - view.xy;

    if(coord.z < 0.7) velocity *= 0.01;

    return velocity;
  }

  float FilterCubic(in float x, in float B, in float C)
  {
     float y = 0.0f;
     float x2 = x * x;
     float x3 = x * x * x;
     if(x < 1)
         y = (12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B);
     else if (x <= 2)
         y = (-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C);

     return y / 6.0f;
  }

  #define Gaussian_Sharpen 0
  #define Bicubic 1
  #define HistorySamplerFitlter Gaussian_Sharpen

  vec4 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
    vec4 result = vec4(0.0);

    #if 0
      float sigma = 0.83;
            sigma = 2.0 * pow2(sigma);

      for(float i = -1.0; i <= 1.0; i += 1.0){
        for(float j = -1.0; j <= 1.0; j += 1.0){
          vec2 samplePos = vec2(i, j) * 0.1;
          
          float x = pow2(length(samplePos) + 1e-5);

          float weight = exp(-x / sigma);

          vec4 sampler = vec4((texture2D(tex, pixelPos + samplePos * pixel).rgb), 1.0);

          result += sampler * weight;
        }
      }

      result.rgb /= result.a;
    #else
    vec2 position = resolution * pixelPos;
    vec2 centerPosition = floor(position - 0.5) + 0.5;

    vec2 f = position - centerPosition;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    float c = TAA_Sharpen  * 0.009;
    vec2 w0 =         -c  *  f3 + 2.0 * c          *  f2 - c  *  f;
    vec2 w1 =  (2.0 - c)  *  f3 - (3.0 - c)        *  f2            + 1.0;
    vec2 w2 = -(2.0 - c)  *  f3 + (3.0 - 2.0 * c)  *  f2 + c  *  f;
    vec2 w3 =          c  *  f3 - c                *  f2;
    vec2 w12 = w1 + w2;

    vec2 tc12 = pixel * (centerPosition + w2 / w12);
    vec2 tc0 = pixel * (centerPosition - 1.0);
    vec2 tc3 = pixel * (centerPosition + 2.0);

    result =  vec4(texture2D(tex, vec2(tc12.x, tc0.y)).rgb, 1.0) * (w12.x * w0.y) +
              vec4(texture2D(tex, vec2(tc0.x, tc12.y)).rgb, 1.0) * (w0.x * w12.y) +
              vec4(texture2D(tex, vec2(tc12.x, tc12.y)).rgb, 1.0) * (w12.x * w12.y) +
              vec4(texture2D(tex, vec2(tc3.x, tc12.y)).rgb, 1.0) * (w3.x * w12.y) +
              vec4(texture2D(tex, vec2(tc12.x, tc3.y)).rgb, 1.0) * (w12.x * w3.y);

    result /= result.a;
    result.rgb = saturate(result.rgb);
    #endif

    result.rgb = encode(result.rgb);

    return result;
  }

  void ResolverAABB(in sampler2D tex, in vec2 coord, inout vec3 minColor, inout vec3 maxColor, inout vec3 variance, float gamma){
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec3 m = encode(texture2D(tex, coord + vec2(i, j) * pixel).rgb);

        m1 += m;
        m2 += m * m;
      }
    }

    vec3 mu = m1 / 9.0;
    
    vec3 sigma = sqrt(abs(m2 / 9.0 - mu * mu));
    variance = sigma;

    minColor = mu - gamma * sigma;
    maxColor = mu + gamma * sigma;
  }

  vec3 CalculateTAA(in sampler2D currentSampler, in sampler2D previousSampler){
    vec3 maxColor = vec3(-1.0);
    vec3 minColor = vec3(1.0);

    vec3 variance = vec3(0.0);

    ResolverAABB(currentSampler, texcoord, minColor, maxColor, variance, 2.0);
    /*
    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec3 color_temp = texture2D(currentSampler, texcoord + vec2(i, j) * pixel).rgb;
             color_temp = encode(color_temp);

        maxColor = max(maxColor, color_temp);
        minColor = min(minColor, color_temp);
      }
    }
    */

    vec3 currentColor = encode(texture2D(currentSampler, texcoord).rgb);

    vec3 closest = GetClosest(texcoord);
    vec2 velocity = GetMotionVector(closest);
    vec2 previousCoord = texcoord - velocity;

    float motion = length(velocity * resolution);

    float inScreenPrev = step(abs(previousCoord.x - 0.5), 0.5) * step(abs(previousCoord.y - 0.5), 0.5);

    vec3 previousColor = ReprojectSampler(previousSampler, previousCoord).rgb; 
    //vec3 previousColor = encode(texture2D(previousSampler, previousCoord).rgb);

    if(maxComponent(previousColor) == 0.0) previousColor = currentColor;

    #ifndef TAA_No_Clip
    previousColor = clipToAABB(previousColor, minColor, maxColor);
    #endif

    float blend = (0.95) * inScreenPrev;

    float luminance = maxComponent(decode(variance));
    float motionBlend = (remap(Less_Edge, 1.0, luminance) + Base_Motion_Sharpen) / (1.0 + Base_Motion_Sharpen) * step(0.05, motion) * 0.05;

    blend = blend - min(0.2, motionBlend);
    blend = max(0.0, blend);

    vec3 antialiased = mix(currentColor, previousColor, vec3(blend));

    return decode(antialiased);
  }
#endif

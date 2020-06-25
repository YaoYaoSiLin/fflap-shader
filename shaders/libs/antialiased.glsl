#define TAA_Sharpen 50 //[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]

#define Yuv 0
#define YCoCg 1
#define Color_Format YCoCg //[Yuv YCoCg]

#ifdef Enabled_TAA
  vec3 encodePalYuv(vec3 rgb) {
      rgb = (rgb);

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

      return (rgb);
  }

  // https://software.intel.com/en-us/node/503873
	vec3 RGB_YCoCg(vec3 c)
	{
		// Y = R/4 + G/2 + B/4
		// Co = R/2 - B/2
		// Cg = -R/4 + G/2 - B/4
    //return c;
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
		return saturate(vec3(
			c.x + c.y - c.z,
			c.x + c.z,
			c.x - c.y - c.z
		));
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

    if (ma_unit > 1.0)
        return p_clip + v_clip / ma_unit;
    else
        return color;// point inside aabb
  }

  vec3 GetClosest(in vec2 coord){
    vec3 closest = vec3(0.0, 0.0, 1.0);

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 neighborhood = vec2(i, j) * pixel;
        float neighbor = texture(depthtex0, coord + neighborhood).x;

        if(neighbor < closest.z){
          closest.z = neighbor;
          closest.xy = neighborhood;
        }
      }
    }

    closest.xy += coord;

    return closest;
  }

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

    if(coord.z < 0.7) velocity *= 0.0001;

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

  vec4 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
    vec4 result = vec4(0.0);

    float weights = 0.0;

    //vec2 velocity = pixelPos - texcoord - jittering * pixel;
    //pixelPos = pixelPos - velocity;

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 samplePos = vec2(i, j);
        float weight = gaussianBlurWeights(samplePos + 0.0001);

        vec4 sampler = texture2D(tex, samplePos * pixel / sqrt(32.0) + pixelPos);

        weights += weight;
        result += sampler * weight;
      }
    }

    result /= weights;

    /*
    sum = texture2D(tex, pixelPos);
    sum.rgb = RGB_YCoCg(sum.rgb);
    */

    //pixelPos = round(pixelPos * resolution);

    //sum = SampleTextureCatmullRom(tex, pixelPos, resolution);
    //sum.rgb = RGB_YCoCg(sum.rgb);
    /*
    vec2 samplePos = pixelPos * resolution;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    vec2 f = samplePos - texPos1;

    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 *= pixel;
    texPos3 *= pixel;
    texPos12 *= pixel;

    result += texture2D(tex, vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += texture2D(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += texture2D(tex, vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += texture2D(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += texture2D(tex, vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += texture2D(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;
    result.rgb = RGB_YCoCg(result.rgb);

    */
/*
    pixelPos = pixelPos * resolution;

    float weights = 0.0;

    for(float i = -1.0; i <= 2.0; i += 3.0){
      for(float j = -1.0; j <= 2.0; j += 3.0){
        vec2 samplePos = round(pixelPos + vec2(i, j));
        vec2 sampleDist = abs(samplePos - pixelPos);

        float weight = FilterCubic(sampleDist.x, 0, 0.5) * FilterCubic(sampleDist.y, 0, 0.5);
        weights += weight;

        sum += texture2D(tex, samplePos * pixel) * weight;
      }
    }

    sum /= weights;
    sum.rgb = RGB_YCoCg(sum.rgb);
*/

    /*
    pixelPos = (pixelPos * resolution);

    float weights = 0.0;

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec2 samplePos = round(pixelPos + vec2(i, j) - 0.5) + 0.5;

        vec4 sampler = texture2D(tex, samplePos*pixel);

        float weight = gaussianBlurWeights(abs(samplePos-pixelPos) + 0.0001);

        result += sampler * weight;

        weights += weight;
      }
    }

    result /= weights;
    */
    /*
    vec2 samplePos = pixelPos * resolution;
    vec2 texPos1 = round(samplePos+0.5)-0.5;

    vec2 f = samplePos - texPos1;

    float w0 = gaussianBlurWeights(f.x);
    float w3 = gaussianBlurWeights(f.y);
    float w12 = w0 + w3;

    vec2 texPos0 = texPos1 - vec2(2.0) * 1.0;
    vec2 texPos3 = texPos1 + vec2(1.0) * 1.0;
    vec2 texPos12 = texPos1 + vec2(w0) / (w0 + w3) * 1.0;

    texPos0 *= pixel;
    texPos3 *= pixel;
    texPos12 *= pixel;

    result += texture2D(tex, vec2(texPos0.x, texPos0.y)) * w0 * w0;
    result += texture2D(tex, vec2(texPos0.x, texPos12.y)) * w0 * w12;
    result += texture2D(tex, vec2(texPos0.x, texPos3.y)) * w0 * w3;

    result += texture2D(tex, vec2(texPos12.x, texPos0.y)) * w12 * w0;
    result += texture2D(tex, vec2(texPos12.x, texPos12.y)) * w12 * w12;
    result += texture2D(tex, vec2(texPos12.x, texPos3.y)) * w12 * w3;

    result += texture2D(tex, vec2(texPos3.x, texPos0.y)) * w3 * w0;
    result += texture2D(tex, vec2(texPos3.x, texPos12.y)) * w3 * w12;
    result += texture2D(tex, vec2(texPos3.x, texPos3.y)) * w3 * w3;

    result /= w0*w0 + w12*w12 + w3*w3 + w0*w12*2.0 + w0*w3*2.0 + w12*w3*2.0;
    */
    //result = texture2D(tex, texPos1 * pixel);

/*
    vec2 position = resolution * pixelPos;
    vec2 centerPosition = floor(position - 0.5) + 0.5;

    vec2 f = position - centerPosition;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    float c = 50.0  * 0.01;
    vec2 w0 = -c * f3 + 2.0 * c * f2 - c * f;
    vec2 w1 = (2.0 - c) * f3 - (3.0 - c) * f2 + 1.0;
    vec2 w2 = -(2.0 - c) * f3 + (3.0 - 2.0 * c) * f2 + c * f;
    vec2 w3 = c * f3 - c * f2;
    vec2 w12 = w1 + w2;

    vec2 tc12 = pixel * (centerPosition + w2 / w12);
    vec3 centerColor = texture2D(tex, vec2(tc12.x, tc12.y)).rgb;
    vec2 tc0 = pixel * (centerPosition - 1.0);
    vec2 tc3 = pixel * (centerPosition + 2.0);

    result = vec4(texture2D(tex, vec2(tc12.x, tc0.y)).rgb, 1.0) * (w12.x * w0.y) +
                  vec4(texture2D(tex, vec2(tc0.x, tc12.y)).rgb, 1.0) * (w0.x * w12.y) +
                  vec4(centerColor, 1.0) * (w12.x * w12.y) +
                  vec4(texture2D(tex, vec2(tc3.x, tc12.y)).rgb, 1.0) * (w3.x * w12.y) +
                  vec4(texture2D(tex, vec2(tc12.x, tc3.y)).rgb, 1.0) * (w12.x * w3.y);
         result /= result.a;
*/

    result.rgb = encode(result.rgb);

    return result;
  }

  vec3 CalculateTAA(in sampler2D color_tex, in sampler2D last_color){
    vec2 unjittering = texcoord + jittering * pixel;

    vec3 maxColor = vec3(-1.0);
    vec3 minColor = vec3(1.0);
    vec3 sharpen  = vec3(0.0);

    vec3 closest = GetClosest(texcoord);

    for(float i = -1.0; i <= 1.0; i += 1.0){
      for(float j = -1.0; j <= 1.0; j += 1.0){
        vec3 color_temp = texture2D(color_tex, unjittering + vec2(i, j) * pixel).rgb;
             color_temp = encode(color_temp);

        maxColor = max(maxColor, color_temp);
        minColor = min(minColor, color_temp);

        //if(i != 0 || j != 0)
        sharpen += color_temp * min(1.0, abs(sign(i)) + abs(sign(j)));
      }
    }

    sharpen /= 8.0;
    //color = RGB_YCoCg(color);

    vec3 currentColor = texture2D(color_tex, unjittering).rgb;
    currentColor = encode(currentColor);

    vec2 previousCoord = GetMotionVector(closest);
    vec2 velocity = previousCoord;

    float motion = length(previousCoord*resolution);
          //motion = step(0.001, motion);

    previousCoord = texcoord - previousCoord;

    float inScreenPrev = step(0.0, previousCoord.x) * step(previousCoord.x, 1.0) * step(0.0, previousCoord.y) * step(previousCoord.y, 1.0);

    vec3 previousColor = ReprojectSampler(last_color, previousCoord).rgb;
         previousColor = clipToAABB(previousColor, minColor, maxColor);

    #define LowFreqWeight 0.85
    #define HiFreqWeight  0.95

    vec3 weightA = vec3(0.05) * inScreenPrev;
    vec3 weightB = 1.0 - weightA;

    #define Static_Blend 0
    #define Lum_Base_Blend 1
    #define Color_Base_Blend 2
    #define TAA_Blend Color_Base_Blend //[Static_Blend Lum_Base_Blend Color_Base_Blend]

    #if TAA_Blend == Lum_Base_Blend
    vec3 Lweight = vec3(0.2126, 0.7152, 0.0722);
    float maxL = dot(maxColor, Lweight);
    float minL = dot(minColor, Lweight);
    float cenL = dot(currentColor, Lweight);
    float blend = clamp01(abs(maxL - minL) / cenL);
    #elif TAA_Blend == Color_Base_Blend
    vec3 blend = clamp01(abs(maxColor - minColor) / currentColor);
    #endif

    #if TAA_Blend > 0 && TAA_Blend < 3
    weightB = lerq(vec3(LowFreqWeight), vec3(HiFreqWeight), blend) * inScreenPrev;
    weightA = 1.0 - weightB;
    #endif

    vec3 antialiased = (currentColor * weightA + previousColor * weightB);

    sharpen = clamp(currentColor - sharpen, vec3(-0.001), vec3(0.001));
    antialiased += sharpen * inScreenPrev * TAA_Sharpen * 0.01 * lerq(vec3(0.7071), vec3(sqrt(2.0)), blend);

    return decode(antialiased);
  }
#endif

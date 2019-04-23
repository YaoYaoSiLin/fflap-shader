#define Enabled_Bloom							//
  #define Bloom_Steps 5						//[1 2 3 4 5 6 7 8]
  #define Bloom_First_Steps 1			//[1 2 3 4 5 6 7 8]
  #define Bloom_Sample_Scale 1.25 //[0.8 0.9 1.0 1.1 1.25] enabled DEGBUG_Bloom to change bloom sample scale

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
/*
const float bloomWeights[9] = float[9](0.0541, 0.2326, 0.0541,
                                       0.2326, 0.4806, 0.2326,
                                       0.0541, 0.2326, 0.0541);
*/
#if Stage == Bloom && defined(bloomSampler) && defined(Enabled_Bloom)
//bloom sampler
  vec3 GetBloom(in float x, in vec2 coord, in vec2 offset, inout bool alive, in float dither){
    vec3 bloom = vec3(0.0);
  	float r = 1.0;

    float lod = exp2(Bloom_First_Steps + x);

    //dither *= lod;
    //dither *= 1.0;

    float lastSalce = exp2(x + Bloom_First_Steps - 1.0);
    float fristSampleScale = exp2(1);

    coord -= offset;

  	vec2 bloomCoord = (coord + vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;

    if(((1.0 - pixel.x) + ((1.0 / lastSalce) * 1.1 - 1.1) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale < 1.0){
      //return vec3(coord, 0.0);
      bloomCoord = (coord - (offset * 2.0 + vec2(-(1.0 / fristSampleScale), 1.0 / fristSampleScale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;
    }

  	//if(bloomCoord == clamp(bloomCoord, vec2(0.0), vec2(1.0))){
  	if(bloomCoord.x > -0.002 && bloomCoord.x < 1.002
  	&& bloomCoord.y > -0.002 && bloomCoord.y < 1.002){

      float weights = 0.0;

      int rounds = 0;

      for(float i = -1.0; i <= 1.0; i += 1.0){
    		for(float j = -1.0; j <= 1.0; j += 1.0){
  				vec2 offsets = (vec2(i, j));
          float r = (1.0 + float(rounds) * 2.0 * 3.14159) / 9.0 + dither;

          float weight = gaussianBlurWeights(offsets);
          offsets += vec2(cos(r), sin(r)) * (8.0 + lod) * 0.5;

  				bloom += rgb2L(texture2DLod(bloomSampler, bloomCoord + offsets * pixel, 0.0).rgb) * weight;//5 + lod * 0.12
          weights += weight;

  				rounds++;
  			}
  		}

      if(weights > 0.0)
      bloom /= weights;

  		alive = true;
  		//r = lod;
  	}

  	return bloom;
  }

  vec4 CalculateBloomSampler(in vec2 coord) {
    vec3 bloom = vec3(0.0);
    bool alive = false;

    //#define DEBUG_Bloom
    //  #define DEBUG_Bloom_ScreenOverScale 4.0
    //  #define DEBUG_Bloom_ScreenOverScale_Line
    //    #define DEBUG_Bloom_ScreenOverScale_Scale 0.001
    //    #define DEBUG_Bloom_ScreenOverScale_LineColor_R 1.0
    //    #define DEBUG_Bloom_ScreenOverScale_LineColor_G 1.0
    //    #define DEBUG_Bloom_ScreenOverScale_LineColor_B 1.0

    //#if defined(DEBUG_Bloom) && defined(DEBUG_Bloom_ScreenOverScale)
    //coord = coord * 2.0 - 1.0;
    //coord *= DEBUG_Bloom_ScreenOverScale;
    //coord = coord * 0.5 + 0.5;
    //#endif

    float dither = bayer_32x32(coord, resolution);

    for(int i = 0; i < int(Bloom_Steps); i++) {
      //if(alive) break;
  		bloom += GetBloom(float(i), coord, vec2(0.002), alive, dither);
  	}

    //bloom = GetBloom(1.0, coord, vec2(0.002), alive);

    //#if defined(DEBUG_Bloom) && defined(DEBUG_Bloom_ScreenOverScale_Line)
    //if((coord.x > 1.0 && coord.x < 1.0 + DEBUG_Bloom_ScreenOverScale_Scale)
    //|| (coord.x < 0.0 && coord.x > 0.0 - DEBUG_Bloom_ScreenOverScale_Scale)
    //|| (coord.y > 1.0 && coord.y < 1.0 + DEBUG_Bloom_ScreenOverScale_Scale * aspectRatio)
    //|| (coord.y < 0.0 && coord.y > 0.0 - DEBUG_Bloom_ScreenOverScale_Scale * aspectRatio)
    //){
      //bloom.xy += coord * 0.3;
    //  bloom = vec3(DEBUG_Bloom_ScreenOverScale_LineColor_R, DEBUG_Bloom_ScreenOverScale_LineColor_G, DEBUG_Bloom_ScreenOverScale_LineColor_B);
    //}
    //#endif

    //if(!alive){
    //  bloom = vec3(coord, 0.0);
    //}

    return vec4(L2rgb(bloom), float(alive));
  }
//end bloom sampler
#endif

#if Stage == Final && defined(bloomSampler) && defined(Enabled_Bloom)
  vec4 cubic(float x) {
      float x2 = x * x;
      float x3 = x2 * x;
      vec4 w;
      w.x =   -x3 + 3.0*x2 - 3.0*x + 1.0;
      w.y =  3.0*x3 - 6.0*x2       + 4.0;
      w.z = -3.0*x3 + 3.0*x2 + 3.0*x + 1.0;
      w.w =  x3;
      return w / 6.0;
  }

  vec4 BicubicTexture(in sampler2D tex, in vec2 coord) {
  	coord *= resolution;

  	float fx = fract(coord.x);
    float fy = fract(coord.y);
    coord.x -= fx;
    coord.y -= fy;

    fx -= 0.5;
    fy -= 0.5;

    vec4 xcubic = cubic(fx);
    vec4 ycubic = cubic(fy);

    vec4 c = vec4(coord.x - 0.5, coord.x + 1.5, coord.y - 0.5, coord.y + 1.5);
    vec4 s = vec4(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
    vec4 offset = c + vec4(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;

    vec4 sample0 = texture2D(tex, vec2(offset.x, offset.z) / resolution);
    vec4 sample1 = texture2D(tex, vec2(offset.y, offset.z) / resolution);
    vec4 sample2 = texture2D(tex, vec2(offset.x, offset.w) / resolution);
    vec4 sample3 = texture2D(tex, vec2(offset.y, offset.w) / resolution);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix( mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
  }

  void CalculateBloom(inout vec3 color, in vec2 coord){
    vec3 bloom = vec3(0.0);

    vec2 offset = vec2(0.002);

    for(int i = 0; i < int(Bloom_Steps); i++){
      float lod = exp2(Bloom_First_Steps + i) / Bloom_Sample_Scale;

      float lastSalce = exp2(i + Bloom_First_Steps - 1.0);
      float fristSampleScale = exp2(1);

      //vec2 bloomCoord = (coord + vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;
      //vec2 bloomCoord = (coord / x * Bloom_Sample_Scale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0) / Bloom_Sample_Scale;
      //vec2 bloomCoord = ((coord) / x * Bloom_Sample_Scale) + offset;
      //vec2 bloomCoord = (coord + vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;
      vec2 bloomCoord = (coord / lod) - (vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) + offset;

      if(((1.0 - pixel.x) + ((1.0 / lastSalce) * 1.1 - 1.1) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale < 1.0){
        //return vec3(coord, 0.0);
        //bloomCoord = (coord - (offset * 2.0 + vec2(-(1.0 / fristSampleScale), 1.0 / fristSampleScale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;
        //bloomCoord = (coord / lod) + offset * 3.0 + vec2((1.0 / fristSampleScale), (1.0 / fristSampleScale)) + vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0);
        //bloomCoord = (coord / lod) + (offset * 2.0 + vec2(-(1.0 / fristSampleScale), 1.0 / fristSampleScale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) * Bloom_Sample_Scale + offset;
        // + offset * 3.0 + vec2((1.0 / fristSampleScale), -(1.0 / fristSampleScale)) + vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)
        bloomCoord = (coord / lod) + (offset * 2.0 + vec2(-(1.0 / fristSampleScale), 1.0 / fristSampleScale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) * Bloom_Sample_Scale + offset;
      }

      bloom += rgb2L(BicubicTexture(bloomSampler, bloomCoord).rgb) * (1.0 + gaussianBlurWeights(float(i)));

      //if(((1.0 - pixel.x) + ((1.0 / lastSalce) * 1.1 - 1.1) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale < 1.0){
      //  //return vec3(coord, 0.0);
      //  bloomCoord = (coord - (offset * 2.0 + vec2(-(1.0 / fristSampleScale), 1.0 / fristSampleScale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;
      //}

    }

    //bloom /= Bloom_Steps;

    //color *= 3.0;

    //bloom /= float(Bloom_Steps);
    //bloom = (bloom * 65535);
    //color = mix(color, bloom * 65535, 0.0007);
    //color = rgb2L(BicubicTexture(bloomSampler, coord).rgb) * 65535 * 0.01;
    //color += bloom * (1.0 + clamp01(getLum(color * 10.0 - bloom))) * 255.;
    color = bloom;
    //color = rgb2L(BicubicTexture(bloomSampler, coord).rgb) * 65535.0;

    //color += mix(bloom, color, 0.000001);
    //color = mix(color, bloom, 0.0152);
    //color += bloom * 0.0152;

    //color = mix(color, bloom, 0.001);

    //color += bloom;
    //color = rgb2L(texture2D(bloomSampler, coord).rgb);
  }
#endif

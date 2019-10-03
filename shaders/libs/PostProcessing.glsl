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

vec2 bloomSamplOffset[8] = vec2[8](vec2(0.0),
                                   vec2(0.508, 0.0),
                                   vec2(0.762, 0.0),
                                   vec2(0.891, 0.0),
                                   vec2(0.9575, 0.0),
                                   vec2(0.0, 0.508),
                                   vec2(0.0, 0.508),
                                   vec2(0.0, 0.508)
                                   );

#if Stage == Bloom && defined(bloomSampler) && defined(Enabled_Bloom)
//bloom sampler
  vec3 GetBloom(in float x, in vec2 coord, in vec2 offset, in mat2 rotate){
    vec3 bloom = vec3(0.0);

    float lod = exp2(1.0 + x);

  	vec2 bloomCoord = (coord - offset + 0.0002) * lod;

  	if(bloomCoord.x > -0.004 && bloomCoord.x < 1.004 && bloomCoord.y > -0.004 && bloomCoord.y < 1.004){

      float weights = 0.0;

      int rounds = 0;

      for(float i = -1.0; i <= 1.0; i += 1.0){
    		for(float j = -1.0; j <= 1.0; j += 1.0){
  				vec2 offsets = vec2(i, j);
          //float r = (1.0 + float(rounds) * 2.0 * 3.14159) / 9.0;

          float weight = gaussianBlurWeights(offsets);
          //      weight = 1.0;
          //offsets += vec2(cos(r), sin(r)) * (8.0 + lod) * 0.4;

          //offsets -= offsets * (rotate) * lod * 0.0625;

  				bloom += (texture2DLod(bloomSampler, bloomCoord + offsets * pixel * lod, lod * 0.24).rgb * weight);
          weights += weight;
          //rounds++;
  			}
  		}

      bloom /= weights;
  	}

  	return bloom;
  }

  vec4 CalculateBloomSampler(in vec2 coord) {
    vec3 bloom = vec3(0.0);

    float dither = bayer_32x32(coord, resolution) * 2.0 * Pi;

    mat4 rotate = mat4(cos(dither), -sin(dither), 0.0, 0.0,
                       sin(dither),  cos(dither), 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0);

    coord -= 0.004 * vec2(1.0, aspectRatio);

    for(int i = 0; i < int(Bloom_Steps); i++) {
  		bloom += GetBloom(float(i), coord, bloomSamplOffset[i], mat2(rotate));
  	}

    return vec4(bloom, 1.0);
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
    //coord = floor(coord + 0.5);

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
    vec2 offset = vec2(0.002);
    vec3 bloom = vec3(0.0);

    float weights = 0.0;

    for(int i = 0; i < int(Bloom_Steps); i++){
      float lod = exp2(1.0 + i);

      //vec2 bloomCoord = (coord / lod) - (vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) + offset;
      vec2 bloomCoord = (coord) / lod + bloomSamplOffset[i] + 0.004 * vec2(1.0, aspectRatio);

      float weight = gaussianBlurWeights(float(i));
      bloom += BicubicTexture(bloomSampler, bloomCoord).rgb / (1.0 + weight);
      weights += weight;

      //if(((1.0 - pixel.x) + ((1.0 / lastSalce) * 1.1 - 1.1) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale < 1.0){
      //  //return vec3(coord, 0.0);
      //  bloomCoord = (coord - (offset * 2.0 + vec2(-(1.0 / fristSampleScale), 1.0 / fristSampleScale) - vec2((1.0 / lastSalce) * 1.1 - 1.1, 0.0)) * Bloom_Sample_Scale) * lod / Bloom_Sample_Scale;
      //}

    }

    //bloom *= overRange;
    //bloom /= weights;

    //bloom = texture2D(bloomSampler, coord).rgb * 16.0;

    bloom = rgb2L(bloom) * overRange * overRange;
    //bloom *= pow(overRange, 2.2);

    //if(isEyeInWater == 1) color = mix(bloom, color, abs(getLum(bloom) - getLum(color)));
    //if(coord.x < 0.5) color = rgb2L(BicubicTexture(bloomSampler, coord).rgb * );

    //bloom /= Bloom_Steps;

    //color = bloom;

    float lum = 1.0 + dot(bloom, vec3(0.7874, 0.2848, 0.9278));

    color += bloom * 0.333 * lum * 0.02;

    //color = bloom;

    //color = BicubicTexture(colortex, coord).rgb;

    //bloom *= lum;

    //color += bloom * overRange * 100.0 * lum;

    //bloom *= lum;
    //color += clamp01(bloom + (bloom - color)) * lum * 0.0333;

    //color += clamp01(bloom - color) * 0.033 * lum;


    //color = texture2D(bloomSampler, coord).rgb;

    //bloom *= lum * 75.0;
    //color += clamp01(bloom - color) * lum * 7.0;

    //color += clamp01(bloom + (bloom - color) * 0.75);


    //color += bloom + clamp(bloom - color, -bloom, bloom);

    //color *= 3.0;

    //bloom /= float(Bloom_Steps);
    //bloom = (bloom * 65535);
    //color = mix(color, bloom * 65535, 0.0007);
    //color = rgb2L(BicubicTexture(bloomSampler, coord).rgb) * 65535 * 0.01;
    //color += bloom * (1.0 + clamp01(getLum(color * 10.0 - bloom))) * 255.;
    //color += bloom * 4.0;
    //color = rgb2L(BicubicTexture(bloomSampler, coord).rgb) * 65535.0;

    //color += mix(bloom, color, 0.000001);
    //color = mix(color, bloom, 0.0152);
    //color += bloom * 0.0152;

    //color = mix(color, bloom, 0.001);

    //color += bloom;
    //color = rgb2L(texture2D(bloomSampler, coord).rgb);
  }
#endif

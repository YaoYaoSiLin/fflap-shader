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
  vec3 GetBloom(in float x, in vec2 coord, in vec2 offset){
    vec3 bloom = vec3(0.0);

    float lod = exp2(1.0 + x);

  	coord = (coord - offset + 0.0002) * lod;

  	if(coord.x > -0.004 && coord.x < 1.004 && coord.y > -0.004 && coord.y < 1.004){

      float weights = 0.0;

      int rounds = 0;

      vec2 bloomCoord = coord * resolution / lod;

      for(float i = -1.0; i <= 1.0; i += 1.0){
    		for(float j = -1.0; j <= 1.0; j += 1.0){
  				vec2 offsets = vec2(i, j);
          vec2 samplePos = (floor(bloomCoord + offsets + 1.5) - 1.0) * pixel * lod;

          float weight = gaussianBlurWeights(offsets);
  				bloom += (texture2D(bloomSampler, samplePos).rgb * weight);
          weights += weight;
  			}
  		}

      bloom /= weights;
  	}

  	return bloom;
  }

  vec4 CalculateBloomSampler(in vec2 coord) {
    vec3 bloom = vec3(0.0);

    coord -= 0.004 * vec2(1.0, aspectRatio);

    for(int i = 0; i < int(Bloom_Steps); i++) {
  		bloom += GetBloom(float(i), coord, bloomSamplOffset[i]);
  	}
    /*
    bloom += GetBloom(0.0, coord, bloomSamplOffset[0]);
    bloom += GetBloom(1.0, coord, bloomSamplOffset[1]);
    bloom += GetBloom(2.0, coord, bloomSamplOffset[2]);
    bloom += GetBloom(3.0, coord, bloomSamplOffset[3]);
    bloom += GetBloom(4.0, coord, bloomSamplOffset[4]);
    */

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

  vec4 BicubicTexture(in sampler2D tex, in vec2 coord, in vec2 texSize) {
  	coord *= texSize;
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

    vec4 sample0 = texture2D(tex, vec2(offset.x, offset.z) / texSize);
    vec4 sample1 = texture2D(tex, vec2(offset.y, offset.z) / texSize);
    vec4 sample2 = texture2D(tex, vec2(offset.x, offset.w) / texSize);
    vec4 sample3 = texture2D(tex, vec2(offset.y, offset.w) / texSize);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix( mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
  }

  //from : https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
  vec4 SampleTextureCatmullRom(sampler2D tex, vec2 uv, vec2 texSize )
  {
      // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
      // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
      // location [1, 1] in the grid, where [0, 0] is the top left corner.
      vec2 invtexSize = 1.0 / texSize;

      vec2 samplePos = uv * texSize;
      vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

      // Compute the fractional offset from our starting texel to our original sample location, which we'll
      // feed into the Catmull-Rom spline function to get our filter weights.
      vec2 f = samplePos - texPos1;

      // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
      // These equations are pre-expanded based on our knowledge of where the texels will be located,
      // which lets us avoid having to evaluate a piece-wise function.
      vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
      vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
      vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
      vec2 w3 = f * f * (-0.5 + 0.5 * f);

      // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
      // simultaneously evaluate the middle 2 samples from the 4x4 grid.
      vec2 w12 = w1 + w2;
      vec2 offset12 = w2 / (w1 + w2);

      // Compute the final UV coordinates we'll use for sampling the texture
      vec2 texPos0 = texPos1 - vec2(1.0);
      vec2 texPos3 = texPos1 + vec2(2.0);
      vec2 texPos12 = texPos1 + offset12;

      texPos0 *= invtexSize;
      texPos3 *= invtexSize;
      texPos12 *= invtexSize;

      vec4 result = vec4(0.0);
      result += texture2D(tex, vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
      result += texture2D(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
      result += texture2D(tex, vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

      result += texture2D(tex, vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
      result += texture2D(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
      result += texture2D(tex, vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

      result += texture2D(tex, vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
      result += texture2D(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
      result += texture2D(tex, vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

      return result;
  }

  void CalculateBloom(inout vec3 color, in vec2 coord){
    vec2 offset = vec2(0.002);
    vec3 bloom = vec3(0.0);

    float weights = 0.0;

    for(int i = 0; i < int(Bloom_Steps); i++){
      float lod = 1.0 / exp2(1.0 + i);

      vec2 bloomCoord = (coord) * lod + bloomSamplOffset[i] + 0.004 * vec2(1.0, aspectRatio);

      //if(i != Bloom_Steps - 1) continue;

      float weight = gaussianBlurWeights(float(i) / float(Bloom_Steps) + 0.0001);
      vec3 sampler = clamp01(SampleTextureCatmullRom(bloomSampler, bloomCoord, resolution).rgb);

      bloom += (sampler) * weight;
      weights += weight;
    }

    bloom /= weights;

    bloom = rgb2L(bloom * overRange);
    bloom *= 0.026 + dot(bloom, vec3(0.3847, 0.5642, 1.0)) * 0.5131;
    //bloom = pow(bloom, vec3(0.6));

    color = (bloom) * 3.0;
  }
#endif

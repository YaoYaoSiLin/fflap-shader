#version 130

#define SSR_Rendering_Scale 1.0

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform int isEyeInWater;

in vec2 texcoord;

in vec4 eyesWaterColor;

const bool gaux3Clear = false;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"

vec3 KarisToneMapping(in vec3 color){
	float a = 0.00002;
	float b = float(0xfff) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-b*luma)/(2.0*a-b-luma));
	return color;
}

vec3 invKarisToneMapping(in vec3 color){
	float a = 0.002;
	float b = float(0x2fff) / 65535.0;

	float luma = maxComponent(color);

	if(luma > a) color = color/luma*((a*a-(2.0*a-b)*luma)/(b-luma));
	return color;
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 F(vec3 F0, float cosTheta){
 return F0 + (1.0 - F0) * cosTheta;
}

float DistributionTerm( float roughness, float ndoth )
{
	float d	 = ( ndoth * roughness - ndoth ) * ndoth + 1.0;
	return roughness / ( d * d * Pi );
}

float VisibilityTerm( float roughness, float ndotv, float ndotl )
{
	float gv = ndotl * sqrt( ndotv * ( ndotv - ndotv * roughness ) + roughness );
	float gl = ndotv * sqrt( ndotl * ( ndotl - ndotl * roughness ) + roughness );
	return min(1.0, 0.5 / max( gv + gl, 0.00001 ));
}

void CalculateBRDF(out vec3 f, out float g, out float d, in float roughness, in float metallic, in vec3 F0, in vec3 normal, in vec3 view, in vec3 L){
	vec3 h = normalize(L + view);

	float ndoth = clamp01(dot(normal, h));
	float vdoth = pow5(1.0 - clamp01(dot(view, h)));
	float ndotl = clamp01(dot(L, normal));
	float ndotv = 1.0 - clamp01(dot(view, normal));

	f = F(F0, vdoth);
	d = DistributionTerm(roughness, ndoth);
	g = VisibilityTerm(d, ndotv, ndotl);
	//g /= 4.0 * max(0.0, dot(normal, L)) * max(0.0, dot(normal, view));
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

  if(texture(depthtex0, texcoord).x < 0.7) velocity *= 0.001;

  return velocity;
}

// https://software.intel.com/en-us/node/503873
vec3 RGB_YCoCg(vec3 c)
{
  // Y = R/4 + G/2 + B/4
  // Co = R/2 - B/2
  // Cg = -R/4 + G/2 - B/4
  //return c;
  //if(texcoord.x > 0.5) return c;
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
  //if(texcoord.x > 0.5) return c;
  return saturate(vec3(
    c.x + c.y - c.z,
    c.x + c.z,
    c.x - c.y - c.z
  ));
}

vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    // note: only clips towards aabb center (but fast!)
    vec3 center  = 0.5 * (maximum + minimum);
    vec3 extents = 0.5 * (maximum - minimum);

    // This is actually `distance`, however the keyword is reserved
    vec3 offset = color - center;

    vec3 ts = abs(extents / (offset + 0.0001));

    float t = minComponent((ts));
		t = clamp(t, 0.0, 1.0);

    return center + offset * t;
}

vec4 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
  vec4 result = vec4(0.0);
  float weights = 0.0;

	vec2 position = resolution * pixelPos;
	vec2 centerPosition = floor(position - 0.5) + 0.5;

	vec2 f = position - centerPosition;
	vec2 f2 = f * f;
	vec2 f3 = f * f2;

	float sharpen = 50.0;

	float c = sharpen  * 0.01;
	vec2 w0 =         -c  *  f3 + 2.0 * c          *  f2 - c  *  f;
	vec2 w1 =  (2.0 - c)  *  f3 - (3.0 - c)        *  f2            + 1.0;
	vec2 w2 = -(2.0 - c)  *  f3 + (3.0 - 2.0 * c)  *  f2 + c  *  f;
	vec2 w3 =          c  *  f3 - c                *  f2;
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

	/*
  int steps = 8;
  float invsteps = 1.0 / float(steps);

  for(int i = 0; i < steps; i++){
    float r = float(i) * invsteps * 2.0 * Pi;
    vec2 samplePos = vec2(cos(r), sin(r));
    float weight = gaussianBlurWeights(samplePos + 0.0001);
    samplePos = samplePos * pixel + pixelPos;

    vec4 sampler = texture2D(tex, samplePos);
    result += sampler * weight;
    weights += weight;
  }

  result /= weights;
	*/
  //result.rgb = RGB_YCoCg(result.rgb);

  return result;
}

float GetRayDepth(in vec2 coord){
	return texture2D(gaux1, coord * SSR_Rendering_Scale).a;
}

vec3 GetClosestRayDepth(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      //float neighbor = texture(depthtex0, texcoord).x;
      float neighbor = GetRayDepth(coord + neighborhood);

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;

  return closest;
}

vec3 GetClosest(in vec2 coord, in float scale){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      //float neighbor = texture(depthtex0, texcoord).x;
      float neighbor = texture(depthtex0, coord + neighborhood).x;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood;
      }
    }
  }

  closest.xy += coord;
	closest.xy *= scale;

  return closest;
}

vec3 CalculateReflection(in sampler2D sampler, in vec3 currentColor, in vec2 coord){
  vec2 coord_jittering = coord + jittering * pixel;

  vec3 minColor = vec3(1.0);
  vec3 maxColor = vec3(-1.0);

  for(float i = -2.0; i <= 2.0; i += 1.0){
    for(float j = -2.0; j <= 2.0; j += 1.0){
      vec2 offset = vec2(i, j);
      vec3 color = RGB_YCoCg(texture2D(sampler, (coord_jittering * resolution + offset / sqrt(1.0)) * pixel).rgb);

      minColor = min(minColor, color);
      maxColor = max(maxColor, color);
    }
  }

  vec3 closest = GetClosestRayDepth(texcoord);
  //     closest.xy = texcoord.xy + vec2(0.01);
  //float border = float(floor(closest.xy) == vec2(0.0));

  vec2 velocity = GetMotionVector(closest);
	vec2 previousCoord = texcoord.xy - velocity;

	float motion = length(velocity*resolution);
	//		  motion = min(0.5, motion);
				motion = step(0.001, motion);

  //vec3 currentColor = RGB_YCoCg(texture2D(sampler, coord_jittering).rgb);
	currentColor = RGB_YCoCg(currentColor);
  vec3 previousColor = RGB_YCoCg(ReprojectSampler(gaux3, previousCoord).rgb);

	//minColor = mix(minColor, previousColor, 0.9);
	//maxColor = mix(maxColor, previousColor, 0.9);

	//minColor = min(minColor, mix(previousColor, currentColor, 0.5));
	//maxColor = max(maxColor, mix(previousColor, currentColor, 0.5));

  vec3 weightA = vec3(0.95 - motion * 0.0625) * float(floor(previousCoord) == vec2(0.0));
  vec3 weightB = vec3(1.0 - weightA);

  vec3 blend = clamp01(abs(maxColor - minColor) / currentColor);

	vec3 antialiased = previousColor;
			 antialiased = clipToAABB(antialiased, minColor, maxColor);
			 antialiased = (currentColor * weightB + antialiased * weightA);
			 //antialiased = mix(antialiased, reflection, mix(vec3(0.7071), vec3(0.99), blend) * mix(1.0, 0.7071, motion));
  	 	 antialiased = YCoCg_RGB(antialiased);

	return antialiased;
}

vec3 SpecularReflectionResolve(in vec2 coord){
	vec3 color;
	float totalWeight;

	//if(texcoord.x > 0.5)
	coord = round(coord * resolution) * pixel;

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			vec2 samplePosition = vec2(i, j) - 1.5;
			float weight = gaussianBlurWeights(samplePosition + 0.001);

			samplePosition = samplePosition * pixel * SSR_Rendering_Scale + coord;
			vec3 sampleColor = texture2D(gaux1, samplePosition).rgb;

			color += sampleColor * weight;
			totalWeight += weight;
		}
	}

	color /= totalWeight;

	return color;
}

void main() {
	vec2 coord = texcoord;
			 //coord = GetClosest(coord, SSR_Rendering_Scale).st;
			 coord *= SSR_Rendering_Scale;
	vec2 unjitterUV = texcoord + jittering * pixel;

	vec3 albedo = texture2D(gcolor, texcoord).rgb;

  vec3 normalSurface = normalDecode(texture2D(composite, texcoord).xy);
  vec3 normalVisible = normalDecode(texture2D(gnormal, texcoord).xy);

	float smoothness = texture2D(gnormal, texcoord).r;
	float metallic   = texture2D(gnormal, texcoord).g;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, albedo.rgb, step(0.5, metallic));

	float depth = texture2D(depthtex0, texcoord).x;
	vec3 vP = vec3(texcoord, depth) * 2.0 - 1.0;
			 vP = nvec3(gbufferProjectionInverse * nvec4(vP));
	vec3 nvP = normalize(vP);

  float viewLength = length(vP.xyz);

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).zw);

	vec3 nreflectVector = normalize(reflect(nvP, normal));

	vec3 color = texture2D(gaux2, texcoord).rgb;

	float g = 0.0;
	float d = 0.0;
	vec3 f = vec3(0.0);
	CalculateBRDF(f, g, d, roughness, metallic, F0, normal, -nvP, nreflectVector);
	float brdf = max(0.0, g * d);
	//f = F(F0, pow5(1.0 - max(0.0, dot(normalize(nreflectVector-nvP), -nvP))));

	vec3 reflection = texture2D(gaux1, coord).rgb;
			 //reflection = SpecularReflectionResolve(coord);
			 reflection = CalculateReflection(gaux1, reflection, coord);

  vec3 antialiased = reflection;
	//reflection *= f;

	bool isSky = bool(step(texture(gdepth, texcoord).z, 0.999));

  //reflection = KarisToneMapping(reflection);
	//color = vec3(0.0);
	if(isEyeInWater == 0)
	color += reflection / overRange * step(texture(gdepth, texcoord).z, 0.999);

/* DRAWBUFFERS:56 */
  gl_FragData[0] = vec4(color, 1.0);
  gl_FragData[1] = vec4(antialiased, 1.0);
}

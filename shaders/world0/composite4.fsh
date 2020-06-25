#version 130

#define SSR_Rendering_Scale 0.5

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
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);;

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"

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

  if(texture(depthtex0, texcoord).x < 0.7) velocity *= 0.0001;

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
    float t = clamp(minComponent(ts), 0.0, 1.0);
    return center + offset * t;
}

vec4 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
  vec4 result = vec4(0.0);
  float weights = 0.0;

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 samplePos = pixelPos + vec2(i, j) * pixel * 0.159;

      vec4 sampler = texture2D(tex, samplePos);

      float weight = gaussianBlurWeights(vec2(i, j) * 0.159 + 0.0001);

      result += sampler * weight;

      weights += weight;
    }
  }

  result /= weights;
  //result.rgb = RGB_YCoCg(result.rgb);

  return result;
}

vec3 GetClosest(in vec2 coord){
  vec3 closest = vec3(0.0, 0.0, 1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 neighborhood = vec2(i, j) * pixel;
      float neighbor = texture2D(gaux1, coord * SSR_Rendering_Scale + neighborhood).a;

      if(neighbor < closest.z){
        closest.z = neighbor;
        closest.xy = neighborhood / SSR_Rendering_Scale;
      }
    }
  }

  closest.xy = coord;

  return closest;
}

void CalculateReflection(inout vec3 reflection, in vec2 coord){
  vec2 coord_jittering = coord + jittering * pixel;

  vec3 minColor = vec3(1.0);
  vec3 maxColor = vec3(-1.0);

  for(float i = -1.0; i <= 1.0; i += 1.0){
    for(float j = -1.0; j <= 1.0; j += 1.0){
      vec2 offset = vec2(i, j) * pixel;
      vec3 color = RGB_YCoCg(texture2D(gaux1, coord_jittering + offset).rgb);

      minColor = min(minColor, color);
      maxColor = max(maxColor, color);
    }
  }

  vec3 closest = GetClosest(texcoord);
  //     closest.xy = texcoord.xy + vec2(0.01);
  //float border = float(floor(closest.xy) == vec2(0.0));

  vec2 velocity = GetMotionVector(closest);
	vec2 previousCoord = texcoord.xy - velocity;

	float weight = 0.99;
        //weight -= motionScale * 0.074999;
				//weight *= float(floor(previousCoord) == vec2(0.0));

  vec3 currentColor = RGB_YCoCg(texture2D(gaux1, coord_jittering).rgb);

  vec3 previousColor = RGB_YCoCg(texture2D(gaux3, previousCoord).rgb);
       previousColor = clipToAABB(previousColor, minColor, maxColor);

  vec3 weightA = vec3(0.001);
  vec3 weightB = vec3(1.0 - weightA);
  /*
  vec3 blend = clamp01(abs(maxColor - minColor) / currentColor);
  weightB = lerq(vec3(0.99), vec3(0.999), blend);
  weightA = 1.0 - weightB;
  */

  //weightB = mix(vec3(0.97), vec3(0.999), clamp01((maxColor - minColor) / currentColor));
  //weightA = 1.0 - weightB;

  //reflection = (0.001 * currentColor + 0.999 * previousColor);
  reflection = (currentColor * weightA + previousColor * weightB);
  reflection = YCoCg_RGB(reflection);

  //reflection = clipToAABB(previousColor, minColor, maxColor);
  //reflection = mix(reflection, previousColor, weight);
  //reflection = YCoCg_RGB(reflection);
  //reflection += (reflection - previousColor) * 0.0025;
}

void main() {
	vec2 coord = texcoord * SSR_Rendering_Scale;
	vec2 unjitterUV = texcoord + jittering * pixel;

	vec3 albedo = texture2D(gcolor, texcoord).rgb;

  vec3 normalSurface = normalDecode(texture2D(composite, texcoord).xy);
  vec3 normalVisible = normalDecode(texture2D(gnormal, texcoord).xy);

	float smoothness = texture2D(gnormal, texcoord).b;
	float metallic   = texture2D(composite, texcoord).b;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, albedo.rgb, step(0.5, metallic));

	float depth = texture2D(depthtex0, texcoord).x;
	vec3 vP = vec3(texcoord, depth) * 2.0 - 1.0;
			 vP = nvec3(gbufferProjectionInverse * nvec4(vP));
	vec3 nvP = normalize(vP);

  float viewLength = length(vP.xyz);

  float ndotv = dot(nvP, normalSurface);
  if(-0.15 > ndotv) normalVisible = normalSurface;
  vec3 normal = normalVisible;

	vec3 nreflectVector = normalize(reflect(nvP, normal));

	vec3 color = texture2D(gaux2, texcoord).rgb;

	vec3 reflection = vec3(0.0);
	CalculateReflection(reflection, coord);
  //reflection = color*overRange;

  vec3 antialiased = reflection;

	float g = 0.0;
	float d = 0.0;
	vec3 f = vec3(0.0);
	CalculateBRDF(f, g, d, roughness, metallic, F0, normal, -nvP, nreflectVector);
	float brdf = clamp01(g * d);

	reflection = mix(color, reflection, brdf);
  reflection *= brdf * f;

  //color = vec3(0.0);
  /*
  if(isEyeInWater == 1){
  float scatteringFactor = 1.0 - min(1.0, exp(-eyesWaterColor.a * viewLength * Pi));
  reflection = reflection * (1.0 - scatteringFactor);

  vec3 absorption = min(1.0, eyesWaterColor.a * viewLength * Pi) * (1.0 - eyesWaterColor.rgb);
       absorption = 1.0 - exp(-(absorption));
  reflection -= reflection * absorption;
  }
*/
	color += reflection / overRange * step(texture(gdepth, texcoord).z, 0.999);
  //color = texture2D(gaux1, texcoord*0.5).rgb / overRange;

  //if(checker < 0.5) color = vec3(0.0);


/* DRAWBUFFERS:56 */
  gl_FragData[0] = vec4(color, 1.0);
  gl_FragData[1] = vec4(antialiased, 1.0);
}

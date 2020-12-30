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

uniform sampler2D depthtex2;

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

in float fading;

in vec2 texcoord;

in vec3 sunLightingColorRaw;

in vec4 eyesWaterColor;


vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel      = 1.0 / vec2(viewWidth, viewHeight);

#define Gaussian_Blur

#include "../libs/common.inc"
#include "../libs/dither.glsl"
#include "../libs/jittering.glsl"
#include "../libs/brdf.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
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
    float t = clamp(minComponent(ts), 0.0, 1.0);
    return center + offset * t;
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

vec3 ReprojectSampler(in sampler2D tex, in vec2 coord){
	#if 0
	return texture2D(tex, coord).rgb;
	#else
	vec3 color = vec3(0.0);
	float totalWeight = 0.0;

	//if(texcoord.x > 0.5)
	coord = round(coord * resolution) * pixel;

	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			vec2 samplePosition = vec2(i, j) - 1.5;
			float weight = gaussianBlurWeights(samplePosition + 0.001);

			samplePosition = samplePosition * pixel * SSR_Rendering_Scale + coord;
			vec3 sampleColor = texture2D(tex, samplePosition).rgb;

			color += sampleColor * weight;
			totalWeight += weight;
		}
	}

	color /= totalWeight;

	return color;
	#endif
}

vec4 ImportanceSampleGGX(in vec2 E, in float roughness){
  roughness *= roughness;

  float Phi = E.x * 2.0 * Pi;
  float CosTheta = sqrt((1 - E.y) / ( 1 + (roughness - 1) * E.y));
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

  vec3 H = vec3(cos(Phi) * SinTheta, sin(Phi) * SinTheta, CosTheta);
	float D = DistributionTerm(roughness, CosTheta) * CosTheta;

  return vec4(H, D);
}

vec3 mulTBN(in vec3 texture, in vec3 normal){
  vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

  vec3 upVector = abs(worldNormal.z) < 0.4999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  upVector = mat3(gbufferModelView) * upVector;

  vec3 t = normalize(cross(upVector, normal));
  vec3 b = cross(normal, t);
  mat3 tbn = mat3(t, b, normal);

  return normalize(tbn * texture);
}

vec3 KarisToneMapping(in vec3 color){
	float a = 0.0027;
	float b = float(0x9fff) / 65535.0;

	float lum = maxComponent(color);

	if(bool(step(lum, a))) return color;

	return color/lum*((a*a-b*lum)/(2.0*a-b-lum));
}

void main() {
	vec2 coord = texcoord;
			 //coord = GetClosest(coord, SSR_Rendering_Scale).st;
			 //coord *= SSR_Rendering_Scale;
	vec2 unjitterUV = texcoord + jittering * pixel;

	vec3 albedo = texture2D(gcolor, texcoord).rgb;
			 albedo = decodeGamma(albedo);

	//vec3 flatNormal = normalDecode(texelFetch(gnormal, ivec2(round(texcoord * resolution * 0.5) * 2.0), 0).xy);
	vec3 flatNormal = normalDecode(texture2D(gnormal, texcoord).xy);
	vec3 texturedNormal = normalDecode(texture2D(composite, texcoord).xy);
	//vec3 texturedNormal = normalDecode(texelFetch(composite, ivec2(round(texcoord * resolution * 0.5) * 2.0), 0).xy);
  vec3 visibleNormal = texturedNormal;
  if(bool(step(texture2D(gcolor, texcoord).a, 0.99))) flatNormal = texturedNormal;

  vec2 specularPackge = unpack2x8(texture(composite, texcoord).b);
	float smoothness = specularPackge.x;
	float metallic   = specularPackge.y;
	float roughness  = 1.0 - smoothness;
				roughness  = roughness * roughness;

	vec3 F0 = vec3(max(0.02, metallic));
			 F0 = mix(F0, albedo.rgb, step(0.5, metallic));

	bool isSky = bool(step(texture(gdepth, texcoord).z, 0.999));

	float depth = texture2D(depthtex0, texcoord).x;
	vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0));
	vec3 nvP = normalize(vP);
  float viewLength = length(vP.xyz);

	if(bool(step(dot(-nvP, texturedNormal), 0.2))) visibleNormal = flatNormal;
  vec3 normal = visibleNormal;

	vec3 rayDirection = normalize(reflect(nvP, normal));
			 rayDirection = (texture2D(gdepth, coord).rgb * 2.0 - 1.0);

	vec3 color = texture2D(gaux2, texcoord).rgb;

	vec2 E = vec2(GetBlueNoise(depthtex2, texcoord, resolution.y, jittering),
							  GetBlueNoise(depthtex2, 1.0 - texcoord, resolution.y, jittering));
			 E.y = ApplyBRDFBias(E.y);

	vec4 rayPDF = ImportanceSampleGGX(E, roughness);
			 rayPDF.xyz = mulTBN(rayPDF.xyz, normal);

	float totalWeight = 0.0;
	vec3 tex = vec3(0.0);

	//rayDirection = normalize(directionSum);
	//if(dot(-rayDirection, normal) < 0.2) rayDirection = normalize(rayDirection + normalize(reflect(nvP, normal)));

	//rayDirection = texelFetch(gdepth, ivec2(round(texcoord * resolution * 0.5)), 0).rgb * 2.0 - 1.0;
	//tex = decodeGamma(texelFetch(gaux1, ivec2(round(texcoord * resolution * 0.5)), 0).rgb);

	//if(dot(rayDirection, normal) < 0.2)
	//rayDirection = normalize(reflect(nvP, normal));

	float g, d = 0.0;
	vec3 f = vec3(0.0);
	FDG(f, g, d, -nvP, rayDirection, normal, (F0), roughness);
	float c = 4.0 * abs(dot(rayDirection, normal)) * abs(dot(-nvP, normal)) + 1e-5;

	float ndotl = abs(dot(rayDirection, normal));

	vec3 h = normalize(normalize(reflect(nvP, normal)) - nvP);
	float fr = g * d;

	coord = min(texcoord * 0.5, 0.5 - pixel);
	//coord = texcoord;

	vec3 specular = decodeGamma(texture2D(gaux1, coord).rgb);
	float specularDepth = texture2D(gaux1, coord).a;

	for(float i = -2.0; i <= 2.0; i += 1.0){
		for(float j = -2.0; j <= 2.0; j += 1.0){
			vec2 offset = vec2(i, j) * mix(2.0, 0.5, smoothness);
			vec2 coordoffset = coord + offset * pixel;

			//float ndotv = dot(normalize(offset), normal.xy);
			//if(bool(step(ndotv, 0.01)) || bool(step(0.9999, texture(depthtex0, coordoffset * 2.0).x))) continue;

			vec3 stepRayDirection = (texture2D(gdepth, coordoffset).rgb * 2.0 - 1.0);

			float ndotv = dot(stepRayDirection, normal);
			if(bool(step(ndotv, 1e-5))) continue;

			//float brdf = CalculateBRDF(-nvP, stepRayDirection, normal, roughness);
			//float pdf = texture2D(gdepth, coordoffset).a * 128.0;

			float targetRoughness = pow2(1.0 - unpack2x8(texture(composite, texcoord).b).x);

			float weight = (1.0 - abs(targetRoughness - roughness));
			totalWeight += weight;

			vec3 colorSample = decodeGamma(texture2D(gaux1, coordoffset).rgb);
			tex += colorSample * weight;
		}
	}

	specular = tex / totalWeight;

	//vec3 
	//specular = decodeGamma(texture2D(gaux1, coord).rgb);//texcoord.x > 0.5 ? fr : tex;
	//float specularDepth = texture2D(gaux1, texcoord).a;

	specular = encodeGamma(specular);

	/* DRAWBUFFERS:4 */
	gl_FragData[0] = vec4(specular, specularDepth);
}

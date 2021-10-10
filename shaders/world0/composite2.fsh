#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D gaux3;
uniform sampler2D colortex8;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform sampler2D shadowtex0;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

uniform mat4 shadowModelViewInverse;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform vec3 lightVectorView;
uniform vec3 upVectorView;
uniform vec3 lightVectorWorld;
uniform vec3 sunVectorWorld;
uniform vec3 moonVectorWorld;

uniform float aspectRatio;
uniform vec2 resolution;
uniform vec2 pixel;

uniform float frameTimeCounter;

uniform int isEyeInWater;

in vec2 texcoord;

in float fading;
in vec3 skyLightingColorRaw;
in vec3 sunLightingColorRaw;

const bool gaux3Clear = false;
const bool colortex8Clear = false;

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}

vec2 float32x3_to_oct(in vec3 v) {
     v = v.xzy;

    // Project the sphere onto the octahedron, and then onto the xy plane
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}

vec3 oct_to_float32x3(vec2 e) {
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    return normalize(v.xzy);
}

#include "/libs/common.inc"
#include "/libs/dither.glsl"
#include "/libs/lighting.glsl"

vec2 CalculateVolocity(in vec3 coord){
  vec4 position = gbufferProjectionInverse * nvec4(coord * 2.0 - 1.0);
       position /= position.w;
       position = gbufferModelViewInverse * position;
       position.xyz += cameraPosition - previousCameraPosition;
       position = gbufferPreviousModelView * position;
       position = gbufferPreviousProjection * position;
       position /= position.w;
       position.xyz = position.xyz * 0.5 + 0.5;
  vec2 velocity = coord.xy - position.xy;

  if(texture(depthtex0, coord.xy).x < 0.7) velocity *= 0.01;

  //z = position.z;

  return velocity;
}
#if 0
void main() {
     vec2 coord = (texcoord);

     float depth0 = texture(depthtex0, coord).x;

     vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, depth0) * 2.0 - 1.0));
     vec4 wP = (gbufferModelViewInverse) * nvec4(vP);
     vec3 viewDirection = normalize(vP);
     vec3 eyeDirection = -viewDirection;
     float viewLength = length(vP);

     vec3 albedo = decodeGamma(texture2D(gcolor, coord).rgb);

     vec3 texturedNormal = normalDecode(texture2D(composite, coord).xy);
     vec3 geometryNormal = texture2D(gcolor, coord).a > 0.99 ? texturedNormal : normalDecode(texture2D(gnormal, coord).xy);
     vec3 visibleNormal = dot(eyeDirection, texturedNormal) < 0.2 ? geometryNormal : texturedNormal;

     float tileMaterial  = round(texture2D(gnormal, coord).z * 255.0);

     bool isSky      = bool(step(254.5, tileMaterial));

     vec2 specularPackge = unpack2x8(texture2D(composite, coord).b);

     float roughness = pow2(1.0 - specularPackge.x);
     float metallic  = specularPackge.y;
     float material  = floor(texture2D(composite, coord).a * 255.0);
     float metal = step(isMetallic, metallic);
     vec3 F0 = mix(vec3(max(0.02, metallic)), albedo.rgb, metal);

     vec3 color = decodeGamma(texture2D(gaux2, coord).rgb);

     vec2 coord2 = min(vec2(0.5) - pixel, texcoord * 0.5) - jitter;

     float rayDepth = texture2D(gaux1, coord2).x;

     vec2 velocity = CalculateVolocity(vec3(texcoord, rayDepth));
     vec2 previousCoord = texcoord - velocity; //previousCoord = round(previousCoord * resol)
     float velocityLength = length(velocity * resolution);

     float blend = 0.95 * step(max(abs(previousCoord.x - 0.5), abs(previousCoord.y - 0.5)), 0.5);

     vec3 previous_ = texture2D(colortex8, previousCoord).rgb;

     vec3 previous_worldNormal = oct_to_float32x3(previous_.rg);

     vec3 worldNormal = texture2D(gcolor, texcoord).a > 0.99 ? normalDecode(texture2D(composite, texcoord).xy) : normalDecode(texture2D(gnormal, texcoord).xy);
     //vec3 worldNormal = normalDecode(texture2D(gnormal, coord2 * 2.0).xy);
          worldNormal = mat3(gbufferModelViewInverse) * worldNormal;
          worldNormal = abs(worldNormal);
          worldNormal = saturate(worldNormal);

     vec3 test_worldNormal = worldNormal / (max(worldNormal.x, max(worldNormal.y, worldNormal.z)) + 1e-5);

     float penumbra = saturate(abs(viewLength - previous_.z) / max(1e-5, min(viewLength, previous_.z)) * 16.0);
     
     float visibility = saturate(dot(test_worldNormal, previous_worldNormal));

     //blend *= visibility;
     //blend *= 1.0 - max(penumbra, step(0.05, velocityLength) * 0.2);
     //blend *= 1.0 - penumbra;
     //blend *= 1.0 - step(0.05, velocityLength) * 0.2;
     //blend = 0.0;

     vec3 pc = texture2D(colortex8, previousCoord).rgb;
     vec3 pp = nvec3(gbufferProjectionInverse * nvec4(pc * 2.0 - 1.0));
     vec3 cc = vec3(texcoord, depth0);
     vec3 cp = nvec3(gbufferProjectionInverse * nvec4(cc * 2.0 - 1.0));

     float test0 = max(0.05, length(pp));
     float test1 = length(cp);

     //blend *= 1.0 - saturate(abs(test0 - test1) / min(test0, test1) * 128.0);
     blend *= 1.0 - saturate(length(pp - cp) * 16.0 / test1);

     vec3 previousSample = decodeGamma(texture2D(gaux3, previousCoord).rgb);

     vec3 specular = decodeGamma(texture2D(gaux1, coord2).rgb);
     specular = mix(specular, previousSample, blend);
/*
     vec3 specular = vec3(0.0);

     vec3 m1 = vec3(0.0);
     vec3 m2 = vec3(0.0);
     float total = 0.0;

     for(float i = -1.0; i <= 1.0; i += 1.0) {
          for(float j = -1.0; j <= 1.0; j += 1.0) {
               vec2 offset = vec2(i, j);
               vec2 position = min(coord2 + offset * pixel, vec2(0.5) - pixel);

               vec3 m = texture2D(gaux1, position).rgb;

               vec3 sampleNormal = normalDecode(texture2D(composite, position * 2.0).xy);
               float normalWeight = 1e-5 + rescale(0.99999, 1.0, dot(sampleNormal, texturedNormal));

               total += normalWeight;

               m1 += m * normalWeight;
               m2 += m * m;
          }
     }

     m1 /= total;
     m2 /= 9.0;

     vec3 v = sqrt(abs(m2 - m1 * m1));

     specular = m1;
     specular = mix(specular, previousSample, blend);
*/
     vec3 rayDirection = normalize(reflect(viewDirection, texturedNormal));// (smoothness > 0.7 ? visibleNormal : geometryNormal)
     if(dot(rayDirection, texturedNormal) < 0.0) rayDirection = normalize(reflect(viewDirection, geometryNormal));

     vec3 fr = SpecularLighting(vec4(albedo, 1.0), rayDirection, eyeDirection, texturedNormal, texturedNormal, F0, roughness, metallic, (material < 64.5 ? 0.0 : material), true);
    
     //color = vec3(0.0);
     color += isSky ? vec3(0.0) : specular;
     //color = specular;

     color = encodeGamma(color);
     specular = encodeGamma(specular);

     //color = texcoord.x > 0.5 ? worldNormal : specular;

     //color = vec3(blend);
     //color = saturate(previous_worldvelocity);

     gl_FragData[0] = vec4(color, 1.0);

     //gl_FragData[0] = vec4(color, 1.0);
     gl_FragData[1] = vec4(specular, 1.0);

     test_worldNormal = vec3(float32x3_to_oct(test_worldNormal), viewLength);
     previous_worldNormal = vec3(float32x3_to_oct(previous_worldNormal), previous_.z);

     gl_FragData[2].rgb = mix(cc, pc, blend);
     gl_FragData[2].a = 1.0;

     //gl_FragData[2] = vec4(mix(test_worldNormal, previous_worldNormal, vec3(blend)), 1.0);
     //gl_FragData[2].rg = saturate(gl_FragData[2].rg);
}
#else

vec3 RGB_YCoCg(vec3 c) {
	return vec3(
		 c.x/4.0 + c.y/2.0 + c.z/4.0,
		 c.x/2.0 - c.z/2.0,
		-c.x/4.0 + c.y/2.0 - c.z/4.0
	);
}

vec3 YCoCg_RGB(vec3 c) {
     c = saturate(vec3(
		c.x + c.y - c.z,
		c.x + c.z,
		c.x - c.y - c.z
	));

     return c;
}

void ResolverColor(in vec2 coord, out vec3 resultColor, out float variance) {
     vec4 result = vec4(0.0);
     float total = 0.0;

     float m1 = (0.0);
     float m2 = (0.0);

     //coord = coord * vec2(1.0, 2.0);

     vec2 texelCoord = coord * vec2(0.5, 0.5);

     float depth = linearizeDepth(texture(depthtex0, coord).x * 2.0 - 1.0);
     //vec3 color = (texture2D(gnormal, coord * vec2(0.5)).rgb);
     //float luminance = maxComponent(color);

     vec4 centerSample = texcoord.x < 0.5 ? (texture2D(gdepth, texelCoord)) : (texture2D(gnormal, texelCoord) * vec4(vec3(1.0), 255.0));
          centerSample.rgb = decodeGamma(centerSample.rgb);

     //for(float i = -2.0; i <= 2.0; i += 1.0) {
          for(float j = -2.0; j <= 2.0; j += 1.0) {  
               vec2 texelPosition = min(texelCoord + vec2(0.0, j) * pixel, 0.5 - pixel);

               vec4 sampleColor = texcoord.x < 0.5 ? (texture2D(gdepth, texelPosition)) : (texture2D(gnormal, texelPosition) * vec4(vec3(1.0), 255.0));
               //float luminance_weight = saturate(1.0 - abs(maxComponent(sampleColor) - luminance));

               float sampleDepth = linearizeDepth(texture2D(gaux1, texelPosition).x * 2.0 - 1.0);
               float depth_weight = saturate(1.0 - abs(depth - sampleDepth) / sampleDepth * 40.0);

               float weight = depth_weight * (1e-5 + sampleColor.a);

               total += weight;
               resultColor += decodeGamma(sampleColor.rgb) * weight;
               float sampleLum = dot(sampleColor.rgb, vec3(1.0 / 3.0));

               m1 += sampleLum * weight;
               m2 += sampleLum * sampleLum * weight;
          }
     //}

     if(total > 0.0) {
          m1 /= total;
          m2 /= total;
          resultColor /= total;
     }else {
          resultColor = centerSample.rgb;
     }

     if(centerSample.a > 100.0) {
          resultColor = centerSample.rgb;
     }

     variance = sqrt(m2 - m1 * m1);

     resultColor = encodeGamma(resultColor);
}

vec3 GetClosest(in vec2 coord) {
     vec3 result = vec3(0.0, 0.0, 1.0);

     for(float i = -2.0; i <= 2.0; i += 1.0) {
          for(float j = -2.0; j <= 2.0; j += 1.0) {
               vec2 offset = vec2(i, j);
               vec2 position = clamp(coord + offset * pixel, pixel, 1.0 - pixel);
               vec2 halfposition = clamp(coord * 0.5 + offset * pixel, pixel, 0.5 - pixel);

               //float depth = texture(depthtex0, position).x;
               float depth = texcoord.x < 0.5 ? texture(depthtex0, position).x : texture(gaux1, halfposition).y;

               if(depth < result.z) {
                    result = vec3(offset, depth);
               }
          }
     }

     result.xy = result.xy * pixel;
     //result.z = texcoord.x < 0.5 ? texture(depthtex0, clamp(result.xy + coord, pixel, 1.0 - pixel)).x : texture(gaux1, clamp(result.xy + coord * 0.5, pixel, 0.5 - pixel)).y;
     result.xy += coord;

     return result;
}

void main() {
     vec2 tileCoord = fract(texcoord * vec2(2.0, 1.0));
     vec2 tile1Coord = texcoord * vec2(1.0, 0.5);
     vec2 tile2Coord = texcoord * vec2(1.0, 0.5) - vec2(0.5, 0.0);

     vec4 diffuse = vec4(0.0);//ResolverRSMGI(tile1Coord);

     //vec3 currentColor = texcoord.x < 0.5 ? diffuse.rgb : decodeGamma(texture2D(gaux1, tile2Coord).rgb);
     vec3 currentColor = vec3(0.0);
     float variance = (0.0);
     ResolverColor(tileCoord, currentColor, variance);
     //vec3 currentColor = ResolverRSMGI(tile1Coord).rgb;

     //float rayDepth = texcoord.x < 0.5 ? texture(depthtex0, tileCoord).x : texture(gaux1, tileCoord * vec2(0.5)).y;
     //vec3 closest = vec3(tileCoord, rayDepth);

     vec3 closest = GetClosest(tileCoord);
     vec2 velocity = CalculateVolocity(closest);
     vec2 previousCoord = (tileCoord - velocity); 
     vec3 previousCoord3 = vec3(previousCoord, texture(depthtex0, previousCoord).x);

     float motionLength = length(velocity * resolution);

     float velocityLength = min(1.0, length(velocity * resolution));

     float blend = 0.95 * step(max(abs(previousCoord.x - 0.5), abs(previousCoord.y - 0.5)), 0.5);

     previousCoord = clamp(previousCoord, pixel * 4.0, vec2(1.0, 1.0) - pixel * 4.0);
     previousCoord *= vec2(0.5, 1.0);

     if(texcoord.x < 0.5) {

     }else{
          previousCoord += vec2(0.5, 0.0);
     }

     vec3 previousColor = (texture2D(gaux3, previousCoord).rgb);
     if(maxComponent(previousColor.rgb) == 0.0) previousColor = currentColor;

     vec3 cc = vec3(closest.xy, closest.z);
     vec3 currentPosition = nvec3(gbufferProjectionInverse * nvec4(cc * 2.0 - 1.0));
     vec3 pc = texture2D(colortex8, previousCoord).rgb;
     vec3 previousPosition = nvec3(gbufferProjectionInverse * nvec4(pc * 2.0 - 1.0));
     vec3 previousPosition2 = nvec3(gbufferProjectionInverse * nvec4(previousCoord3 * 2.0 - 1.0));

     vec3 g_normal = normalDecode(texture2D(composite, tileCoord).xy);

     float test0 = length(previousPosition);
     float test1 = length(currentPosition);

     float cosTheta = abs(dot(normalize(previousPosition), normalize(currentPosition)));

     float sigma = texcoord.x > 0.5 ? 4.0 : 2.0;
           //sigma *= 1.0 + rescale(0.05, 1.0, variance) * 256.0;

     vec3 h = normalize(normalize(currentPosition) / 0.9999 - normalize(previousPosition2));
     float hdotn = dot(g_normal, h);

     sigma *= 1.0 + min(3.0, saturate(hdotn * 0.8 + 0.2) * 4.0);

     float lengthWeight = 1.0 - saturate(length(previousPosition - currentPosition) * sigma / test1);
     blend *= lengthWeight;

     //blend *= 1.0 - (variance);

     //blend *= saturate(dot(normalize(pp - currentPosition), g_normal) * 4.0);

     float ndotv = saturate(-dot(normalize(currentPosition), g_normal));

     //vec3 previousWorldPosition = mat3(gbufferModelViewInverse) * previousPosition2;//currentPosition * 1.0001 - mat3(gbufferPreviousModelView) * (mat3(gbufferModelViewInverse) * currentPosition + cameraPosition - previousCameraPosition);
     //mat3(gbufferModelViewInverse) * previousPosition2 + cameraPosition

     //previousWorldPosition = mat3(gbufferModelViewInverse) * (currentPosition) / 0.999 - previousWorldPosition;
     //previousWorldPosition = mat3(gbufferModelView) * previousWorldPosition;

     vec3 worldPosition = mat3(gbufferModelViewInverse) * currentPosition + cameraPosition;

     vec3 previousWorldPosition = (mat3(gbufferModelViewInverse) * previousPosition) + cameraPosition;
          previousWorldPosition = cameraPosition - previousWorldPosition;
          previousWorldPosition = mat3(gbufferModelView) * previousWorldPosition;

     float ndotv_p = saturate(dot(normalize(previousWorldPosition), (g_normal)) * 1.0);

     //blend *= rescale(0.99, 1.0, dot(normalize(previousPosition), normalize(currentPosition)));
     //blend *= rescale(0.0, 1.0, 1.0 - (ndotv - ndotv_p) * 100.0);

     //blend *= ndotv_p;

     //blend *= step(1e-5, -dot(normalize(currentPosition - previousPosition2 * 0.99), g_normal));

     //blend *= saturate(1.0 - abs(ndotv_previous - ndotv) * 100.0);

     vec3 accumulation = mix(currentColor, previousColor, blend);
          accumulation = (accumulation);

     float weight = texcoord.x > 0.5 ? texture2D(gnormal, tileCoord * vec2(0.5)).a : 0.5;

     gl_FragData[0] = vec4(accumulation, weight);
     gl_FragData[1] = vec4(vec3(0.0), texture2D(gaux1, tileCoord * vec2(0.5)).x);
     gl_FragData[2] = vec4(accumulation, 1.0);
     gl_FragData[3] = vec4(mix(cc, pc, blend), 1.0);
}
#endif
/* RENDERTARGETS: 1,4,6,8 */
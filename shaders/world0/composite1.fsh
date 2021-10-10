#version 130

#define Enabled_TAA

#define RSM_GI_Quality      8   //[4 8 12 16]
#define RSM_GI_Luminance    0.5 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0] original color <============> normalized color
#define RSM_GI_Saturation   0.5 //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0] original color <============> gray scale

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;
uniform sampler2D colortex15;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;

uniform vec3 cameraPosition;

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

#ifndef INCLUDE_SHADOW
const int   shadowMapResolution  = 2048;
const float shadowDistance       = 128.0;
const bool  generateShadowMipmap = false;

const bool  shadowHardwareFiltering = false;

const bool shadowcolor0Nearest = true;
const bool shadowcolor1Nearest = true;
const bool shadowtex0Nearest = true;
const bool shadowtex1Nearest = true;

#define SHADOW_MAP_BIAS 0.9
#define SHADOW_MAP_BIAS_Mul 0.95

uniform sampler2D shadowtex0;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

float shadowPixel = 1.0 / float(shadowMapResolution);

float ShadowCoordDistortion(in vec2 coord){
    return 1.0 / (mix(1.0, length(coord.xy), SHADOW_MAP_BIAS) / SHADOW_MAP_BIAS_Mul);
}

void CalculateShadowCoordDistortion(inout vec3 coord, inout float distortion){
    distortion = ShadowCoordDistortion(coord.xy);
    coord.xy = (coord.xy * distortion);
}

vec2 ShadowCoordResclae(in vec2 coord){
    return min(vec2(1.0), coord) * vec2(1.0, 0.5);
}

void CalculateShadowCoordRescale(inout vec3 coord){
    //coord.xy = min(vec2(1.0), coord.xy) * 0.5;
    coord.xy = ShadowCoordResclae(coord.xy);
}

vec3 RescaleShadowMapCoord(in vec3 coord){
    coord.xy *= ShadowCoordDistortion(coord.xy);
    coord = coord * 0.5 + 0.5;
    coord.xy = ShadowCoordResclae(coord.xy);

    return coord;
}

vec3 ProjectionToShadowMap(in vec3 position){
    vec3 projection = mat3(shadowModelView) * position + shadowModelView[3].xyz;
         projection = vec3(shadowProjection[0].x, shadowProjection[1].y, shadowProjection[2].z) * projection + shadowProjection[3].xyz;

    return projection;
}
#endif

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
// Assume normalized input. Output is on [-1, 1] for each component.
vec2 float32x3_to_oct(in vec3 v) {
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

vec2 unpack2x4bit(in float v) {
    v *= 255.0 / 16.0;

    float ry = floor(v);
    float rx = (v) - ry;

    return vec2((16.0 / 15.0) * rx, 1.0 / 15.0 * ry);
}

#include "/libs/common.inc"
#include "/libs/dither.glsl"
#include "/libs/lighting.glsl"
#include "/libs/volumetric/atmospheric_common.glsl"
#include "/libs/volumetric/clouds_common.glsl"
#include "/libs/volumetric/atmospheric_raymarching.glsl"

float GetDepth() {
    return texture(depthtex0, texcoord).x;
}

vec3 GetNormal() {
    return normalDecode(texture2D(composite, texcoord).xy);
}

vec3 CalculateGeometryNormal(in vec3 worldNormal) {
	vec3 n = abs(worldNormal);
	return n.x > max(n.y, n.z) ? vec3(step(0.0, worldNormal.x) * 2.0 - 1.0, 0.0, 0.0) : 
		   n.y > max(n.x, n.z) ? vec3(0.0, step(0.0, worldNormal.y) * 2.0 - 1.0, 0.0) : 
		   vec3(0.0, 0.0, step(0.0, worldNormal.z) * 2.0 - 1.0);
}

vec3 CalculateLowDetailNormal(in vec3 worldNormal, float level) {
    vec3 n = abs(worldNormal);
    return floor(n / maxComponent(n) * (level) + 1e-5) / level * vec3(step(0.0, worldNormal.x) * 2.0 - 1.0, step(0.0, worldNormal.y) * 2.0 - 1.0, step(0.0, worldNormal.z) * 2.0 - 1.0);
}

vec3 CalculateRSMGI(in vec2 coord, in float depth, in vec3 normal, in vec2 dither) {
    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0));
    vec4 wP = (gbufferModelViewInverse) * nvec4(vP);

    float viewLength = length(vP);

    if(viewLength > 64.0) return vec3(0.0);

    vec3 viewNormal = normal;
    vec3 worldGeoNormal = (mat3(gbufferModelViewInverse) * viewNormal);

    vec3 shadowSpaceNormal = mat3(shadowModelView) * worldGeoNormal;
    vec3 L = mat3(shadowModelView) * lightVectorWorld;

    float distortion = 1.0;

    vec3 shadowSpacePosition = ProjectionToShadowMap(wP.xyz);
    vec3 shadowCoord = shadowSpacePosition;
    CalculateShadowCoordDistortion(shadowCoord, distortion);
    shadowCoord = shadowCoord * 0.5 + 0.5;
    //shadowCoord.xy = floor(shadowCoord.xy * vec2(shadowMapResolution)) / vec2(shadowMapResolution);
    CalculateShadowCoordRescale(shadowCoord);

    shadowSpacePosition = shadowSpacePosition * 0.5 + 0.5;
    shadowSpacePosition.z -= shadowPixel * 2.0;

    float radius = 0.01;

    int steps = 4;
    int rounds = int(RSM_GI_Quality);

    vec3 diffuse = vec3(0.0);

    for(int j = 0; j < rounds; j++) {
    //for(int i = 0; i < steps; i++) {
        //float angle = (dither.x + float(i)) / float(steps) * 2.0 * Pi;
        //vec2 offset = ImportanceSampleGGX(vec2(angle, dither.y), 0.999).xy * radius * float(j + 1.0) / float(rounds);

        vec2 offset = ImportanceSampleGGX(dither, 0.999).xy * radius * float(j + 1.0) / float(rounds);

        vec2 coord = shadowCoord.xy + offset * distortion;
        if(abs(coord.x - 0.5) >= 1.0 || abs(coord.y - 0.25) >= 0.5) break;

        float depth = texture(shadowtex0, coord).x;
        if(depth < shadowCoord.z - shadowPixel * 2.0) continue;

        vec3 halfPosition = mat3(shadowProjectionInverse) * (vec3(shadowSpacePosition.xy + offset, depth) * 2.0 - 1.0) - mat3(shadowProjectionInverse) * (shadowSpacePosition.xyz * 2.0 - 1.0);
        vec3 direction = normalize(halfPosition);

        vec3 albedo = decodeGamma(texture2D(shadowcolor0, coord).rgb);
        vec3 luminance = vec3(maxComponent(albedo) + 1e-5);

        albedo = mix(albedo, luminance, vec3(RSM_GI_Saturation)) / mix(vec3(1.0), vec3(luminance), vec3(RSM_GI_Luminance));

        float attenuation = 16.0 / max(0.33, pow2(length(halfPosition)));

        vec3 normal = mat3(shadowModelView) * (texture2D(shadowcolor1, coord).xyz * 2.0 - 1.0);

        float cosTheta = max(0.0, dot(shadowSpaceNormal, direction)) * max(0.0, -dot(normal, direction)) * max(0.0, dot(normal, L));

        diffuse += albedo * (min(10.0, attenuation) * cosTheta);
    //}
    }

    diffuse /= float(rounds);
    //diffuse /= float(steps);

    return diffuse / (diffuse + 1.0);
}

vec3 projectionToScreen(in vec3 P){
    return nvec3(gbufferProjection * nvec4(P)) * 0.5 + 0.5;
}

vec2 ScreenSpaceRayMarching(in vec3 rayOrigin, in vec3 rayDirection){
    vec2 hitUV = vec2(0.0);

    int count;

    int steps = 20;
    float invsteps = 1.0 / float(steps);

    //rayDirection *= 20.0 * invsteps;

    float stepLength = 1.4;

    vec3 testPosition = rayOrigin;
    vec3 direction = rayDirection * 0.1;// * mix(dither, 1.0, 0.7071) * 0.5;

    float thickness = near / far * 4.0;

    int piercetime = 0;

    for(int i = 0; i < steps; i++){
        testPosition += direction;

        vec3 screenCoord = projectionToScreen(testPosition);

        vec2 coord = screenCoord.xy;
        if(max(abs(coord.x - 0.5), abs(coord.y - 0.5)) > 0.5) break;// || screenCoord.z > 0.99999

        float sampleDepth = texture(depthtex0, coord).x;
        vec3 samplePosition = nvec3(gbufferProjectionInverse * vec4(vec3(coord, sampleDepth) * 2.0 - 1.0, 1.0));

        float testDepth = linearizeDepth(screenCoord.z * 2.0 - 1.0);
        float frontDepth = linearizeDepth(sampleDepth * 2.0 - 1.0);
        float difference = testDepth - frontDepth; 

        if(difference > 0.0){
            if(difference < thickness + testDepth * far * 0.0005) {// + (testDepth) / far * 16.0
                return coord;
            }else{
                testPosition -= direction;
                direction *= 0.15;
            }
        }else{
            direction *= stepLength;
        }
    }

    return vec2(-1.0);
}

vec3 wP2sP(in vec3 wP){
    vec3 shadowSpacePosition = ProjectionToShadowMap(wP.xyz);
    vec3 shadowCoord = shadowSpacePosition;
    shadowCoord.xy *= ShadowCoordDistortion(shadowCoord.xy);
    shadowCoord = shadowCoord * 0.5 + 0.5;
    CalculateShadowCoordRescale(shadowCoord);

	return shadowCoord;
}

vec3 Vec3ToDualParaboloid(in vec3 p){
    p = normalize(p * vec3(1.0 / aspectRatio, 1.0, 1.0));

	//vec2 coord = p.xy / (1.0 + (-p.z) + 0.01) * 0.5 + 0.5;

    vec2 coord0 = p.xy / (1.0 + (-p.z) + 0.01) * 0.5 + 0.5;
         coord0 = vec2(coord0.x, 1.0 - coord0.y) * vec2(0.25, 0.25) + vec2(0.125, 0.625);
         coord0 = clamp(coord0, vec2(0.0, 0.5), vec2(0.5, 1.0));

    vec2 coord1 = p.xy / (1.0 + (p.z) + 0.01) * 0.5 + 0.5;
         coord1 = coord1 * vec2(0.25, 0.25) + vec2(0.625, 0.625);
         coord1 = clamp(coord1, vec2(0.5, 0.5), vec2(1.0, 1.0));

    //coord0 = round(coord0 * vec2(shadowMapResolution)) / vec2(shadowMapResolution);
    //coord1 = round(coord1 * vec2(shadowMapResolution)) / vec2(shadowMapResolution);

    float depth0 = texture(shadowtex0, coord0).x;
    float depth1 = texture(shadowtex0, coord1).x;

    return depth0 < depth1 ? vec3(coord0, depth0) : vec3(coord1, depth1);

    //return true ? vec2(coord.x, 1.0 - coord.y) * vec2(0.25, 0.25) + vec2(0.125, 0.625) : coord * vec2(0.25, 0.25) + vec2(0.625, 0.625);
    //return vec2(coord.x, 1.0 - coord.y) * vec2(0.25, 0.25) + vec2(0.125, 0.625);
    //return p.z < 0.0 ? vec2(0.0) : coord * vec2(0.25, 0.25) + vec2(0.625, 0.625);
}

void CalcualteDualParaboloid(inout vec2 coord0, inout vec2 coord1, in vec3 p) {
    p = normalize(p * vec3(1.0 / aspectRatio, 1.0, 1.0));

    coord0 = p.xy / (1.0 + (-p.z) + 0.01) * 0.5 + 0.5;
    coord1 = p.xy / (1.0 + ( p.z) + 0.01) * 0.5 + 0.5;
}

vec2 IntersectCube(in vec3 shapeCenter, in vec3 direction, in vec3 size) {
    vec3 dr = 1.0 / direction;
    vec3 n = shapeCenter * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    //if(far > near && far > 0.0) {
        return vec2(near, far);
    //}else{
    //    return vec2(-1.0);
    //}
}

vec4 EnvironmentMapReflection(in vec3 origin, in vec3 direction) {
    origin = mat3(gbufferModelViewInverse) * origin;
    direction = mat3(gbufferModelViewInverse) * direction;

    float box = maxComponent(abs(origin));

    vec3 color = vec3(0.0);

    vec2 tracingPlanet = RaySphereIntersection(E, direction, vec3(0.0), rE);
    vec2 tracingAtmospheric = RaySphereIntersection(E, direction, vec3(0.0), rA);

    float start = tracingAtmospheric.x > 0.0 ? tracingAtmospheric.x : 0.0;
    float end = tracingPlanet.x > 0.0 ? tracingPlanet.x : max(0.0, tracingAtmospheric.y);

    vec2 t = RaySphereIntersection(origin, direction, vec3(0.0), 2.0 * 2.0 * length(origin));
    float dist = max(t.x, t.y);

    vec3 enbCoord3 = Vec3ToDualParaboloid((origin + direction * dist * 2.0 * 2.0).xzy);
    vec2 enbCoord2 = enbCoord3.xy;
    
    float depth = enbCoord3.z;

    vec2 lightmap = unpack2x4bit(texture2D(shadowcolor1, enbCoord2).a);//vec2(fract(encodeLightmap / 16.0), floor(encodeLightmap / 16.0) / 15.0);
    float skyLightmap = lightmap.y * 15.0;

    vec3 albedo = decodeGamma(texture2D(shadowcolor0, enbCoord2).rgb);
    float alpha = (texture2D(shadowcolor0, enbCoord2).a - 0.2) / 0.8;

    float cosTheta = max(0.0, alpha * 2.0 - 1.0);
    vec3 shadowCoord = texture2D(shadowcolor1, enbCoord2).xyz;

    float shading = step(shadowCoord.z, texture(shadowtex0, shadowCoord.xy).x + (1.0 / 256.0));

    vec3 sunLight = albedo * shading * sunLightingColorRaw * fading * cosTheta * invPi;

    float remap_skylightMap0 = skyLightmap / pow2(16.0 - skyLightmap) * 0.176775;
    float remap_skylightMap1 = skyLightmap * saturate(pow(skyLightmap / 15.0, 19.0) * 0.5) * 0.176775;
    float remap_skylightMap2 = skyLightmap * pow5(skyLightmap / 15.0) * 0.35355;

    vec3 ambientLight = (albedo.rgb * skyLightingColorRaw) * (invPi);
         ambientLight *= (remap_skylightMap0 + remap_skylightMap1 + remap_skylightMap2);

    vec3 torchLight = albedo.rgb * pow(vec3(1.022, 0.782, 0.344), vec3(2.0));
         torchLight *= max(0.0, lightmap.x * 15.0 - 1.0) * (1.0 / pow2(16.0 - lightmap.x * 15.0));
         torchLight *= invPi * 0.25;

    color = sunLight + ambientLight + torchLight;

    if(depth < 0.9999) return vec4(color, depth);

    vec3 coord = mat3(gbufferModelView) * (direction * end);
    float rayDepth = (1.0 / 1000.0) * far * near / (far - near) + 0.5 * (far + near) / (far - near) + 0.5;

    vec3 atmospheric_color = vec3(0.0);

    atmospheric_color = SimplePlanetSurface(mix(skyLightingColorRaw, sunLightingColorRaw * fading, 0.001), vec3(0.0), direction, vec3(0.0), tracingPlanet.x);
    CalculateAtmospheric(atmospheric_color, atmospheric_color, E, direction, sunVectorWorld, start, end, vec3(0.0));

    #if defined(High_Quality_Clouds) && defined(Enabled_Volumetric_Clouds)
    vec2 cloudsCoord = (float32x3_to_oct(direction) * 0.5 + 0.5) * 0.5;

    vec4 cloudsColor = texture2D(colortex15, cloudsCoord);
         cloudsColor.rgb = decodeGamma(cloudsColor.rgb);

    atmospheric_color = atmospheric_color * cloudsColor.a + cloudsColor.rgb;
    #endif

    return vec4(atmospheric_color, 0.9999);
}

vec3 CalculateImageBasedReflection(in vec3 rayOrigin, in vec3 normal, vec3 Gnormal, in vec2 dither, inout float pdf, inout float depth) {
    vec3 viewDirection = normalize(rayOrigin);
    vec3 eyeDirection = -viewDirection;

    vec2 specularPackge = unpack2x8(texture2D(composite, texcoord).b);
    float smoothness = specularPackge.x;
    float roughness = pow2(1.0 - smoothness);

    vec2 E = dither;
         E.y = ApplyBRDFBias(E.y);

    if(smoothness > 0.95) {
        //roughness = 0.0001;
        //E.y = ApplyBRDFBias(E.y);
        E.y = ApplyBRDFBias(E.y);
    }

    vec3 n = normal;
    vec3 t = normalize(cross(n, mat3(gbufferModelView) * vec3(0.0, 1.0, 1.0)));
    vec3 b = cross(n, t);
    mat3 tbn = mat3(t, b, n);

    vec4 normal_term = ImportanceSampleGGX(E, roughness);
         normal_term.xyz = normalize(tbn * normal_term.xyz);

    vec3 rayDirection = normalize(reflect(viewDirection, normal_term.xyz));

    vec4 color = vec4(vec3(0.0), 0.9999);
    
    color = EnvironmentMapReflection(rayOrigin, rayDirection);

    vec2 coord = vec2(-1.0);
    
    vec3 tracingOrigin = rayOrigin + normal * ((1e-5 + length(rayOrigin) * 0.01) / max(1.0, dot(eyeDirection, normal) * 20.0));
    coord = ScreenSpaceRayMarching(tracingOrigin, rayDirection * mix(dither.x, 1.0, 0.5) / 0.5);

    if(max(coord.x, coord.y) > 0.0) {
        color.rgb = decodeGamma(texture2D(gaux2, coord).rgb);
        color.a = texture(depthtex0, coord).x;
    }

    color.rgb /= color.rgb + 1.0;
    color.rgb = saturate(color.rgb);

    pdf = (normal_term.w / 255.0);
    //depth = mix(color.a, depth, step(0.999, linearizeDepth(depth * 2.0 - 1.0)));

    depth = mix(depth, color.a, step(0.9999, dot(Gnormal, normal_term.xyz)) * saturate(normal_term.w / 100.0 - 1.0));

    return color.rgb;
}

void main() {
    vec2 coord = (texcoord);

    #ifdef Enabled_TAA
        //coord -= jitter;
    #endif

    float depth0 = texture(depthtex0, texcoord).x;

    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, depth0) * 2.0 - 1.0));
    vec4 wP = (gbufferModelViewInverse) * nvec4(vP);
    vec3 viewDirection = normalize(vP);
    vec3 eyeDirection = -viewDirection;

    vec3 albedo = decodeGamma(texture2D(gcolor, texcoord).rgb);

    vec3 texturedNormal = normalDecode(texture2D(composite, texcoord).xy);
    vec3 geometryNormal = texture2D(gcolor, texcoord).a > 0.99 ? texturedNormal : normalDecode(texture2D(gnormal, texcoord).xy);
    vec3 visibleNormal = dot(eyeDirection, texturedNormal) < 0.2 ? geometryNormal : texturedNormal;

    float tileMaterial  = round(texture2D(gnormal, texcoord).z * 255.0);

    bool isSky      = bool(step(254.5, tileMaterial));

    vec2 offset = vec2(0.0);

    offset += jitter;
    //offset.x += frameTimeCounter * 0.01;
    #ifdef Enabled_TAA
    #endif

    float dither = GetBlueNoise(depthtex2, (texcoord) * resolution * 0.5, offset);
    float dither2 = GetBlueNoise(depthtex2, (1.0 - texcoord) * resolution * 0.5, offset);

    vec3 RSMGI = CalculateRSMGI(coord, depth0, geometryNormal, vec2(dither, dither2));
         RSMGI = encodeGamma(RSMGI);

    float rayPDF = 1.0;
    float rayDepth = depth0;

    vec3 reflection = CalculateImageBasedReflection(vP, visibleNormal, geometryNormal, vec2(dither, dither2), rayPDF, rayDepth);
         reflection = encodeGamma(reflection);

    gl_FragData[0] = vec4(RSMGI, 1.0);
    gl_FragData[1] = vec4(reflection, rayPDF);
    gl_FragData[2] = vec4(depth0, rayDepth, 0.0, 0.0);
    //gl_FragData[1] = vec4(encodeGamma(albedo), depth0);
}
/* DRAWBUFFERS:124 */
/* RENDERTARGETS: 1,2,4 */
#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 shadowModelViewInverse;

uniform vec3 cameraPosition;

uniform vec3 lightVectorView;
uniform vec3 upVectorView;
uniform vec3 lightVectorWorld;
uniform vec3 sunVectorWorld;
uniform vec3 moonVectorWorld;

uniform vec2 resolution;
uniform vec2 pixel;

uniform float frameTimeCounter;

uniform int isEyeInWater;

in vec2 texcoord;

in float fading;
in vec3 skyLightingColorRaw;
in vec3 sunLightingColorRaw;

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
    return normalize(v.xyz);
}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

#include "/libs/common.inc"

float HG(in float m, in float g){
    return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}

vec2 unpack2x4bit(in float v) {
    v *= 255.0 / 16.0;

    float ry = floor(v);
    float rx = (v) - ry;

    return vec2((16.0 / 15.0) * rx, 1.0 / 15.0 * ry);
}

#include "/libs/dither.glsl"
#include "/libs/lighting.glsl"
#include "/libs/lighting/shadowmap.glsl"

bool totalInternalReflection(in vec3 i, inout vec3 o, in vec3 n, in float eta){
    float cosi = dot(-i, n);
    float TIR = 1.0 - eta * eta * (1.0 - pow2(cosi));
    bool result = TIR < 0.0;

    o = result ? vec3(0.0) : normalize(refract(i, n, eta));

    return result;
}

const vec4 pack1 = vec4(1.0, 256.0, 65536.0, 16777216.0);
const vec4 unpack4 = vec4(1.0, 1.0/256.0, 1.0/65536.0, 1.0/16777216.0);

float pack2x8bit_1x16bit(in vec2 v){
    v = floor(v * 255.0);

    return dot(v, pack1.xy) / 65535.0;
}

vec2 unpack2x8bit_1x16bit(in float v) {
    v *= 65535.0 / 256.0;

    float ry = floor(v);
    float rx = (v) - ry;

    return vec2(256.0 / 255.0 * rx, 1.0 / 255.0 * ry);
}

bool intersectCube(in vec3 origin, in vec3 direction, in vec3 size, out float near, out float far){
    vec3 dr = 1.0 / direction;
    vec3 n = origin * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    near = max(pin.x, max(pin.y, pin.z));
    far = min(pout.x, min(pout.y, pout.z));

	// check for hit
	return near < far && far > 0.0;
}

float IntersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
    return dot(point - origin, normal) / dot(direction, normal);
}

vec2 IntersectCube(in vec3 shapeCenter, in vec3 direction, in vec3 size, inout vec3 normal){
    vec3 dr = 1.0 / direction;
    vec3 n = shapeCenter * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    vec3 front = -sign(direction) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    vec3 back = -sign(direction) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);

    normal = back;

    if(far > near && far > 0.0) {
        return vec2(near, far);
    }else{
        return vec2(-1.0);
    }
}

vec2 IntersectCube(in vec3 shapeCenter, in vec3 direction, in vec3 size) {
    vec3 dr = 1.0 / direction;
    vec3 n = shapeCenter * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    if(far > near && far > 0.0) {
        return vec2(near, far);
    }else{
        return vec2(-1.0);
    }
}

vec3 Diffusion(in float depth, in vec3 t) {
    depth = max(1e-5, depth);

    return exp(-depth * t) / (4.0 * Pi * t * max(1.0, depth));
}

void CalculateSSS(inout vec3 color, in vec3 direction, in vec2 tracing, vec3 sunlightcolor, vec3 skylightcolor,
                  in vec3 albedo, in float alpha, in float sigma_a, in float sigma_s, in float sigma_e, in float opaque) {
    int steps = 12;
    float invsteps = 1.0 / float(steps);   

    float dither = GetBlueNoise(depthtex2, (texcoord) * resolution, jitter);

    float diffthresh = 0.004 * (opaque);

    if(tracing.y - tracing.x < 0.01) return;

    float stepLength = abs(tracing.y - tracing.x) * invsteps;

    vec3 rayStart = direction * max(0.0, tracing.x);
    vec3 rayStep  = direction * (stepLength);

    vec3 absorption = albedo;
    vec3 scattering = vec3(1.0);
    vec3 extinction = vec3(1.0);

    rayStep  = mat3(shadowModelView) * rayStep;
    rayStart = mat3(shadowModelView) * rayStart + shadowModelView[3].xyz;

    vec3 trans_sunlight = vec3(0.0);
    vec3 trans_skylight = vec3(0.0);

    vec3 rayPosition = rayStart + rayStep * dither;

    for(int i = 0; i < steps; i++){
        vec3 position = rayPosition;

        vec3 projection = vec3(shadowProjection[0].x, shadowProjection[1].y, shadowProjection[2].z) * position + shadowProjection[3].xyz;

        vec3 coord = RescaleShadowMapCoord(projection);
        coord.z -= diffthresh;

        float depth0 = texture(shadowtex0, coord.xy).x;
        vec3 position1 = vec3(shadowProjectionInverse[0].x, shadowProjectionInverse[1].y, shadowProjectionInverse[2].z) * vec3(projection.xy, depth0 * 2.0 - 1.0) + shadowProjectionInverse[3].xyz;
             
        vec3 halfVector = position - position1;

        float shadowDepth = max(0.0, length(halfVector) - 0.03);

        vec3 transmittance = vec3(saturate(exp(-stepLength * sigma_e * alpha)));
        extinction *= transmittance;
        absorption *= mix(albedo, vec3(1.0), saturate(exp(-sigma_a * stepLength * alpha)));
        scattering *= vec3(saturate(exp(-sigma_s * stepLength * alpha)));

        vec3 sunlight = CalculateShading2(coord, shadowDepth);
             sunlight *= 1.0 - scattering;

        trans_sunlight += extinction * absorption * (sunlight);
        trans_skylight += extinction * absorption;

        rayPosition += rayStep;
    }

    float mu = dot(lightVectorWorld, direction);
    float phaseDual = mix(HG(mu, 0.2), HG(mu, 0.7), 0.1);

    trans_sunlight *= sunlightcolor;
    trans_sunlight *= phaseDual * stepLength * sigma_s * alpha;
    //trans_sunlight *= phaseDual * (1.0 / (sigma_s * max(0.2, alpha))) * 1.0;

    trans_skylight *= skylightcolor;
    trans_skylight *= stepLength * sigma_s * alpha * (1.0 - opaque);

    color += (trans_sunlight + trans_skylight) * (invPi * 1.0);
}

float GetDepth(in vec4 p) {
    return (p.w / p.z) * far * near / (far - near) + 0.5 * (far + near) / (far - near) + 0.5;
}

bool RayIntersect(in float rayMin, in float rayMax, float raySample, float thickness) {
    //if(rayMin > rayMax) {
    //    float t = rayMin;
    //    rayMin = rayMax;
    //    rayMax = t;
    //}

    return rayMin > raySample && rayMax < raySample + thickness;
    //return rayMax > raySample && rayMax > raySample;
}

#define Enabled_Screen_Space_Shadow           //enable contact shadow

#ifdef Enabled_Screen_Space_Shadow
float CalculateScreenSpaceShadow(in vec3 origin, in vec3 direction, vec3 normal, float dither) {
    //contact shadow
    float ndotl = dot(direction, normal);
    if(ndotl < 0.01) return 1.0; 

    int steps = 8;
    float invsteps = 1.0 / float(steps);

    float thickness = near / far * 4.0;
    float stepLength = 0.05;
    float bias = 0.1 / max(0.1, length(normalize(origin.xyz) - direction));

    float viewLength = length(origin.xyz);

    float farShadow = clamp((viewLength - 48.0) / 16.0, 0.0, 8.0);

    thickness *= 1.0 + farShadow * 0.5;
    stepLength *= 1.0 + farShadow;

    vec3 rayPosition = origin.xyz + direction * stepLength * dither + normal * bias;
    float depth = linearizeDepth(texture(depthtex0, texcoord).x);

    for(int i = 0; i < steps; i++) {
        vec3 coord = nvec3(gbufferProjection * nvec4(rayPosition)) * 0.5 + 0.5;
        if(max(abs(coord.x - 0.5), abs(coord.y - 0.5)) >= 0.5) break;

        float sampleDepth = texture(depthtex1, coord.xy).x;

        vec4 samplePosition = gbufferProjectionInverse * nvec4(vec3(coord.xy, sampleDepth) * 2.0 - 1.0);
             samplePosition.xyz /= samplePosition.w;
        //samplePosition.xyz = normalize(samplePosition.xyz) * (length(samplePosition.xyz) + sampleParallax);
        //sampleDepth = nvec3(gbufferProjection * nvec4(samplePosition.xyz)).z * 0.5 + 0.5;

        sampleDepth = linearizeDepth(sampleDepth);
        coord.z = linearizeDepth(coord.z);

        //coord.z = sqrt(rayPosition.z * rayPosition.z);
        //sampleDepth = sqrt(samplePosition.z * samplePosition.z);

        //sampleDepth = min(depth, sampleDepth);

        //if()

        //if(abs(sampleDepth - coord.z) < thickness && r1 > r2) return 0.0;

        if(sampleDepth < coord.z && sampleDepth + thickness > coord.z) return 0.0;

        //if(sampleParallax > parallaxDepth) return 0.0;

        //if(sampleDepth > coord.z - 0.1 && coord.z > sampleDepth) return 0.0;
        //if(length(samplePosition) > length(rayPosition) - 0.1 && length(rayPosition) > length(samplePosition)) return 0.0;
        //if(abs(sampleDepth - coord.z) < 0.1 && abs(sampleDepth - coord.z) > 1e-5) return sum3(texture(gcolor, coord.xy).rgb * 0.1);

        rayPosition.xyz += direction * stepLength;
        //rayPosition.xyz -= normal * invsteps * bias * 0.9;
    }

    return 1.0;
}
#endif
#define Enabled_ScreenSpace_Shadow

#ifdef Enabled_ScreenSpace_Shadow
vec3 ScreenSpaceShadow(in vec3 lightDirection, in vec3 viewPosition, in vec3 bias){
  int steps = 8;
  float isteps = 1.0 / steps;

  float trace_distance = 0.25;
  float thickness = 0.125;

  //float dither = R2sq(texcoord * resolution - jittering);
  float dither = GetBlueNoise(depthtex2, texcoord, resolution.y, jitter - frameTimeCounter * pixel * 10.0);

  //vec3 halfVector = bias;//normalize(lightDirection - normalize(viewPosition));
  //float ldoth = max(0.0, dot(lightDirection, halfVector));
  float ndotl = max(1e-5, dot(lightDirection, bias));

  vec3 start = viewPosition;
  vec3 direction = lightDirection * isteps * trace_distance;

  float viewLength = length(viewPosition);
  float distanceFade = clamp01((-viewLength + shadowDistance - 16.0) / 32.0);

  vec3 test = start + dither * direction;
       test += bias * thickness * 0.5 * clamp(1.0 / ndotl, 1.0, 10.0);

  //if(viewLength > shadowDistance - 16.0)
  if(bool(step(distanceFade, 0.99))){
    thickness *= 8.0;
    direction *= 8.0;
  }


  float depth = (texture(depthtex0, texcoord).x);
  if(bool(step(depth, 0.7))) return vec3(1.0);
  float ldepth = linearizeDepth(depth);

  //if(bool(step(depth, 0.7))){
  //  thickness *= 4.0;
  //  direction *= 0.05;
  //}

  //vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);

  //float t = 0.05;
  //float hit = 0.0001;

  float viewz = abs(viewPosition.z);

  for(int i = 0; i < steps; i++){
    vec3 coord = nvec3(gbufferProjection * vec4(test, 1.0)).xyz * 0.5 + 0.5;
    if(abs(coord.x - 0.5) > 0.5 || abs(coord.y - 0.5) > 0.5) break;

    float h = texture(depthtex0, coord.xy).x;
    //if(bool(step(h, 0.7))) continue;
    /*
    float linearZ = linearizeDepth(coord.z);
    float linearD = linearizeDepth(h);
    float dist = (linearZ - linearD);

    //if(linearZ > linearD) continue;

    //dist /= bool(step(h, 0.7)) ? linearD : 1.0;

    if(dist < 1e-5) continue;
    if(dist < thickness) return vec3(0.0);
    */

    vec3 sample_position = nvec3(gbufferProjectionInverse * nvec4(vec3(coord.xy, h) * 2.0 - 1.0));

    float test0 = length(sample_position.xyz);
    float test1 = length(test.xyz);
    float different = test1 - test0;//abs(test0 - test1) / min(test0, test1);

    //if(coord.z <= h) continue;
    if(step(different, thickness) * step(0.0, different) > 0.0) return vec3(0.0);//decodeGamma(texture2D(gcolor, coord.xy).rgb);

    test += direction;

    /*
    float linearZ = linearizeDepth(coord.z);
    float linearD = linearizeDepth(h);
    float dist = (linearZ - linearD);

    if(dist < 1e-5) continue;

    if(dist < thickness) {
      return vec3(0.0);
    }
    */
  }

  return vec3(1.0);
}
#endif
/*
vec3 SimpleSSS(in vec3 L, in vec3 direction, in vec3 albedo, in float t, in float a) {
Diffusion(0.05, vec3(1.0) / (vec3(1e-5) + pow(albedo, vec3(0.9)))) * invPi;    
}
*/

float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}

float sdBox( vec3 p, vec3 b ) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec3 LeavesShading(vec3 L, vec3 eye, vec3 n, vec3 albedo, vec3 sigma_t, vec3 sigma_s) {
    //albedo = pow(albedo, vec3(0.9));

    vec3 R = Diffusion(0.05, sigma_t);

    float mu = dot(L, -eye);
    float phase = mix(HG(mu, 0.2), HG(mu, 0.7), 0.2) * 10.0;

    return (albedo * R * sigma_s) * (invPi * phase);
}

void main() {
    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
    vec4 wP = (gbufferModelViewInverse) * nvec4(vP);
    vec3 viewDirection = normalize(vP);
    vec3 eyeDirection = -viewDirection;

    float viewLength = length(vP.xyz);

    vec3 albedo = decodeGamma(texture2D(gcolor, texcoord).rgb);
    float transMaterials = step(0.99, texture2D(gcolor, texcoord).a);
    float opaque = 1.0 - transMaterials;
    float solid = step(0.9, texture2D(gdepth, texcoord).a);

    vec4 lightingData = texture2D(gdepth, texcoord);
    vec2 lightmapPack = unpack2x8(lightingData.x);
    float torchLightmap = lightmapPack.x;
    float skyLightmap = max(0.0, lightmapPack.y * 16.0 - 1.0) / 15.0; float sky_light_level = skyLightmap * 15.0;
    vec2 unpackLightmapBlue = unpack2x8(texture(gdepth, texcoord).b); unpackLightmapBlue.x = floor(unpackLightmapBlue.x * 255.0);
    float emissive = unpackLightmapBlue.x * step(unpackLightmapBlue.x, 254.5) / 254.0 * opaque;
    float materialAO = max(unpackLightmapBlue.y, transMaterials);

    vec3 texturedNormal = normalDecode(texture2D(composite, texcoord).xy);
    vec3 geometryNormal = texture2D(gcolor, texcoord).a > 0.99 ? texturedNormal : normalDecode(texture2D(gnormal, texcoord).xy);
    vec3 visibleNormal = dot(eyeDirection, texturedNormal) < 0.2 ? geometryNormal : texturedNormal;

    vec2 specularPackge = unpack2x8(texture2D(composite, texcoord).b);

    float smoothness = specularPackge.x;
    float roughness = pow2(1.0 - specularPackge.x);
    float metallic  = specularPackge.y;
    float material  = floor(texture2D(composite, texcoord).a * 255.0);

    float metal = step(isMetallic, metallic);

    vec3 F0 = mix(vec3(max(0.02, metallic)), albedo.rgb, metal);

    float tileMaterial  = round(texture2D(gnormal, texcoord).z * 255.0);

    float water     = CalculateMaskIDVert(8.0, tileMaterial);
    float ewww      = CalculateMaskIDVert(165.0, tileMaterial);
    float leveas    = CalculateMaskIDVert(18.0, tileMaterial);
    float glass     = CalculateMaskIDVert(20.0, tileMaterial);
    float grass     = CalculateMaskIDVert(31.0, tileMaterial);
    float foliage   = min(1.0, grass + leveas);

    bool isSky      = bool(step(254.5, tileMaterial));
    bool isWater    = bool(water);
    bool isLeaves   = bool(leveas);
    bool isGrass    = bool(grass);
	bool isFoliage  = isLeaves || isGrass;
    bool isEwww     = bool(ewww);

    float dither = GetBlueNoise(depthtex2, (texcoord) * resolution, jitter);
    float dither2 = GetBlueNoise(depthtex2, (1.0 - texcoord) * resolution, jitter);

    float parallaxDepth = -(texture2D(gnormal, texcoord).a - 1.0) * 0.25 * (1.0 - transMaterials) * 1.0;
    float parallaxSelfShadow = texture(gdepth, texcoord).y;

    float alpha = transMaterials > 0.99 ? texture(gdepth, texcoord).b : 1.0;
    if(glass > 0.5 && alpha > 0.2) alpha = 1.0;

    float sigma_s = transMaterials > 0.99 ? texture(gnormal, texcoord).g * 255.0 : material;
          sigma_s = (1.0 - (sigma_s - 64.0) / 191.0) * 10.0;

    float sigma_a = texture(gnormal, texcoord).r * 255.0 * transMaterials * 0.0;

    float sigma_e = sigma_s + sigma_a * 0.01;

    const float airIOR = 1.000293;
    const float waterIOR = 1.333;
    float F0toIOR = 1.0 / ((2.0 / (sqrt(metallic) + 1.0)) - 1.0);

    float IOR0 = isEyeInWater == 1 ? waterIOR : airIOR;
    float IOR1 = isEyeInWater == 1 && isWater ? airIOR : F0toIOR;

    float eta0 = IOR0 / IOR1;
    float eta1 = IOR1 / IOR0;

    vec2 refracted = texcoord;

    vec3 backNormal = normalDecode(texture2D(gaux1, texcoord).yz); 

    float depth1 = texture(depthtex1, texcoord).x;
    float depth3 = texture(gaux1, texcoord).x;

    vec3 back_viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth3) * 2.0 - 1.0));
    vec3 solid_viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth1) * 2.0 - 1.0));

    //float backLength = isWater ? length(solid_viewPosition) : length(back_viewPosition);

    float opticalDepth = length(back_viewPosition - vP);

    if(transMaterials > 0.5) {
        //color = vec3(1.0, 0.0, 0.0);
        //versus tales

        //color = texture2D(gaux1, texcoord).rgb;

        //color = mat3(gbufferModelViewInverse) * normalDecode(texture2D(gaux1, texcoord).yz);
        //color = saturate(color);
        //float contactDepth = length();

        //color = saturate(opticalDepth) * vec3(1.0);
        //color = mat3(gbufferModelViewInverse) * normalDecode(texture2D(gaux1, texcoord).yz);
        //color = saturate((color));

        //if()

        bool hitsolid = abs(depth1 - depth3) < 0.001;

        vec3 direction0 = vec3(0.0);
        bool TIR0 = totalInternalReflection(viewDirection, direction0, texturedNormal, eta0);
             direction0 *= isWater ? min(1.414, opticalDepth) : opticalDepth;

        vec3 rayDirection = vP + direction0;

        //vec3 direction1 = vec3(0.0);
        //bool TIR1 = totalInternalReflection(normalize(rayDirection), direction1, backNormal, eta1);

        //if(!hitsolid)
        //rayDirection += normalize(reflect(normalize(rayDirection), backNormal));

        //if(F0toIOR > 1.0)
        //refracted = nvec3(gbufferProjection * nvec4(rayDirection)).xy * 0.5 + 0.5;
        //if(TIR0) refracted = vec2(-1.0);

        //if(max(abs(refracted.x - 0.5), abs(refracted.y - 0.5)) > 0.5) refracted = texcoord;
    }

    vec2 projected = texcoord;

    #ifdef Enabled_TAA
    projected -= jitter;
    #endif

    vec3 color = decodeGamma(texture2D(gaux2, refracted).rgb);

    if(isSky){
        gl_FragData[1] = vec4(encodeGamma(color), 0.0);
        return;
    }

    float m_sigma_sca = opaque > 0.9 ? material : texture(gnormal, texcoord).g * 255.0;
          m_sigma_sca = (1.0 - (m_sigma_sca - 64.0) / 191.0) * 10.0;

    float m_sigma_abs = opaque > 0.9 ? 0.0 : texture(gnormal, texcoord).r * 255.0;

    vec3 material_absorption = vec3(m_sigma_abs) / pow(max(albedo, vec3(0.01)), vec3(1.0 / 2.718));
    vec3 material_scattering = vec3(m_sigma_sca);
    vec3 material_transmittance = material_absorption + material_scattering;

    color *= transMaterials;
    color *= max(refracted.x, refracted.y) == -1.0 ? vec3(0.0) : vec3(1.0);

    color *= 1.0 - step(0.99, alpha);

    float shadowLightFading = saturate(dot(lightVectorView, geometryNormal) * 10.0 - 0.5);

    vec3 shading = CalculateSunShading(projected, geometryNormal, foliage * 0.001, material, parallaxDepth);

    //vec3 sunLight = BRDFLighting(albedo * mix(vec3(0.03 * sigma_s * alpha), vec3(1.0), vec3(step(0.99, alpha))), 1.0, lightVectorView, eyeDirection, visibleNormal, texturedNormal, F0, roughness, metallic, material) * shadowLightFading;
    vec3 sunLight = DiffuseLighting(vec4(albedo, alpha > 0.99 ? 1.0 : sigma_s * 0.03 * alpha), lightVectorView, eyeDirection, texturedNormal, texturedNormal, F0, roughness, metallic, material);// * mix(min(1.0, sigma_s * alpha * 0.03), 1.0, max(alpha * glass, opaque));
         sunLight += SpecularLighting(vec4(albedo, 1.0), lightVectorView, eyeDirection, texturedNormal, texturedNormal, F0, roughness, metallic, material, false);
         sunLight *= shadowLightFading;
         sunLight += LeavesShading(lightVectorView, eyeDirection, texturedNormal, albedo, material_transmittance, material_scattering) * foliage;
         sunLight *= shading * sunLightingColorRaw;
         sunLight *= fading;

    #ifdef Enabled_Screen_Space_Shadow
        sunLight *= foliage > 0.9 ? 1.0 : CalculateScreenSpaceShadow(vP, lightVectorView, geometryNormal, dither);
    #endif

        sunLight *= mix(albedo.rgb / 21.0 * 10.0 * invPi, vec3(1.0), vec3(parallaxSelfShadow));
        //sunLight *= parallaxSelfShadow;

    vec3 rayDirection = normalize(reflect(viewDirection, visibleNormal));
    vec3 halfVector = normalize(rayDirection + eyeDirection);

    vec3 kS = SchlickFresnel(F0, max(0.0, dot(halfVector, eyeDirection)));
    vec3 kD = 1.0 - kS;

    float ambientOcclusion = 1.0;

    float remap_skylightMap0 = skyLightmap * (1.0 / pow(max(15.0 - sky_light_level, 1.0), 0.5)) * pow5(mix(saturate(dot(geometryNormal, texturedNormal)), 1.0, 0.1)) * ambientOcclusion;// * 0.176775;
    float remap_skylightMap1 = skyLightmap * saturate(pow(skyLightmap, 20.0)) * max(dot(upVectorView, texturedNormal), 0.0);// * 0.176775;
    float remap_skylightMap2 = skyLightmap * pow5(skyLightmap) * ambientOcclusion;

    vec3 ambientLight = (albedo.rgb * skyLightingColorRaw) * (invPi);
         //ambientLight *= (vec3(remap_skylightMap0) + vec3(remap_skylightMap1) + vec3(remap_skylightMap2) * kD) / (3.0);// + remap_skylightMap1 + remap_skylightMap2 * ambientOcclusion
         ambientLight *= vec3(remap_skylightMap0) + vec3(remap_skylightMap1) + vec3(remap_skylightMap2) * kD;   //>= 3.0
         ambientLight *= (1.0 - metal) * (1.0 - metallic) * 4.0;
         ambientLight *= 1.0 / pow2(1.0 + abs(parallaxDepth) * 1.0);

    vec3 torchLight = albedo.rgb * pow(vec3(1.022, 0.782, 0.344), vec3(2.0));
         torchLight *= max(0.0, torchLightmap * 15.0 - 1.0) * (1.0 / pow2(16.0 - torchLightmap * 15.0)) * ambientOcclusion;
         torchLight *= invPi * 0.25 * (1.0 - metallic) * (1.0 - metal);
         torchLight *= 1.0 / pow2(1.0 + abs(parallaxDepth) * 1.0);

    /**/

    vec3 direction = normalize(wP.xyz);

    vec3 worldNormal = mat3(gbufferModelViewInverse) * geometryNormal;

    vec3 blockShape = vec3(0.5);

    vec3 cubeCenter = floor(wP.xyz + cameraPosition - worldNormal * 0.1) + 0.5;
    vec2 tracingCube = IntersectCube(cameraPosition - cubeCenter, direction, blockShape);

    if(transMaterials > 0.5){
        //color = vec3(1.0, 0.0, 0.0);
        
        vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(depthtex0, refracted).x) * 2.0 - 1.0));

        vec3 back_viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(gaux1, refracted).x) * 2.0 - 1.0));
        vec3 solid_viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(refracted, texture(depthtex1, refracted).x) * 2.0 - 1.0));

        float opticalDepth2 = texture2D(gcolor, refracted).a < 0.99 ? opticalDepth : length(back_viewPosition - viewPosition);

        //if(isEwww) opticalDepth2 = max(tracingCube.y, opticalDepth2);

        vec3 absorption = vec3(exp(-sigma_a * alpha * opticalDepth2));
        vec3 extinction = vec3(exp(-sigma_e * alpha * opticalDepth2));

        //color = mix(albedo, vec3(1.0), exp(-sigma_a * alpha * 10.0)) * color;

        //color = mix(color * albedo, color, absorption) * extinction;
        //color *= extinction;
        
        //color *= mix(vec3(1.0), albedo, alpha) * mix(albedo, vec3(1.0), absorption) * extinction;
        color *= exp(-material_transmittance * opticalDepth2 * alpha);

        //vec3 trans_scattering = vec3((1.0 - (material - 64.0) / 191.0) * 1.0 * alpha);
        //color = mix(albedo * invPi * sunLightingColorRaw * 0.1, color, exp(-sigma_s * alpha * opticalDepth));
        

        //color = vec3(alpha);
    }

    //float viewLength2 = 

    /*
    float farPoint = -1.0;
    float nearPoint = -1.0;

    //vec3 cubePosition = floor(wP.xyz + cameraPosition - worldNormal * 0.999) - 0.5;
    bool traceing = intersectCube(cameraPosition - cubePosition, direction, blockShape, nearPoint, farPoint);

    int steps = 12;
    float invsteps = 1.0 / float(steps); 

    vec3 viewBorder = vec3( abs(IntersectPlane(vec3(0.0), direction, vec3(far + 16.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0))),
                            0.0,
                            abs(IntersectPlane(vec3(0.0), direction, vec3(0.0, 0.0, far + 16.0), vec3(0.0, 0.0, 1.0))));

    float start = viewLength;
    float end = min(min(viewBorder.x, viewBorder.z), transMaterials > 0.5 ? length(back_viewPosition) : farPoint);//transMaterials > 0.5 ? length(back_viewPosition) : farPoint;
    float stepLength = abs(end - start) * invsteps;
    bool hit = end > 0.0;
    */


    if(material > 64.5 && !isFoliage){
        float mu = dot(direction, lightVectorWorld);
        float phase = mix(HG(mu, -0.1), HG(mu, 0.7), 0.1) * 10.0;

        float start = opaque > 0.9 ? tracingCube.x : viewLength;
        float end = opaque > 0.9 ? tracingCube.y : length(back_viewPosition);

        vec3 viewBorder = vec3( abs(IntersectPlane(vec3(0.0), direction, vec3(far + 16.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0))),
                                0.0,
                                abs(IntersectPlane(vec3(0.0), direction, vec3(0.0, 0.0, far + 16.0), vec3(0.0, 0.0, 1.0))));
        end = min(min(viewBorder.x, viewBorder.z), end);

        if(start > 0.0 || end > start) {
            vec3 scattering = vec3(0.0);
            vec3 transmittance = vec3(1.0);

            int steps = 16;
            float invsteps = 1.0 / float(steps);

            float stepLength = (end - start) * invsteps;

            float current = start + stepLength * dither;

            bool non_solid = min(solid, opaque) < 0.9;

            for(int i = 0; i < steps; i++) {
                vec3 rayPosition = direction * current;
                if(length(rayPosition) > end) break;

                float tracingLight = max(0.0, IntersectCube(rayPosition + cameraPosition - cubeCenter, lightVectorWorld, blockShape).y);
                float r = tracingLight;

                vec3 shadowCoordOrigin = ProjectionToShadowMap((!non_solid ? rayPosition + tracingLight * lightVectorWorld : rayPosition));
                vec3 shadowCoord = shadowCoordOrigin;
                     shadowCoord.xy *= ShadowCoordDistortion(shadowCoord.xy);
                     shadowCoord = shadowCoord * 0.5 + 0.5;
                     CalculateShadowCoordRescale(shadowCoord);

                float d0 = texture(shadowtex0, shadowCoord.xy).x;
                float d1 = texture(shadowtex1, shadowCoord.xy).x;

                vec3 tr = exp(-material_transmittance * stepLength);

                if(non_solid) {
                    float d = 100.0;
                        
                    //for(float ii = -1.0; ii <= 1.0; ii += 1.0) {
                    //    for(float ij = -1.0; ij <= 1.0; ij += 1.0) {
                    //        d = min(d, texture(shadowtex0, ShadowCoordResclae((shadowCoord.xy + vec2(ii, ij) * shadowPixel) * vec2(1.0, 2.0))).x);
                    //    }
                    //}

                    vec3 p0 = mat3(shadowProjectionInverse) * (vec3(shadowCoordOrigin.xy, d0 * 2.0 - 1.0));
                    vec3 p1 = mat3(shadowProjectionInverse) * (vec3(shadowCoordOrigin.xyz));

                    r = length(p1 - p0);
                }

                vec3 luminance = Diffusion(r, material_transmittance);

                float visibility = step(shadowCoord.z, d1 + shadowPixel);

                luminance *= visibility;

                scattering += (luminance - luminance * tr) * transmittance;
                transmittance *= tr;

                current += stepLength;
            }

            color += (scattering * sunLightingColorRaw * albedo * material_scattering) * (invPi * phase);
        }

        //vec3 sss = vec3(0.0);
        //CalculateSSS(sss, direction, vec2(start, end), sunLightingColorRaw, skyLightingColorRaw * (remap_skylightMap0 + remap_skylightMap1 + remap_skylightMap2) / 15.0, albedo, alpha, sigma_a, sigma_s, sigma_e, 1.0 - transMaterials);
        //sss *= kD;
        //color += sss;

        //color = vec3(1.0, 0.0, 0.0);
    }
    
    color += sunLight;
    color += torchLight * mix(vec3(0.03 * alpha * sigma_s), vec3(1.0), vec3(step(0.99, alpha)));
    color += ambientLight * max(step(0.99, alpha), 1.0 - transMaterials);
    color += albedo * emissive * pow(vec3(1.022, 0.782, 0.344), vec3(2.0));

    //if(material > 64.5){
        //vec3 color_absorption = exp(-blockDepth * material_absorption);

    //}


    //color = exp(-(1.0 / (1e-5 + pow(albedo, vec3(1.0 / 2.718)))));
    //color = albedo;

    //color = vec3(0.001 * sdBox(wP.xyz, vec3(10.0)));
    //color = decodeGamma(texture2D(shadowcolor0, texcoord).rgb);

    //color = (albedo) * vec3(parallaxSelfShadow * 0.99 + 0.01);

    //vec3 vPj = nvec3(gbufferProjectionInverse * nvec4(vec3(projected, texture(depthtex0, texcoord).x) * 2.0 - 1.0));

    //color = vec3((vPj, lightVectorView, geometryNormal, dither)) * albedo;

    //color = vec3();

    //color = step(0.18, length(viewDirection * (viewLength + parallaxDepth * 1.0))) * vec3(1.0);
    //color = vec3(ScreenSpaceShadow(lightVectorView, vP, geometryNormal));

    //color = decodeGamma(texture2D(shadowcolor0, texcoord).rgb);

    //color = vec3(1.0) * skyLightmap * (1.0 / pow(max(15.0 - sky_light_level, 1.0), 0.5));

    //color = vec3(1.0 / pow2(1.0 + abs(parallaxDepth) * 128.0));
    //color = vec3(abs(parallaxDepth));

    //if(emissive > 0.) color = vec3(1.0, 0.0, 0.0);

    //color = albedo * 0.1;

    //color = max(dFdx(wP.xyz), dFdy(wP.xyz)) * vec3(1.0);
    //color = max(abs(dFdx(wP.xyz)), abs(dFdy(wP.xyz)));

    //if(material < 64.5 && material > 0.5) color = vec3(1.0, 0.0, 0.0);
    //color = vec3(material / 255.0);

    //if(viewBorder.x > 0.0 && viewBorder.z > 0.0)
    //color = vec3(saturate(min(viewBorder.x, viewBorder.z) / 1000.0));

        //color = length(wP.xyz - parallaxDepth * 10.0) * vec3(0.01);
    //color = -vec3(texture2D(gnormal, texcoord).a - 1.0);
        //color = vec3(-parallaxDepth);

    //color = vec3(material / 255.0) * step(64.5, material);
    

    //color = length(wP.xyz - direction * farPoint) * vec3(0.01);
/*
    int steps = 12;
    float invsteps = 1.0 / float(steps);

    float start = 0.0;
    float end = 100.0;

    float stepLength = (end - start) * invsteps;

    vec3 direction = normalize(wP.xyz);
    
    vec3 rayStep = mat3(shadowModelView) * direction;
         rayStep *= stepLength;

    vec3 rayPosition = rayStep * dither;

    float visibility = 0.0;

    for(int i = 0; i < steps; i++) {
        float rayLength = length(rayPosition);
        if(rayLength > viewLength || rayLength < 0.05) break;

        vec3 position = rayPosition + shadowModelView[3].xyz;

        position = vec3(shadowProjection[0].x, shadowProjection[1].y, shadowProjection[2].z) * position + shadowProjection[3].xyz;
        position.xy *= 1.0 / (mix(1.0, length(position.xy), SHADOW_MAP_BIAS) / SHADOW_MAP_BIAS_Mul);
        position = position * 0.5 + 0.5;
        position.z -= shadowPixel;

        vec2 coord = clamp(position.xy, vec2(0.0), vec2(1.0)) * vec2(0.8125);

        visibility += step(position.z, texture(shadowtex1, coord).x);

        rayPosition += rayStep;
        //rayLength += stepLength;
    }

    color = visibility * vec3(1.0);
*/
    //color = decodeGamma(texture2D(shadowcolor0, coord).rgb);

    /**/

    color = encodeGamma(color);

/*
    vec3 n = mat3(gbufferModelViewInverse) * texturedNormal;

    vec2 oct_n = float32x3_to_oct(n);
    float f_n = pack2x8bit_1x16bit(oct_n * 0.5 + 0.5);

    f_n = floor(f_n * 65535.0) / 65535.0;

    color = oct_to_float32x3(unpack2x8bit_1x16bit(f_n) * 2.0 - 1.0);

    color = vec3(dot(color, n));
    if(color.r < 0.999) color = vec3(1.0, 0.0, 0.0);

    color = saturate(color);
*/
/*
    albedo = encodeGamma(albedo);
    albedo = floor(albedo * 255.0) / 255.0;

    float f = pack2x8bit_1x16bit(albedo.rg);

    color = vec3(unpack2x8bit_1x16bit(f), albedo.b);

    if(color == albedo) color = vec3(1.0, 0.0, 0.0);
*/
    //color *= 1000.0;

    //vec3 n = mat3(gbufferModelViewInverse) * texturedNormal;

    //n = vec3(float32x3_to_oct(n) * 0.5 + 0.5, 0.0);

    //float F16 = dot(n.xy * vec2(255.0), vec2(1.0, 256.0));
          //F16 = floor(F16 * 65535.0) / 65535.0;

    //n = oct_to_float32x3((n.xy * 256.0) * 2.0 - 1.0);

    //color = saturate(n);

    gl_FragData[0] = vec4(normalEncode(visibleNormal), pack2x8(vec2(smoothness, metallic)), 1.0);
    gl_FragData[1] = vec4(color, 1.0);
}
/* DRAWBUFFERS:35 */
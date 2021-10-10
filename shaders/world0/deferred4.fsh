#version 130

#define Moon_Luminance 10.0             //[1.0 5.0 7.5 10.0 15.0 20.0 100.0]
#define Moon_Radius 1.0                 //[0.5 0.75 1.0 1.5 2.0]
#define Moon_Distance 1.0               //[0.5 0.75 1.0 1.5 2.0]

const float moon_radius = 1734000.0;
const float moon_distance = 38440000.0;

#define Stars_Visible 0.005         //[0.00062 0.00125 0.0025 0.005 0.01 0.02 0.04]
#define Stars_Luminance 0.1         //[0.025 0.05 0.1 0.2 0.4]
#define Stars_Speed 1.0             //[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
#define Planet_Angle 0.1            //[-0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5] +:north -:south
#define Polaris_Size 2.0            //[1.0 2.0 3.0 4.0]
#define Polaris_Luminance 1.0       //[1.0]
#define Polaris_Offset_X 4.0        //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]
#define Polaris_Offset_Y 4.0        //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;

uniform sampler2D depthtex0;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

uniform vec3 cameraPosition;
uniform vec3 moonPosition;
uniform vec3 shadowLightPosition;

uniform vec3 sunVectorWorld;
uniform vec3 moonVectorWorld;
uniform vec3 lightVectorWorld;
uniform vec3 lightVectorView;
uniform vec3 upVectorView;

uniform int worldTime;

uniform float frameTimeCounter;

uniform float aspectRatio;
uniform vec2 resolution;
uniform vec2 pixel;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

in vec2 texcoord;

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

#include "/libs/common.inc"
#include "/libs/dither.glsl"
#include "/libs/lighting.glsl"
#include "/libs/volumetric/atmospheric_common.glsl"
#include "/libs/material/sss_low.glsl"

#define Soft_Shadow_Quality Medium              //[OFF Low Medium High]
#define Weather_Shadow_Quality                  //

#define SHADOW_MAP_BIAS 0.9
#define SHADOW_MAP_BIAS_Mul 0.95

const int   shadowMapResolution  = 2048;
const float shadowDistance       = 128.0;

const bool  generateShadowMipmap = false;

const bool  shadowHardwareFiltering = false;

const bool  shadowColor0Nearest = true;
const bool  shadowcolor1Nearest = true;

float shadowPixel = 1.0 / float(shadowMapResolution);

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

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

void CalculateShadowCoordResclae(inout vec3 coord){
    //coord.xy = min(vec2(1.0), coord.xy) * 0.8125;
    coord.xy = ShadowCoordResclae(coord.xy);
}

vec3 ProjectionToShadowMap(in vec3 position){
    vec3 projection = mat3(shadowModelView) * position + shadowModelView[3].xyz;
         projection = vec3(shadowProjection[0].x, shadowProjection[1].y, shadowProjection[2].z) * projection + shadowProjection[3].xyz;

    return projection;
}

vec3 wP2sP(in vec4 wP, out float distortion){
	vec4 position = shadowProjection * shadowModelView * wP;
         position /= position.w;

    distortion = 1.0 / (mix(1.0, length(position.xy), SHADOW_MAP_BIAS) / SHADOW_MAP_BIAS_Mul);
    position.xy *= distortion;

    position = position * 0.5 + 0.5;
    position.xy = min(position.xy, vec2(1.0)) * 0.8125;

	return position.xyz;
}

vec3 wP2sP(in vec4 wP){
	vec4 position = shadowProjection * shadowModelView * wP;
         position /= position.w;

    position.xy /= (mix(1.0, length(position.xy), SHADOW_MAP_BIAS) / SHADOW_MAP_BIAS_Mul);

    position = position * 0.5 + 0.5;
    position.xy = min(position.xy, vec2(1.0)) * 0.8125;

	return position.xyz;
}

vec2 unpack2x4bit(in float v) {
    v *= 255.0 / 16.0;

    float ry = floor(v);
    float rx = (v) - ry;

    return vec2(16.0 / 15.0 * rx, 1.0 / 15.0 * ry);
}

vec3 CalculateShading(in vec3 coord, float shadowDepth){
    vec3 shading = vec3(0.0);

    float alpha = texture2D(shadowcolor0, coord.xy).a;
    vec3 tint = decodeGamma(texture2D(shadowcolor0, coord.xy));

    vec2 material2 = unpack2x4bit(texture2D(shadowcolor1, coord.xy).a) * vec2(15.0, 255.0);
    float sigma_a = material2.r; 
    float sigma_s = material2.g; sigma_s = (1.0 - (sigma_s - 64.0) / 191.0) * 16.0;
    float sigma_e = sigma_s + sigma_a * 0.01;

    vec3 absorption = vec3(saturate(exp(-sigma_a * alpha * shadowDepth)));
    vec3 extinction = vec3(saturate(exp(-sigma_e * alpha * shadowDepth)));

    float depth0 = texture(shadowtex0, coord.xy).x;
    float depth1 = texture(shadowtex1, coord.xy).x;

    float visibility0 = step(coord.z, depth0);
    float visibility1 = step(coord.z, depth1);
    float hlaf_visibility = saturate(visibility1 - visibility0);

    //shading = mix(vec3(1.0), extinction * mix(tint, vec3(1.0), absorption), vec3(max(step(1.0, sigma_s), hlaf_visibility))) * vec3(visibility1);

    shading = mix(vec3(1.0), extinction * mix(tint, vec3(1.0), absorption), vec3(hlaf_visibility)) * vec3(mix(visibility1, visibility0, step(0.99, alpha)));

    return shading;
}

vec3 CalculateShading(in vec3 coord, vec3 halfvector){
    return CalculateShading(coord, length(halfvector));
}

vec3 CalculateSunShading(in vec2 texel, in vec3 normal, in float scale){
    vec4 worldPosition = gbufferProjectionInverse * nvec4(vec3(texel, texture(depthtex0, texel).x) * 2.0 - 1.0);
         worldPosition = gbufferModelViewInverse * (worldPosition / worldPosition.w);

    //worldPosition.xyz += mat3(gbufferModelViewInverse) * normal / (1e-4 + max(0.0, dot(normal, lightVectorView))) * 0.001;

    //float distortion = 1.0;
    //vec3 shadowCoord = wP2sP(worldPosition, distortion);
    float distortion = 1.0;

    vec3 projection = ProjectionToShadowMap(worldPosition.xyz);

    vec3 shadowCoord = projection;
    CalculateShadowCoordDistortion(shadowCoord, distortion);
    shadowCoord = shadowCoord * 0.5 + 0.5;
    CalculateShadowCoordResclae(shadowCoord);

    float visibility = dot(normal, lightVectorView);

    float diffthresh = (4.0 / distortion / (0.1 + abs(visibility) * 0.9));// / (1e-2 + pow2(abs(visibility))) * 0.4 + distortion * 0.1;
          diffthresh *= shadowPixel;
          diffthresh += scale;

    vec2 offset = jitter;
    float dither = GetBlueNoise(depthtex2, (texcoord) * resolution, offset);
    float dither2 = GetBlueNoise(depthtex2, (1.0 - texcoord) * resolution, offset);
    //float dither = GetBlueNoise(depthtex2, texcoord, resolution.y, jitter);
    float rotate_angle = dither * 2.0 * Pi;
    mat2 rotate = mat2(cos(rotate_angle), sin(rotate_angle), -sin(rotate_angle), cos(rotate_angle));
    //mat2 rotate = mat2(1.0, 0.0, 0.0, 1.0);

    float radius = 2.0;
    float max_radius = sqrt(radius * radius + radius * radius);

    vec3 shading = vec3(0.0);

    float texelSize = shadowPixel * distortion;
    float penumbra = 1.0;

    float receiver = shadowCoord.z - shadowPixel;
    shadowCoord.z -= diffthresh;

    float blocker0 = 0.0;
    float blocker1 = 0.0;

    int count0 = 0;
    int count1 = 0;

    float weight = 0.0;
    
    #if 1

    float depth0 = texture(shadowtex0, shadowCoord.xy).x;
    float depth1 = texture(shadowtex1, shadowCoord.xy).x;
    //float visibility0 = step(shadowCoord.z, depth0);

    //shading = vec3(visibility0);

    vec3 position0 = vec3(shadowProjectionInverse[0].x, shadowProjectionInverse[1].y, shadowProjectionInverse[2].z) * (projection) + shadowProjectionInverse[3].xyz;
    vec3 position1 = vec3(shadowProjectionInverse[0].x, shadowProjectionInverse[1].y, shadowProjectionInverse[2].z) * vec3(projection.xy, depth0 * 2.0 - 1.0) + shadowProjectionInverse[3].xyz;

    shading = CalculateShading(shadowCoord, position1 - position0);

    /*
    float depth1 = texture(shadowtex1, shadowCoord.xy).x;
    float visibility1 = step(shadowCoord.z, depth1);

    float sigma_e = texture2D(shadowcolor0, shadowCoord.xy).a;
    float sigma_a = texture2D(shadowcolor1, shadowCoord.xy).a;
    vec3 stained = decodeGamma(texture2D(shadowcolor0, shadowCoord.xy).rgb);

    vec3 p0 = mat3(shadowProjectionInverse) * (vec3(shadowCoord.xy, depth0) * 2.0 - 1.0);
    vec3 p1 = mat3(shadowProjectionInverse) * (vec3(shadowCoord.xy, depth1) * 2.0 - 1.0);

    float shadowDepth = length(p1.xyz - p0.xyz);


    if(sigma_e < 0.99){
        //shading *= saturate(exp(-shadowDepth * sigma_e));
    }

    if(sigma_a > 0.01){
        //shading *= mix(stained, vec3(1.0), saturate(exp(-shadowDepth * sigma_a)) * (visibility1 - visibility0));
        shading *= mix(stained, vec3(1.0), saturate(exp(-shadowDepth * sigma_a * 16.0)));
    }
    */
    #else
    for(float i = -radius; i <= radius; i += 1.0) {
        for(float j = -radius; j <= radius; j += 1.0) {

            #if Soft_Shadow_Quality == Medium
            if(length(vec2(i, j)) > max_radius * 0.78) continue;
            #endif

            vec2 offset = vec2(i - dither2 + 0.5, j + dither2 - 0.5) * rotate * (penumbra);

            vec2 coord = shadowCoord.xy + offset * texelSize;

            float depth0 = texture(shadowtex1, coord).x;

            if(receiver > depth0) {
                blocker0 += depth0;
                count0++;
            }
        }
    }

    blocker0 /= max(1.0, float(count0));

    penumbra *= abs(receiver - blocker0) / blocker0;

    for(float i = -radius; i <= radius; i += 1.0) {
        for(float j = -radius; j <= radius; j += 1.0) {
            vec2 offset = vec2(i, j);

            #if Soft_Shadow_Quality == Medium
            if(length(vec2(i, j)) > max_radius * 0.78) continue;
            #endif

            vec2 position = vec2(offset.x - dither2 + 0.5, offset.y + dither2 - 0.5) * rotate * (penumbra);

            vec2 coord = shadowCoord.xy + position * texelSize;

            float view = shadowCoord.z;// - shadowPixel * length(offset) / distortion * 32.0 * penumbra;
            float depth0 = texture(shadowtex1, coord).x;

            shading += step(view, depth0);
            weight += 1.0;
        }
    }

    shading /= weight;
    #endif
/*
    for(float i = -2.0; i <= 2.0; i += 1.0) {
        for(float j = -2.0; j <= 2.0; j += 1.0) {
            vec2 offset = vec2(i, j) * penumbra;
            vec2 coord = shadowCoord.xy + offset * radius;

            shading += vec3(step(shadowCoord.z, texture(shadowtex0, coord).x + shadowPixel / distortion * 64.0 * length(offset)));
        }
    }
*/

    return shading;
}

#define Enabled_TAA

const float moon_in_one_tile = 9.0;

uniform int moonPhase;

uniform sampler2D depthtex1;

vec3 DrawMoon(in vec3 L, vec3 direction, float hit_planet) {
    vec2 traceingMoon = RaySphereIntersection(vec3(0.0) - L * (moon_distance * Moon_Distance + moon_radius * Moon_Radius), direction, vec3(0.0), moon_radius * Moon_Radius);
    vec2 traceingMoon2 = RaySphereIntersection(vec3(0.0) - L * (moon_distance * Moon_Distance + moon_radius * Moon_Radius), L, vec3(0.0), moon_radius * Moon_Radius);

    mat3 lightModelView = mat3(shadowModelView[0].xy, L.x,
                               shadowModelView[1].xy, L.y,
                               shadowModelView[2].xy, L.z);

    vec3 coord3 = lightModelView * direction; 
    vec2 coord2 = coord3.xy / coord3.z;
         coord2 *= max(0.0, traceingMoon2.x) / (moon_radius * Moon_Radius) * inversesqrt(moon_in_one_tile); 
         coord2 = coord2 * 0.5 + 0.5;

    float moon = float(moonPhase);
    vec2 chosePhase = vec2(mod(moon, 4), step(3.5, moon));

    vec2 coord = (coord2 + chosePhase) * vec2(0.25, 0.5);

    vec4 moon_texture = texture2D(depthtex1, coord + chosePhase); moon_texture.rgb = decodeGamma(moon_texture.rgb);
    float hit_moon = float(abs(coord2.x - 0.5) < 0.5 && abs(coord2.y - 0.5) < 0.5 && coord3.z > 0.0) * step(hit_planet, 0.0);

    return moon_texture.rgb * (hit_moon * MoonLight * moon_texture.a * Moon_Luminance);    
}

vec3 DrawStars(in vec3 direction, float hit_planet) {
    vec2 coord = vec2(0.0);

    float angle = Planet_Angle * 2.0 * Pi;
    mat2 rotate = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    direction.yz *= rotate;

    float time_angle = frameTimeCounter / (1200.0) * Stars_Speed * 2.0 * Pi;
    mat2 time_rotate = mat2(cos(time_angle), sin(time_angle), -sin(time_angle), cos(time_angle));

    direction.xz *= time_rotate;

    vec3 n = abs(direction);
    vec3 coord3 = n.x > max(n.y, n.z) ? direction.yzx :
                  n.y > max(n.x, n.z) ? direction.zxy : 
                  direction;

    float stars = rescale((1.0 - Stars_Visible), 1.0, hash(floor(coord3.xy / coord3.z * 256.0)));
          stars += float(floor(coord3.xy / coord3.z * 256.0 / Polaris_Size - vec2(Polaris_Offset_X, Polaris_Offset_Y)) == vec2(0.0)) * Polaris_Luminance * float(n.y > max(n.x, n.z) && coord3.y > 0.0);
          stars *= step(hit_planet, 0.0);

    return vec3(stars);
}

void main() {
    vec2 jitter_coord = texcoord;

    #ifdef Enabled_TAA
    jitter_coord -= jitter;
    #endif

    vec3 color = vec3(0.0);

    vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, texture(depthtex0, texcoord).x) * 2.0 - 1.0));
    vec4 wP = (gbufferModelViewInverse) * nvec4(vP);
    vec3 viewDirection = normalize(vP);
    vec3 eyeDirection = -viewDirection;

    vec3 albedo = decodeGamma(texture2D(gcolor, texcoord).rgb);

    vec4 lightingData = texture2D(gdepth, texcoord);
    vec2 lightmapPack = unpack2x8(lightingData.x);
    float torchLightmap = lightmapPack.x;
    float skyLightmap = lightmapPack.y;

    float material  = round(texture2D(gnormal, texcoord).z * 255.0);
    float leveas    = CalculateMaskIDVert(18.0, material);
    float grass     = CalculateMaskIDVert(31.0, material);
    float foliage   = min(1.0, grass + leveas);

    bool isSky      = bool(step(254.5, material));
    bool isLeaves   = bool(leveas);
    bool isGrass    = bool(grass);
	bool isFoliage  = isLeaves || isGrass;

    vec3 flatNormal = normalDecode(texture2D(gnormal, texcoord).xy);
    vec3 texturedNormal = normalDecode(texture2D(composite, texcoord).xy);
    vec3 visibleNormal = dot(eyeDirection, texturedNormal) < 0.2 ? flatNormal : texturedNormal;

    vec2 specularPackge = unpack2x8(texture2D(composite, texcoord).b);

    float roughness = pow2(1.0 - specularPackge.x);
    float metallic = specularPackge.y;
    float materials = floor(texture2D(composite, texcoord).a * 255.0);

    vec3 F0 = mix(vec3(max(0.02, metallic)), albedo.rgb, step(isMetallic, metallic));

    vec3 shading = CalculateSunShading(jitter_coord, flatNormal, 0.0);

    vec3 sunLight = BRDFLighting(albedo, 1.0, lightVectorView, eyeDirection, visibleNormal, texturedNormal, F0, roughness, metallic, materials);
         sunLight *= saturate(dot(lightVectorView, flatNormal) * 10.0 - 0.5);
         //sunLight += LeavesShading(lightVectorView, eyeDirection, texturedNormal, albedo, materials) * foliage;
         sunLight *= sunLightingColorRaw * shading;
         sunLight *= fading;

    vec3 ambientLight = albedo.rgb * skyLightingColorRaw;
         ambientLight *= 8.0 * invPi * max(skyLightmap * 15.0 - 1.0, 0.0) / 14.0;

    color = sunLight + ambientLight;
    //if(fading == 1.0) color = vec3(1.0, 0.0, 0.0);
    //color = CalculateSkyLightColor(E, vec3(0.0, 1.0, 0.0), lightVectorWorld) * sunLightingColorRaw;

    vec3 jitter_viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(jitter_coord, texture(depthtex0, jitter_coord).x) * 2.0 - 1.0));
    vec3 jitter_worldPosition = mat3(gbufferModelViewInverse) * (jitter_viewPosition);

    if(isSky) {
        vec3 transmittance = texture2D(gaux1, min(vec2(0.5) - pixel, texcoord * 0.5)).rgb;

        vec3 skyColor = decodeGamma(texture2D(gaux2, min(vec2(0.5) - pixel, texcoord * 0.5)).rgb);
        color = vec3(0.0);

        vec3 direction = normalize(jitter_worldPosition);
        vec2 tracingPlanet = RaySphereIntersection(E, direction, vec3(0.0), planet_radius);
        vec2 tracingAtmospheric = RaySphereIntersection(E, direction, vec3(0.0), atmosphere_radius);

        color += DrawMoon(moonVectorWorld, direction, tracingPlanet.x);

        float sunDiscAngle = rescale(0.9995, 1.0, dot(direction, sunVectorWorld));
        color += SunLight * 10.0 * (step(1e-5, sunDiscAngle) * step(tracingPlanet.x, 0.0) * rescale(-0.05, 1.0, sunDiscAngle));

        vec3 stars = DrawStars(direction, tracingPlanet.x) * Stars_Luminance;
        color += vec3(10.0) * saturate(stars - vec3(dot(vec3(1.0 / 3.0), skyLightingColorRaw) * 100.0 * (tracingAtmospheric.y > 0.0 || tracingAtmospheric.x > 0.0 ? 1.0 : 0.0)));

        color *= transmittance;

        color += skyColor;
    }

    color = encodeGamma(color);

    gl_FragData[0] = vec4(encodeGamma(albedo), 0.2);
    gl_FragData[1] = vec4(texture(depthtex0, texcoord).x, texture2D(gnormal, texcoord).xy, 1.0);
    gl_FragData[2] = vec4(color, 1.0);
}
/* DRAWBUFFERS:045 */
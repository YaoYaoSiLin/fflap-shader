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

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

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

    shading = mix(vec3(1.0), extinction * mix(tint, vec3(1.0), absorption), vec3(hlaf_visibility) * step(alpha, 0.99));
    //shading = vec3(mix(visibility1, visibility0, step(0.99, alpha)));
    shading *= visibility1;

    return shading;
}

vec3 CalculateShading2(in vec3 coord, float shadowDepth){
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

    shading = vec3(visibility1);
    shading = shading * mix(tint, vec3(1.0), absorption) * extinction;

    return shading;
}

vec3 CalculateShading(in vec3 coord, vec3 halfvector){
    return CalculateShading(coord, length(halfvector));
}

vec3 CalculateSunShading(in vec2 texel, in vec3 normal, in float scale, in float material, in float parallaxDepth){
    vec4 worldPosition = gbufferProjectionInverse * nvec4(vec3(texel, texture(depthtex0, texel).x) * 2.0 - 1.0);
         worldPosition = gbufferModelViewInverse * (worldPosition / worldPosition.w);

         //worldPosition.xyz = normalize(worldPosition.xyz) * (length(worldPosition.xyz) + parallaxDepth);

    float distortion = 1.0;
/*
    //worldPosition.xyz += mat3(gbufferModelViewInverse) * normal / (1e-4 + max(0.0, dot(normal, lightVectorView))) * 0.001;

    vec3 shadowCoord = wP2sP(worldPosition, distortion);
*/

    vec3 projection = ProjectionToShadowMap(worldPosition.xyz);

    vec3 shadowCoord = projection;
    CalculateShadowCoordDistortion(shadowCoord, distortion);
    shadowCoord = shadowCoord * 0.5 + 0.5;
    CalculateShadowCoordRescale(shadowCoord);

    float visibility = dot(normal, lightVectorView);

    float diffthresh = (5.0 / distortion / (0.1 + abs(visibility) * 0.9));// / (1e-2 + pow2(abs(visibility))) * 0.4 + distortion * 0.1;
          diffthresh *= shadowPixel * 1.5;
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
    
    #if Soft_Shadow_Quality == OFF
    
    float depth0 = texture(shadowtex0, shadowCoord.xy).x;

    vec3 position0 = vec3(shadowProjectionInverse[0].x, shadowProjectionInverse[1].y, shadowProjectionInverse[2].z) * (projection) + shadowProjectionInverse[3].xyz;
    vec3 position1 = vec3(shadowProjectionInverse[0].x, shadowProjectionInverse[1].y, shadowProjectionInverse[2].z) * vec3(projection.xy, depth0 * 2.0 - 1.0) + shadowProjectionInverse[3].xyz;

    shading = CalculateShading(shadowCoord, max(length(position1 - position0) - 0.05, 0.0));
    #else

    #if Soft_Shadow_Quality == Medium || Soft_Shadow_Quality == High
    for(float i = -radius; i <= radius; i += 1.0) {
        for(float j = -radius; j <= radius; j += 1.0) {

            #if Soft_Shadow_Quality != High
            if(length(vec2(i, j)) > max_radius * 0.78) continue;
            #endif

            vec2 offset = vec2(i - dither2 + 0.5, j + dither2 - 0.5) * rotate * (penumbra);

            vec2 coord = shadowCoord.xy + offset * texelSize;

            float depth0 = texture(shadowtex0, coord).x;

            if(receiver > depth0) {
                blocker0 += depth0;
                count0++;
            }
        }
    }

    blocker0 /= max(1.0, float(count0));

    penumbra *= abs(receiver - blocker0) / blocker0;
    #else
    penumbra *= 0.1;
    #endif

    if(material > 64.0) penumbra += (1.0 - material / 191.0) * 0.5;

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
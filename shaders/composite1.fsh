#version 130

#define SHADOW_MAP_BIAS 0.9

#define SpecularityReflectionPower 2.0            //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

uniform vec3 cameraPosition;
uniform vec3 sunPosition;
uniform vec3 shadowLightPosition;

uniform float viewWidth;
uniform float viewHeight;

uniform int isEyeInWater;

in vec2 texcoord;

in float fading;
in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#define CalculateHightLight 1
#define CalculateShadingColor 2

#include "libs/common.inc"
#include "libs/dither.glsl"
#include "libs/brdf.glsl"
#include "libs/light.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

void main(){
  vec4 albedo = texture2D(gcolor, texcoord);

  vec4 color = texture2D(gaux2, texcoord);
       //color = rgb2L(color);

  float alpha = texture2D(gnormal, texcoord).z;

  float smoothness = texture2D(composite, texcoord).r;
  float metallic   = texture2D(composite, texcoord).g;
  float emissive   = texture2D(composite, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;
  float r = 1.000293 / (1.0 + texture2D(composite, texcoord).b);
  if(isEyeInWater > 0) r = 1.333 / 1.000293;

  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);

  float depth = texture2D(depthtex0, texcoord).x;

  bool isSky = texture2D(gdepth, texcoord).z > 0.999;
  bool isWater = int(round(texture2D(gdepth, texcoord).z * 255.0)) == 8;

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0); vP /= vP.w;
  vec4 wP = gbufferModelViewInverse * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(nvP, normal);
  vec3 nrP = normalize(reflect(nvP, normal));
  vec3 refractP = refract(nvP, normal, r);

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  float dither = bayer_32x32(texcoord, resolution);

  if(albedo.a > 0.99){
    vec3 h = normalize(nrP - nvP);

    float ndoth = clamp01(dot(normal, h));

    float vdoth = 1.0 - clamp01(dot(-nvP, h));

    vec3 f = F(F0, pow5(vdoth));

    float ndotl = clamp01(dot(nrP, normal));
    float ndotv = 1.0 - clamp01(dot(-nvP, normal));

    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);
    float specularity = pow((1.0 - g) * clamp01(d), SpecularityReflectionPower);

    float dither2 = hash(texcoord * resolution);

    vec2 rcoord = nvec3(gbufferProjection * nvec4(vP.xyz + refractP * (clamp01(dot(-nvP, normal))))).xy * 0.5 + 0.5;
    //if(floor(rcoord) != vec2(0.0) || (texture2D(composite, texcoord).b > 0.519 && texture2D(composite, texcoord).b < 0.521)) rcoord = clamp(rcoord + (vec2(dither2, 1.0 - dither) - 0.5) * pixel * (1.0 + alpha * 7.0), pixel, 1.0 - pixel);

    vec4 color2 = texture2D(gaux2, rcoord);
    float tranBlockR = texture2D(gcolor, rcoord).a;
    vec4 solidBlockSpecular = texture2D(gaux3, rcoord);

    //color.rgb = mix(color2.rgb, color.rgb, clamp01(color2.a * 1.95 - 0.95));
    color.rgb = color2.rgb;
    color.rgb = mix(color.rgb, color.rgb * 2.0 - 1.0, tranBlockR * tranBlockR);
    //color.rgb += solidBlockSpecular.rgb;

    vec3 transBlockAlbedo = rgb2L(albedo.rgb);

    vec4 sunDirctLighting = CalculateShading(shadowtex0, wP);
    vec3 shading = mix(vec3(1.0), sunDirctLighting.rgb, sunDirctLighting.a);

    vec3 transBlockColor = transBlockAlbedo * skyLightingColorRaw;
         transBlockColor *= 1.0 - metallic;
         //transBlockColor += BRDF(transBlockAlbedo, normalize(shadowLightPosition), -nvP, normal,  roughness, metallic, F0) * shading * sunLightingColorRaw * pow(fading, 5.0) * 4.0;

    transBlockColor = L2rgb(transBlockColor);

    vec3 halfVector = nvec3(gbufferProjectionInverse * nvec4(vec3(rcoord, texture2D(depthtex0, rcoord).x) * 2.0 - 1.0)) - nvec3(gbufferProjectionInverse * nvec4(vec3(rcoord, texture2D(depthtex1, rcoord).x) * 2.0 - 1.0));
    // = clamp01(1.0 - exp(-length(halfVector) * alpha)) * pow5(clamp01(dot(normalize(vP), normalize(vP + refract(normalize(vP), normalMap, 1.000293 / r)))));

    if(length(refractP) < 0.99) color.rgb = albedo.rgb;

    vec3 absorption = mix(color.rgb, color.rgb * albedo.rgb, alpha * (1.0 - albedo.rgb));

    vec3 scattering = L2rgb(transBlockAlbedo * skyLightingColorRaw);
         scattering *= 1.0 - metallic;
         scattering = L2rgb(rgb2L(scattering) + BRDF(transBlockAlbedo, normalize(shadowLightPosition), -nvP, normal, roughness, metallic, F0) * shading * sunLightingColorRaw * pow(fading, 5.0) * 4.0);

    vec3 absorptionFactor = 1.0 - albedo.rgb;
    float scatteringFactor = 1.0 - pow5(clamp01(exp(-min(1.0 + far * 0.5 * float(isWater), length(halfVector)) * (alpha * alpha))));
    if(isEyeInWater > 0) scatteringFactor = 0.0;

    vec3 solidBlockColor = color.rgb;

    color.rgb = mix(absorption, scattering, scatteringFactor);
    //color.rgb = mix(color.rgb, transBlockColor, alpha);
    color.rgb *= clamp01(1.0 - f * specularity);
    color.rgb = rgb2L(color.rgb);
    color.rgb += BRDF(transBlockAlbedo, normalize(shadowLightPosition), -nvP, normal, roughness, metallic, F0) * shading * sunLightingColorRaw * pow(fading, 5.0) * 4.0;
    color.rgb = L2rgb(color.rgb);

    //color.rgb = shading;

    //color.rgb = vec3(min(alpha, color.a));
  }

  //color.rgb = texture2D(gaux3, texcoord).rgb;

  //color = L2rgb(color);

/* DRAWBUFFERS:5 */
  gl_FragData[0] = vec4(color.rgb, float(!isSky) + albedo.a);
}

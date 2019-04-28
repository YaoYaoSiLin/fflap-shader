#version 130

#define SpecularityReflectionPower 2.0            //[1.0 1.2 1.5 1.75 2.0 2.25 2.5 2.75 3.0]

uniform sampler2D gcolor;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;

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

#include "libs/common.inc"
#include "libs/atmospheric.glsl"
#include "libs/brdf.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

vec3 RemovalColor(in vec3 image, in vec3 color, in float t){
  return (image - color) / (1.0 - t) + color;
}

float maxComponent(in vec3 a) {
  return max(a.x, max(a.y, a.z));
}

void main(){
  vec4 albedo = texture2D(gcolor, texcoord);
  bool isTranslucentBlocks = albedo.a > 0.99;

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);
  float alpha = texture2D(gnormal, texcoord).z;

  float smoothness = texture2D(composite, texcoord).r;
  float metallic   = texture2D(composite, texcoord).g;
  float emissive   = texture2D(composite, texcoord).b;
  float roughness  = 1.0 - smoothness;
        roughness  = roughness * roughness;
  float r = texture2D(composite, texcoord).b;
  float IOR = 1.000293;
  if(isTranslucentBlocks){
    r = 1.0 + r;
    if(isEyeInWater > 0) IOR = r / IOR;
    else IOR = IOR / r;
  }

  vec4 color = texture2D(gaux2, texcoord);

  float depth = texture2D(depthtex0, texcoord).x;

  vec3 vP = nvec3(gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0));
  vec3 wP = mat3(gbufferModelViewInverse) * vP;
  vec3 nvP = normalize(vP.xyz);
  vec3 rP = reflect(nvP, normal);
  vec3 refractP = refract(nvP, normal, IOR);

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  //albedo.rgb = rgb2L(albedo.rgb);

  //Side A
  if(albedo.a > 0.99){
    vec3 skySpecularReflection = L2rgb(CalculateSky(normalize(rP), sP, cameraPosition.y, 0.5));

    //vec2 rcoord = nvec3(gbufferProjection * nvec4(vP.xyz + refractP)).xy * 0.5 + 0.5;

    //color = texture2D(gaux2, rcoord);

    vec3 F0 = vec3(max(0.02, metallic));
         F0 = mix(F0, albedo.rgb, step(0.5, metallic));

    vec3 h = normalize(rP - nvP);

    float ndoth = clamp01(dot(normal, h));
    float vdoth = 1.0 - clamp01(dot(-nvP, h));

    vec3 f = (F(F0, pow5(vdoth)));

    float ndotl = clamp01(dot(rP, normal));
    float ndotv = 1.0 - clamp01(dot(-nvP, normal));

    float d = DistributionTerm(roughness, ndoth);
    float g = VisibilityTerm(d, ndotv, ndotl);
    float specularity = pow(1.0 - g, SpecularityReflectionPower);

    //if(floor(alpha) == 0.0) {
      vec3 translucentBlockColor = rgb2L(texture2D(gcolor, texcoord).rgb);

      float shading = pow(clamp01(dot(normalize(shadowLightPosition), normal)), 0.2);

      translucentBlockColor = translucentBlockColor * sunLightingColorRaw * fading * shading + translucentBlockColor * skyLightingColorRaw;
      translucentBlockColor = L2rgb(translucentBlockColor);

      vec3 tranBlockWithSpecular = mix(translucentBlockColor, skySpecularReflection, f * specularity);

      //if(color.a < 0.999){
        vec3 solidBlockColor = RemovalColor(color.rgb, tranBlockWithSpecular.rgb, color.a);
             solidBlockColor = mix(solidBlockColor, vec3(0.0), clamp01((color.a * 1.0 - 0.92) * 12.5));

        //color.rgb = RemovalColor(color.rgb, clamp01(solidBlockColor), 1.0 - color.a);
        //color.rgb = RemovalColor(texture2D(gaux2, texcoord).rgb, color.rgb, color.a);

        color.rgb = RemovalColor(color.rgb, tranBlockWithSpecular, color.a);
        //if(alpha > 0.9) color.rgb = texture2D(gaux3, texcoord).rgb;
        //color.rgb = mix(clamp01(color.rgb), texture2D(gaux3, texcoord).rgb, float(floor(color.r) != 0.0 || floor(color.g) != 0.0 || floor(color.b) != 0.0));

        //color.rgb = RemovalColor(color.rgb, skySpecularReflection, dot(f, vec3(0.3333)) * specularity);
        solidBlockColor = mix(solidBlockColor, vec3(0.0), clamp01((color.a * 1.0 - 0.92) * 12.5));
        //color.rgb = clamp01(color.rgb);

        //color.rgb = RemovalColor(color.rgb, solidBlockColor, 1.0 - color.a);
        //color.rgb = RemovalColor(color.rgb, skySpecularReflection, dot(vec3(0.3333), f) * specularity);
      //}else{
        //color.rgb = vec3(0.0);
      //}

      //color.rgb = RemovalColor(color.rgb, translucentBlockColor.rgb, color.a);
    //}
    //color.rgb = color.aaa * 0.1;
  }

  //color.rgb = mix(color.rgb, texture2D(gaux3, texcoord).rgb, clamp01((alpha * 1.0 - 0.92) * 12.5));
  //color.rgb = mix(texture2D(gaux3, texcoord).rgb, color.rgb, clamp01((color.a * 1.0 - 0.92) * 12.5));
  color.rgb = mix(color.rgb, color.rgb * 0.5 + 0.5, albedo.a * albedo.a);

  //color = L2rgb(color);

/* DRAWBUFFERS:05 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = color;
}

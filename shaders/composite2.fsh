#version 130

uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D gaux1;
uniform sampler2D gaux2;
uniform sampler2D gaux3;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform vec3 upPosition;
uniform vec3 sunPosition;
uniform vec3 cameraPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;
uniform int isEyeInWater;

in float fading;

in vec2 texcoord;

in vec3 sunLightingColorRaw;
in vec3 skyLightingColorRaw;

in vec4 waterColor;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

#include "libs/common.inc"
#include "libs/jittering.glsl"
#include "libs/dither.glsl"

#define Low_Quality_Reflection
  #define Reflection_Scale Medium                //[Low Medium High]
  #define Reflection_Scale_Type Checker_Board    //[Checker_Board Render_Scale]

#if Reflection_Scale == Low
  #define reflection_resolution_scale 4
#elif Reflection_Scale == Medium
  #define reflection_resolution_scale 2
#elif Reflection_Scale == High
  #define reflection_resolution_scale 1
#endif

#define Stage ScreenReflection
#define depthOpaque depthtex0
#define reflectionSampler gaux2

//#define CheckerBoard_Rendering
//  #define Reflection_CBR_Screen_Scale 2 //[2 4]

#include "libs/brdf.glsl"
#include "libs/atmospheric.glsl"
#include "libs/specular.glsl"

vec3 normalDecode(vec2 enc) {
    vec4 nn = vec4(2.0 * enc - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

void main() {
  float depth = texture2D(depthtex0, texcoord).x;
  //float particlesDepth = texture2D(gaux1, texcoord).x;
  //if(linearizeDepth(depth) > linearizeDepth(particlesDepth) && particlesDepth > 0.0) depth = particlesDepth;

  vec4 vP = gbufferProjectionInverse * nvec4(vec3(texcoord, depth) * 2.0 - 1.0);
       vP /= vP.w;
  vec4 wP = gbufferModelViewInverse * vP;
  vec3 nvP = normalize(vP.xyz);

  vec2 juv = vec2(0.0);
  vec3 jvP = vP.xyz;

  #ifdef Enabled_TAA
    juv = clamp(texcoord + R2sq2[int(mod(frameCounter, 16))] * pixel, pixel, 1.0 - pixel);
    jvP = nvec3(gbufferProjectionInverse * nvec4(vec3(juv, texture2D(depthtex0, juv).x) * 2.0 - 1.0));
  #endif

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  vec4 albedo = texture2D(gcolor, texcoord);

  int id = int(round(texture2D(gdepth, texcoord).z * 255.0));
  bool isPlants = id == 18 || id == 31 || id == 83;
  bool isParticles = id == 253;
  bool isSky = id == 255;

  float torchLightMap = texture2D(gdepth, texcoord).x;
  float skyLightMap = texture2D(gdepth, texcoord).y;

  vec3 normal = normalDecode(texture2D(gnormal, texcoord).xy);

  vec3 color = texture2D(gaux2, texcoord).rgb;

  vec3 reflectP = reflect(nvP, normal);
  vec3 h = normalize(normalize(reflectP) - nvP);

  float alpha = 1.0;
  if(albedo.a > 0.9) alpha = texture2D(gnormal, texcoord).z;

  float smoothness = (texture2D(composite, texcoord).x);
        //smoothness = pow2(smoothness);
  float roughness = 1.0 - smoothness;
        //roughness = roughness * roughness;
  float metallic = texture2D(composite, texcoord).g;
  vec3 F0 = vec3(max(0.02, metallic));
       F0 = mix(F0, albedo.rgb, step(0.5, metallic));

  float ndoth = clamp01(dot(normal, h));
  float vdoth = pow5(1.0 - clamp01(dot(-nvP, h)));
  float ndotl = clamp01(dot(normalize(reflectP), normal));
  float ndotv = 1.0 - clamp01(dot(-nvP, normal));

  vec3 f = F(F0, vdoth);
  float d = DistributionTerm(roughness, ndoth);
  float g = VisibilityTerm(d, ndotv, ndotl);

  vec4 reflection = vec4(vec3(0.0), depth);

  //vec2 checkerBoard = vec2(mod(fragCoord.x, 2), mod(fragCoord.y, 2));

  bool cutPixel = true;

  #ifdef Low_Quality_Reflection
  #if reflection_resolution_scale == 2
  vec2 fragCoord = floor(texcoord * resolution);
       fragCoord.x = fragCoord.x + fragCoord.y;
  cutPixel = mod(fragCoord.x, 2) < 0.5;
  #elif reflection_resolution_scale == 4
  vec2 fragCoord = floor(texcoord * resolution);
       fragCoord.x = fragCoord.x * fragCoord.y;
  cutPixel = mod(fragCoord.x, 2) > 0.5;
  #endif
  #endif

  //if(cutPixel) color = vec3(0.0);

  vec3 torchLightingColor = rgb2L(vec3(1.022, 0.782, 0.344));

  if(!isSky && cutPixel && smoothness > 0.0001){
  //vec3 diffuse = rgb2L(albedo.rgb);
  //     diffuse = L2rgb(diffuse * skyLightMap * skyLightingColorRaw * 1.99 + diffuse * torchLightMap * torchLightingColor) / overRange * (1.0 - metallic);

  /*
  xxx
  xox
  xxx
  */

  vec2 normalMuti[8] = vec2[8](vec2(-0.7, 0.7),
                               vec2(0.0, 1.0),
                               vec2(0.7, 0.7),
                               vec2(0.0, 1.0),
                               vec2(0.7, -0.7),
                               vec2(0.0, 1.0),
                               vec2(-0.7, -0.7),
                               vec2(-1.0, 0.0));

  #if reflection_resolution_scale > 1
  vec2 resize = vec2(sqrt(resolution.x * resolution.y / reflection_resolution_scale), 0.0);
       resize = vec2(resize.x * aspectRatio, resize.x / aspectRatio);

  float dither = R2sq((texcoord - vec2(frameCounter) * pixel * 0.0625) * resize);
  #else
  float dither = R2sq(texcoord * resolution - vec2(frameCounter) * 0.125);
  #endif

  //float dither = R2sq(texcoord * resolution);

  vec3 t = normalize(cross(normalize(-upPosition), normal));
  vec3 b = cross(normal, t);
  mat3 tbn = mat3(t, b, normal);

  dither *= 2.0 * Pi;
  mat2 rotate = mat2(cos(dither), -sin(dither), sin(dither), cos(dither));

  vec3 n = vec3(normalMuti[int(mod(frameCounter, 8))] , 1.0);
       //n.xy = (n.xy - n.xy * dither * 0.1591) * 0.04 * rotate;
       //n.xy = n.xy * rotate * 0.04;
       n.xy = n.xy * 0.01 * rotate / max(0.04, d * g);
       n = normalize(tbn * n);
  //     n = mix(normal, n, clamp01((-dot(nvP, n) - 0.05) / 0.95));
  //     if(dot(-nvP, n) < 0.15) n = normal;

  reflectP = reflect(normalize(jvP.xyz), n);

  vec3 skySpecularReflection = CalculateSky(normalize(reflectP), sP, 0.0, 0.5);
       skySpecularReflection = CalculateAtmosphericScattering(skySpecularReflection, -(mat3(gbufferModelViewInverse) * normalize(reflectP)).y + 0.15);
       skySpecularReflection = L2rgb(skySpecularReflection) / overRange;

  float fogScattering = waterColor.a * min(far, length(vP.xyz)) * pow3(waterColor.a + 1.0);
        fogScattering = clamp01(exp(-fogScattering));

  float fogAbsorption = 1.0 - minComponent(waterColor.rgb);
        fogAbsorption = 1.0 - clamp01(exp(-(1.0 + fogAbsorption * min(far, length(vP.xyz)))));

  if(isEyeInWater == 1) skySpecularReflection = waterColor.rgb / overRange * L2rgb(skyLightingColorRaw * 0.7);
  skySpecularReflection = mix(color, skySpecularReflection, min(step(0.7, skyLightMap) + metallic * skyLightMap, clamp01(d * g)));

  reflection.a = P2UV(reflectP + vP.xyz).z;

  vec4 ssR = vec4(0.0);
  if(!isPlants && !isPlants){
    vec3 hitPosition = vec3(0.0);
    ssR = raytrace(vP.xyz + normal * 0.1 * vdoth, reflectP, hitPosition);

    ssR.rgb = mix(skySpecularReflection, ssR.rgb, clamp01(d * g));
    //if(ssR.a > 0.5) reflection.a = P2UV(hitPosition + vP.xyz).z;
    reflection.a = mix(reflection.a, P2UV(hitPosition + vP.xyz).z, step(0.5, ssR.a));
  }

  reflection.rgb = mix(skySpecularReflection, ssR.rgb, ssR.a);

  if(isEyeInWater == 1){
    reflection.rgb *= fogScattering;
    reflection.rgb *= mix(vec3(1.0), waterColor.rgb, fogAbsorption);
  }
  //reflection.rgb *= fogScattering;
  //reflection.rgb = vec3(dot(normalize(upPosition - vP.xyz), normal));

  //reflection.rgb = vec3(0.1);

  //#ifndef Low_Quality_Reflection
  //color += reflection.rgb * f * clamp01(g * d);
  //#endif
  }

/* DRAWBUFFERS:34 */
  gl_FragData[1] = reflection;

  //#ifndef Low_Quality_Reflection
  //gl_FragData[2] = vec4(color, 1.0);
  //#else
  gl_FragData[0] = vec4(f, g * d);
  //#endif
}

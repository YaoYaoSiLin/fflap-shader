#version 130

#define SHADOW_MAP_BIAS 0.9

const float shadowDistance = 140.0;

uniform sampler2D texture;
uniform sampler2D shadowtex1;
uniform sampler2D noisetex;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform vec3 shadowLightPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float far;

in float shadowPass;
in float isWater;
in float blockDepth;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 vP;

in vec4 color;

#include "libs/water.glsl"

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
	vec4 tex = texture2D(texture, texcoord) * color;

  //tex.rgb = mix(tex.rgb, vec3(0.0), tex.a);
	//tex.rgb *= (1.0 - tex.a);

	if(isWater > 0.5) {
    //tex = vec4(0.02);
    tex = vec4(color.rgb, 0.05);
    tex = CalculateWaterColor(tex);
    //tex.a = 0.5;
    //tex.rgb *= tex.a * 0.1;
    //tex.a = 0.2;
    //tex.a = 0.0;
	}

  //if(isWater > 0.5) {
    //blockDepthAndLiquid = length(vP - vPsolid.xyz);
  //}

  float scatteringFactor = clamp(exp(-tex.a * blockDepth * pow3(tex.a + 1.0)), 0.0, 1.0);
  tex.rgb *= scatteringFactor;
  tex.rgb *= clamp(pow2(dot(normalize(normalize(shadowLightPosition) + normal), (normal))), 0.0, 1.0);

  float receiver = (-vP.z) * 0.5 + 0.5 - 0.001;
  float blocker = gl_FragCoord.z;
  if(blocker > 0.9996) tex.a = 1.0;

  //float penumbra = (receiver - blocker) / blocker;

  //if(isWater > 0.5){
    //tex.rgb *= clamp(penumbra, 0.0, 1.0);
  //}

  //vec3 uv = shado

  //tex.rgb = gl_FragCoord.zzz * 2.0 - 1.0;
  //if(penumbra > 0.1) tex.rgb = vec3(1.0, 0.0, 0.0);
  //if(length(vPsolid) > 10.0) tex.rgb = vec3(1.0, 0.0, 0.0);

	//tex.rgb *= pow(clamp(dot(normal, mat3(gbufferModelViewInverse) * normalize(shadowLightPosition)) * 0.5 + 0.5, 0.0, 1.0), 0.2);
  //tex.rgb = mix(tex.rgb, vec3(0.0), tex.a);

/* DRAWBUFFERS:01 */
	gl_FragData[0] = vec4(blocker, vec2(0.0, 0.0), tex.a);
	gl_FragData[1] = vec4(tex.rgb, step(tex.a, 0.9) * 0.9);
}

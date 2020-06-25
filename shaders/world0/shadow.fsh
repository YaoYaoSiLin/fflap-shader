#version 130

#define SHADOW_MAP_BIAS 0.9

const float shadowDistance = 140.0;

uniform sampler2D texture;
uniform sampler2D noisetex;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

uniform vec3 shadowLightPosition;

uniform float viewWidth;
uniform float viewHeight;
uniform float far;

in float shadowPass;
in float isWater;
in float isLava;
in float blockDepth;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 worldNormal;
in vec3 vP;

in vec4 color;

const float Pi = 3.14159265;

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

  if(!gl_FrontFacing) discard;

	if(isWater > 0.5) {
    tex.rgb = vec3(1.0);
    tex.a = 0.05;
	}

  //tex.rgb *= 1.0 - isLava;

  vec3 worldLightPosition = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
  float ndotl = max(0.0, dot(worldLightPosition, worldNormal));
  tex.rgb *= min((ndotl * ndotl * 32.0), 1.0);

  float scatteringFactor = tex.a * blockDepth * Pi;
  vec3 absorption = scatteringFactor * (1.0 - tex.rgb);

  //scatteringFactor = min(exp(-scatteringFactor), 1.0);
  absorption = exp(-tex.a * Pi * blockDepth * (1.0 - tex.rgb));

  //tex.rgb = absorption;
  //tex.rgb *= scatteringFactor;


  //if(length(vP.xy) < 10.0) discard;

  //scatteringFactor = step(tex.a, 0.99);
  //tex.rgb *= clamp(pow2(dot(normalize(normalize(shadowLightPosition) + normal), (normal))), 0.0, 1.0);
  //tex.rgb = absorption;
  //tex.rgb *= 1.0 - scatteringFactor;

  //float receiver = (-vP.z) * 0.5 + 0.5 - 0.001;

  //vec3 uv = nvec3(shadowProjection * nvec4(vP)).xyz * 0.5 + 0.5;

  float blocker = gl_FragCoord.z;
  //if(blocker > 0.996) tex.rgba = vec4(vec3(0.0), 1.0);

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

  vec3 normal = worldNormal;
       //normal = mat3()
       //normal = mat3()
       //normal -= normal * 0.05;

/* DRAWBUFFERS:01 */
	gl_FragData[0] = vec4(worldNormal * 0.5 + 0.5, tex.a);
	gl_FragData[1] = vec4(tex.rgb, tex.a);
}

#version 130

#define SHADOW_MAP_BIAS 0.9

const float shadowDistance = 140.0;

uniform sampler2D texture;
uniform sampler2D specular;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

uniform vec3 shadowLightPosition;

uniform float far;

in float isSphereMap;
in float sphereMapDepth;
in vec4 sphereViewPosition;

out float blockID;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 world_normal;

in vec4 color;

const float Pi = 3.14159265;

#define pow5(x) (x*x*x*x*x)

float saturate(in float x){
  return clamp(x, 0.0, 1.0);
}

vec3 nvec3(vec4 pos) {
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos) {
    return vec4(pos.xyz, 1.0);
}

vec3 F(vec3 F0, float cosTheta){
 return F0 + (1.0 - F0) * cosTheta;
}

vec3 F(vec3 F0, vec3 V, in vec3 N){
  float cosTheta = pow5(1.0 - saturate(dot(V, N)));

 return F(F0, cosTheta);
}


#define CalculateMaskID(id, x) bool(step(id - 0.5, x) * step(x, id + 0.5))

void main() {
	vec4 tex = texture2D(texture, texcoord) * color;

  if(!gl_FrontFacing || tex.a < 0.004) discard;

  float maxDistance = 1.0;

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);

	if(CalculateMaskID(8.0, blockID)) {
    tex = color;
    maxDistance = (16.0);
	}

  if(CalculateMaskID(106.0, blockID) || CalculateMaskID(160.0, blockID)){
    maxDistance = 0.125;
  }
  /*
  if(!bool(isSphereMap)){
    float metallic = texture2D(specular, texcoord).g;

    vec3 F0 = mix(vec3(max(0.02, metallic)), tex.rgb, step(0.5, metallic));
    vec3 f = F(F0, worldLightVector, world_normal);

    tex.rgb *= 1.0 - f;
  }
  */
  vec3 scatteringcoe = vec3(1.0);
       scatteringcoe = scatteringcoe * Pi * (tex.a * tex.a);

  vec2 lightMap = vec2(lmcoord.x, lmcoord.y);

  float data1A = dot(floor(lightMap * 15.0), vec2(1.0, 16.0)) / 255.0;
  if(!bool(isSphereMap)) data1A = maxDistance / 16.0;

  vec3 data1RGB = mix(world_normal * 0.5 + 0.5, sphereViewPosition.xyz, step(0.5, isSphereMap));

  float ndotl = dot(worldLightVector, world_normal);

  vec4 data0 = tex;
  data0.a = mix(data0.a, ndotl * 0.5 + 0.5, step(0.5, isSphereMap));

  /* DRAWBUFFERS:01 */
	gl_FragData[0] = data0;
	gl_FragData[1] = vec4(data1RGB, data1A);
}

#version 130

#extension GL_ARB_shader_texture_lod : enable

#define SHADOW_MAP_BIAS 0.9

const float shadowDistance = 140.0;

uniform sampler2D texture;
uniform sampler2D specular;

uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

uniform vec4 gbufferProjection0;
uniform vec4 gbufferProjection1;
uniform vec4 gbufferProjection2;
uniform vec4 gbufferProjection3;

uniform vec3 shadowLightPosition;

uniform float far;
uniform float near;

in float isSphereMap;
in float sphereMapDepth;
in vec4 sphereViewPosition;

in vec3 vworldPosition;
in float vLength;

in float blockID;

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

float pack2x4(in vec2 x) {
  return dot(floor(x * 15.0), vec2(1.0, 16.0)) / 255.0;
}

#define CalculateMaskID(id, x) bool(step(id - 0.5, x) * step(x, id + 0.5))

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

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
	vec4 tex = texture2D(texture, texcoord) * color;
  //tex.rgb = texture2DLod(texture, texcoord, 3).rgb * color.rgb;

  float maxDistance = 1.0;

  float absorption = 0.0;
  float material = 0.0;

	if(CalculateMaskID(8.0, blockID)) {
    tex = color;
    tex.a = 0.1;

    material = 230.0;
    absorption = 2.0;

    maxDistance = (16.0);
	}else if(CalculateMaskID(20.0, blockID) || CalculateMaskID(95.0, blockID) || CalculateMaskID(106.0, blockID) || CalculateMaskID(160.0, blockID)){
    if(CalculateMaskID(106.0, blockID) || CalculateMaskID(160.0, blockID)) maxDistance = 0.125;
    absorption = 15.0;
    material = 255.0;
  }else{
    material = 0.0;//max(0.0, texture2D(specular, texcoord).b * 255.0 - 64.0) / 191.0 * 255.0;
    if(tex.a < 0.2) discard;
  }

  //if(!gl_FrontFacing) discard;
  //if(!CalculateMaskID(8.0, blockID) && tex.a < 0.05) discard;

  //if(material > 65.0 && tex.a < 0.99){
  //  tex.a *= max(0.05, (1.0 - (material - 65.0) / 190.0) * 10.0);
  //  tex.a = min(tex.a, 0.98);
  //}

  vec3 worldLightVector = mat3(gbufferModelViewInverse) * normalize(shadowLightPosition);
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

  vec2 lightMap = lmcoord;

  vec2 material2 = vec2(absorption / 15.0, material / 255.0);

  float data1A = 0.5 < isSphereMap ? dot(floor(lightMap * 15.0), vec2(1.0, 16.0)) / 255.0 : dot(floor(material2 * 15.0), vec2(1.0, 16.0)) / 255.0;
  //data1A = float(packUnorm2x8(lightMap)) / 255.0;

  //if(!bool(isSphereMap)) data1A = maxDistance / 16.0;

  vec3 data1RGB = mix(world_normal * 0.5 + 0.5, sphereViewPosition.xyz, step(0.5, isSphereMap));

  float ndotl = dot(worldLightVector, world_normal);

  float l = length(vworldPosition);
  vec3 coord = vworldPosition;

  vec3 n = abs(coord);

  if(n.x > max(n.y, n.z)) {
    coord = coord.yzx;
    if(vworldPosition.x > 0.0) coord = -coord;
  }else if(n.y > max(n.x, n.z)) {
    coord = coord.zxy;
    if(vworldPosition.y > 0.0) coord = -coord;
  }else if(n.z > max(n.x, n.y)) {
    coord = coord.xyz;
    if(vworldPosition.z > 0.0) coord = -coord;
  }

  float depth = (1.0 / (coord.z)) * far * near / (far - near) + 0.5 * (far + near) / (far - near) + 0.5;

  float normalID = 0.0;

  vec4 data0 = tex;
  data0.a = isSphereMap > 0.9 ? 0.2 + (ndotl * 0.5 + 0.5) * 0.8 : tex.a;

	gl_FragData[0] = data0;
	gl_FragData[1] = vec4(data1RGB, data1A);

  gl_FragDepth = isSphereMap > 0.9 ? depth : gl_FragCoord.z;
}
/* DRAWBUFFERS:01 */

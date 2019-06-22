#version 120

#define SHADOW_MAP_BIAS 0.9

const float shadowDistance = 140.0;

attribute vec4 mc_Entity;
attribute vec4 mc_midTexCoord;

uniform sampler2D noisetex;

uniform float frameTimeCounter;
uniform float far;

uniform mat4 gbufferModelView;

varying float shadowPass;
varying float isWater;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 normal;
varying vec3 vP;

varying vec4 color;

void main() {
	shadowPass = 0.0;
	if(mc_Entity.x == 95 || mc_Entity.x == 160 || mc_Entity.x == 79 || mc_Entity.x == 20 || mc_Entity.x == 165){
	//if(mc_Entity.y == 3){
		shadowPass = 1.0;
	}

	isWater = 0.0;
	if(mc_Entity.x == 8){
		isWater = 1.0;
	}

	texcoord = gl_MultiTexCoord0.xy;
	lmcoord  = gl_MultiTexCoord1.xy;

	color = gl_Color;

	normal = normalize(gl_NormalMatrix * gl_Normal);

	bool leaves = mc_Entity.x == 18;

  bool plant = mc_Entity.x == 31;

  bool double_plant_upper = mc_Entity.x == 175;
  bool double_plant_lower = mc_Entity.x == 176;
  bool double_plant = double_plant_upper || double_plant_lower;

  //bool farm = mc_Entity.x == 59 || mc_Entity.x == 141 || mc_Entity.x == 142 || mc_Entity.x == 207;

	vP = (gl_ModelViewMatrix * gl_Vertex).xyz;

	vec4 position = gl_Vertex;

  if(double_plant || plant) {
    vec2 noise = texture2D(noisetex, (position.xz) / 64.0).xz * 8.0;
    float time = frameTimeCounter * 0.74;
    float time2 = sin(frameTimeCounter * 0.48) * 1.21;
    vec2 noise2 = vec2(sin(time2 + noise.x * 1.3), sin(time2 + noise.y * 1.3));

    vec2 wave = vec2(sin((time + noise.x) * 3.14159 * 0.5 + noise2.y + time2), sin((time + noise.y) * 3.14159 * 0.5 + noise2.x + time2)) * 0.016;

    if(double_plant_lower || plant) {
      wave *= float(mc_midTexCoord.y > gl_MultiTexCoord0.y);
    }else{
      wave *= 1.0 + (mc_midTexCoord.y - gl_MultiTexCoord0.y);
    }

    if(double_plant) wave *= 0.632;

    position.xz += wave;
    //cutoutBlock = 1.0;
  }

  if(leaves) {
    float time2 = sin(frameTimeCounter * 1.07);
    float time = frameTimeCounter * 1.14;

    vec3 noise  = texture2D(noisetex, (position.xz + vec2(time2, time * 0.45) * 0.019) / 8.0).xyz * 4.0;
         noise += texture2D(noisetex, (position.xz - vec2(time * 0.45, time2) * 0.019) / 4.0).xyz * 2.0;
         noise *= 0.74;

    //vec3 noise2 = vec3(sin(noise.x * time2), sin(noise.y * time2), sin(noise.z * time2)) * 0.1;

    vec3 wave = vec3(sin(time + noise.x), sin(time + noise.y), sin(time + noise.z));

    position.xyz += wave * 0.021;
    //cutoutBlock = 1.0;
  }

	//if(mc_Entity.x == 0) position.xyz += 10000;

	position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;
	position.xy /= mix(1.0, length(position.xy), SHADOW_MAP_BIAS) / 0.95;
	position.z /= max(1.0, far / shadowDistance);

	gl_Position = position;
	//gl_Position.z +
}

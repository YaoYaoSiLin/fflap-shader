#version 130

#define SHADOW_MAP_BIAS 0.9

#define disable -255
#define normal_biomes 0
#define swamp 6
#define frozen_ocean_and_river 10
#define warm_ocean 44
#define lukewarm_ocean 45
#define cold_ocean 46

#define Water_Color_Test disable //[disable normal_biomes swamp frozen_ocean_and_river warm_ocean lukewarm_ocean cold_ocean]

const float shadowDistance = 140.0;

attribute vec4 mc_Entity;
attribute vec4 mc_midTexCoord;

uniform sampler2D noisetex;

uniform float frameTimeCounter;
uniform float far;

uniform mat4 gbufferModelView;

out float shadowPass;
out float isWater;
out float isLava;
out float blockDepth;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 worldNormal;
out vec3 vP;

out vec4 color;

#include "../libs/water.glsl"

#define GetBlockID(x, id) (step(id - 0.5, x) * step(x, id + 0.5) * id)

vec4 opElongate( in vec3 p, in vec3 h )
{
    //return vec4( p-clamp(p,-h,h), 0.0 ); // faster, but produces zero in the interior elongated box

    vec3 q = abs(p)-h;
    return vec4( max(q,0.0), min(max(q.x,max(q.y,q.z)),0.0) );
}

void main() {
  color = gl_Color;

	float blockID = float(mc_Entity.x);

	shadowPass = 0.0;
	isWater = 0.0;
	isLava = GetBlockID(10.0, blockID);

	blockDepth = 1.0;

	if(mc_Entity.x == 95 || mc_Entity.x == 160 || mc_Entity.x == 79 || mc_Entity.x == 20 || mc_Entity.x == 165){
		shadowPass = 1.0;
	}

	if(mc_Entity.x == 160 || mc_Entity.x == 106){
		blockDepth = 0.125;
	}

	if(mc_Entity.x == 8){
		isWater = 1.0;
		blockDepth = 0.0625;

		#if Water_Color_Test > disable
			#if Water_Color_Test == normal_biomes
				color.rgb = vec3(0.247 , 0.4627, 0.8941);
			#elif Water_Color_Test == swamp
				color.rgb = vec3(0.3803, 0.4823, 0.3921);
			#elif Water_Color_Test == frozen_ocean_and_river
				color.rgb = vec3(0.2235, 0.2196, 0.7882);
			#elif Water_Color_Test == warm_ocean
				color.rgb = vec3(0.2627, 0.8352, 0.9333);
			#elif Water_Color_Test == lukewarm_ocean
				color.rgb = vec3(0.2705, 0.6784, 0.949 );
			#elif Water_Color_Test == cold_ocean
				color.rgb = vec3(0.2392, 0.3411, 0.8392);
			#endif
		#endif

		color = CalculateWaterColor(color);
	}

	texcoord = gl_MultiTexCoord0.xy;
	lmcoord  = gl_MultiTexCoord1.xy;

	worldNormal = gl_Normal.xyz;

	bool leaves = bool(step(1799.5, blockID) * step(blockID, 1805.5));

  bool plant = mc_Entity.x == 31;

  bool double_plant_upper = mc_Entity.x == 175;
  bool double_plant_lower = mc_Entity.x == 176;
  bool double_plant = double_plant_upper || double_plant_lower;

  //bool farm = mc_Entity.x == 59 || mc_Entity.x == 141 || mc_Entity.x == 142 || mc_Entity.x == 207;

	vP = (gl_ModelViewMatrix * gl_Vertex).xyz;
	//vP.xy /= (mix(1.0, length(vP.xy), SHADOW_MAP_BIAS) / 0.95);
	//vP.z /= max(far / shadowDistance, 1.0);

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

	//if(mc_Entity.x == 35) position.xyz += 10000;


	position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;

	vec4 cylinder = opElongate(vec3(position.xy, 0.0), vec3(0.7071, 0.7071, 0.0));

	float distortion = length(position.xy);
	//position.xy /= mix(1.0, distortion, 0.7);
	position.xy /= mix(1.0, distortion, SHADOW_MAP_BIAS) / 0.95;
	//position.z *= 0.25;
	//position.z /= max(1.0, far / shadowDistance);
	//position.z = exp(128.0 * (position.z * 0.5 + 0.5)) / 3000.0;
	//position.z = exp((position.z * 0.5 + 0.5) * 15.0) / exp(15.0) * 2.0 - 1.0;

	gl_Position = position;
}

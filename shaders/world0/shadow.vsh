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

uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;
uniform mat4 shadowModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

uniform vec3 cameraPosition;

uniform float frameTimeCounter;
uniform float far;
uniform float near;

out vec4 worldPosition;

out float vertex_ID;
out float vertexBlockID;
out vec2 vertex_lmcoord;
out vec2 vertex_texcoord;
out vec3 vertex_normal;
out vec4 vertex_color;

#include "../libs/water.glsl"

#define GetBlockID(x, id) (step(id - 0.5, x) * step(x, id + 0.5) * id)

#define MaskID(id, E) (step(id - 0.5, E) * step(E, id + 0.5) * id)
#define MaskIDBetween(id, id2, E) (step(id - 0.5, E) * step(E, id + 0.5) * id + step(id2 - 0.5, E) * step(E, id2 + 0.5) * id)

void main() {
  vertex_ID = float(mc_Entity.x == 0.0);
  vertex_lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
  vertex_texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  vertex_normal = gl_Normal.xyz;
  vertex_color = gl_Color;

	float blockID = float(mc_Entity.x);
  vertexBlockID = MaskID(blockID, 8.0) + MaskID(blockID, 20) + MaskID(blockID, 106) + MaskID(blockID, 95) + MaskID(blockID, 160);

	if(mc_Entity.x == 8){
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

		vertex_color = CalculateWaterColor(vertex_color);
	}

	bool leaves = bool(step(1799.5, blockID) * step(blockID, 1805.5));

  bool plant = mc_Entity.x == 31;

  bool double_plant_upper = mc_Entity.x == 175;
  bool double_plant_lower = mc_Entity.x == 176;
  bool double_plant = double_plant_upper || double_plant_lower;

  //bool farm = mc_Entity.x == 59 || mc_Entity.x == 141 || mc_Entity.x == 142 || mc_Entity.x == 207;

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
  }

  float angle = -0.25 * 2.0 * Pi;
  mat2 rotate = mat2(cos(angle), sin(angle),-sin(angle), cos(angle));

  position = gl_ModelViewMatrix * position;

  worldPosition = shadowModelViewInverse * position;
  gl_Position = position;
}

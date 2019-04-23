#version 120

uniform sampler2D texture;

uniform vec3 shadowLightPosition;

uniform mat4 gbufferModelViewInverse;

varying float shadowPass;
varying float isWater;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 normal;

varying vec4 color;

/* DRAWBUFFERS:01 */

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
		tex = vec4(0.02, 0.02, 0.02, 0.06);
    //tex.a = 0.05;
		//tex.rgb = vec3(1.0);
	}

  //tex.rgb = mix(tex.rgb, vec3(1.0), tex.a);

	float p = 0.0;

  //if(shadowPass < 0.9) tex.a = 1.0;

	tex.rgb *= pow(clamp(dot(normal, mat3(gbufferModelViewInverse) * normalize(shadowLightPosition)) * 0.5 + 0.5, 0.0, 1.0), 0.2);

	gl_FragData[0] = tex;
	gl_FragData[1] = vec4(tex.rgb * (1.0 - tex.a), max(isWater, shadowPass));
}

#define customTexture depthtex2

uniform sampler2D customTexture;      //custom texture

#define moonSize 24.0	//[12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0]

#define starFade 16.0 //[0.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0]

float nrandY(vec2 co){
	float a = fract(cos(co.x * 8.3e-3 + co.y) * 4.7e5);
	float b = fract(sin(co.x * 0.3e-3 + co.y) * 1.0e5);

	float c = mix(a, b, 0.5);

	return c;
}

void GetStars(inout vec3 color, in vec3 wP){
  wP = normalize(wP);
  //wP.y = abs(wP.y);

  if(wP.y > 0.0){
		float time = frameTimeCounter / 1.0;

		vec2 uv = wP.xz;
			   uv /= 1.0 + (abs(wP.y));
			   uv *= 1024.0;

		vec2 p = vec2(uv / 4) + vec2(1, -1.3);
			   p += 0.2 * vec2(sin(time / 16.0), sin(time / 12.0) * 10);

		vec2 seed = p.xy * 2.0;
			 	 seed = floor(seed);
		float star = pow(nrandY(seed + vec2(0.175, -0.225)), 50.0);

    //color += (1.0 - color) * dot(max(vec3(0.0), 1.0 - color * starFade), vec3(0.2126, 0.7152, 0.0722)) * star * 0.03 * min(wP.y * 10.0, 1.0);
		color += (1.0 - color) * max(0.0, 1.0 - dot(color, vec3(0.2126, 0.7152, 0.0722)) * starFade * 5.0) * star * min(wP.y * 10.0, 1.0) * 0.03;
	}
}

vec2 GetMoonTexture(in vec3 wP, in vec3 sP, in float s){
	//sP = normalize(mat3(gbufferModelViewInverse) * -sunPosition);
	sP = -sP;

	vec3 p = normalize(wP.xyz);
			 //p = vec3(p.x, p.z, p.y);
			 //p.z = p.z * p.z;
			 p.xy = p.xy / (1.0 + p.z);
			 p.xy = (sP.xy / (1.0 + sP.z)) - p.xy;
			 p.xy = p.xy * (1.0 + p.z);
			 p.xy *= s;
			 p.xy = p.xy * 0.5 + 0.5;

	if(p.xy == clamp(p.xy, vec2(0.0), vec2(1.0)) && p.z > -0.5){

		//p = floor(p * 100.0) / 100.0;

		//return vec2(dot(p.xy * 2.0 - 1.0, sP.xy));
		return pow(texture2D(customTexture, p.xy).ba, vec2(2.2, 1.0)) * vec2(1.0, clamp(wP.y * 1.0, 0.0, 1.0));
		//return vec2(p.z * p.z, 1.0);
	}
}
/*
void GetMoonTexture(inout vec3 color, in vec3 wP, in vec3 sP, in float s){
	vec3 p = normalize(wP.xyz);
			 p = vec3(p.x, p.z, p.y);
			 p.xz = p.xz / (1.0 + p.y);

			 p.xz = (-sP.xy / (1.0 + -sP.z)) - p.xz;
			 p.xz *= 1.0 + p.y;
			 p.xz *= s;

	if(p.xz == clamp(p.xz, vec2(-1.0), vec2(1.0)) && p.y > -0.5){
		//p.z = -p.z;
		p = p * 0.5 + 0.5;

		//color = p * 0.1;
		//color += texture2D(depthtex2, p.xz).a * texture2D(depthtex2, p.xz).b * 0.5 * clamp(wP.y * 1.0, 0.0, 1.0) * moonColor;
		color = mix(color, texture2D(depthtex2, p.xz).b * moonColor, texture2D(depthtex2, p.xz).a * clamp(wP.y * 1.0, 0.0, 1.0));
		//color = vec3(pow(texture2D(depthtex2, p.xz).a, 1.0));
	}
}
*/

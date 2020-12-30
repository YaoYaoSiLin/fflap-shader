#version 130

uniform sampler2D gnormal;

uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;

uniform int frameCounter;

in vec2 texcoord;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / resolution;

//const bool gnormalMipmapEnabled = true;

void main() {
  float radius = 2.0;

  vec4 blur = vec4(0.0);

	for(float j = -radius; j <= radius; j += 1.0){
    //if(j == 0.0) continue;

    vec2 direction = vec2(0.0, j * 2.0);
    vec2 coord = round(texcoord * resolution + direction) * pixel;
    if(coord.y > 0.6 || floor(coord.x) != 0.0) continue;

    float l2 = direction.x * direction.x + direction.y * direction.y + 1e-5;
    float weight = exp(-l2 / 1.3778);

    blur += vec4(pow(texture2D(gnormal, coord).rgb, vec3(2.2)), 1.0) * weight;
  }

  /* DRAWBUFFERS:2 */
  gl_FragData[0] = vec4(pow(blur.rgb / blur.a, vec3(1.0 / 2.2)), 1.0);
}

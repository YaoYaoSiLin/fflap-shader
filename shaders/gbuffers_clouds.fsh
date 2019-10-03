#version 130

uniform sampler2D texture;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;

out vec4 color;

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 clouds = texture2D(texture, texcoord) * color;

  if(clouds.a < 0.05) discard;
  clouds.a = 1.0;

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = clouds;
  gl_FragData[1] = vec4(0.0, 1.0, 0.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normal), 1.0, 1.0);
  gl_FragData[3] = vec4(0.333, 0.02, 0.0, 1.0);
}

#version 120

uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2D specular;

uniform vec3 upPosition;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec4 color;

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

void main() {
  vec4 albedo = texture2D(texture, texcoord) * color;
  if(albedo.a < 0.001) discard;

  //if(albedo.a > 0.999){
    //albedo.rgb = pow(albedo.rgb, vec3(2.2));
    //albedo.rgb *= pow(max(lmcoord.x, lmcoord.y), 2.0);
    //albedo.rgb = pow(albedo.rgb, vec3(1.0 / 2.2));

    //albedo.rgb = pow(albedo.rgb, vec3(0.5));
  //}

  //if(albedo.a > 0.999){
  //  albedo.rgb = pow(albedo.rgb, vec3(2.2));
  //}

  //if(albedo.a < 0.001) discard;
  //albedo.a = 1.0;

  //albedo.rgb = pow(albedo.rgb, vec3(1.0 / 2.0));
  //albedo.a   = pow(1.0 - albedo.a, 1.0 / 2.0);

  vec4 speculars = texture2D(specular, texcoord);
       //speculars = mix(vec4(0.0), speculars, albedo.a);

  //vec3 n = texture2D(normals, texcoord).xyz * 2.0 - 1.0;
  //n = normalize((n) * normalize(upPosition));
  //n.xy = normalEncode(normalize(upPosition));

/*
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord.xy, 0.0, 1.0);
  gl_FragData[2] = vec4(n.xy, 0.0, 1.0);
  gl_FragData[3] = speculars;
  gl_FragData[4] = vec4(0.0);
  gl_FragData[5] = vec4(0.0);
*/

/* DRAWBUFFERS:0123 */
  gl_FragData[0] = albedo;
  gl_FragData[1] = vec4(lmcoord.xy, 0.0, 1.0);
  gl_FragData[2] = vec4(normalEncode(normalize(upPosition)), 1.0, 1.0);
  gl_FragData[3] = speculars;

  //gl_FragData[0] = albedo;      //rain layer
  //gl_FragData[1] = vec4(0.0, 0.0, gl_FragCoord.z, 1.0);



  /*
  gl_FragData[0] = texture2D(texture, texcoord) * color;
  gl_FragData[1] = vec4(lmcoord.xy, 0.0, 1.0);
  gl_FragData[2] = vec4(n.xy, 0.0, 1.0);
  gl_FragData[3] = speculars;
  gl_FragData[4] = vec4(0.0);      //rain layer
  gl_FragData[5] = vec4(0.0);

  if(albedo.a < 0.999){
  gl_FragData[4] = albedo;      //rain layer
  gl_FragData[5] = vec4(0.0, 0.0, 0.0, 1.0);
  }
  */
}

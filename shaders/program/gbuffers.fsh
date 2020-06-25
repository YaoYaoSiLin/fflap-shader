uniform sampler2D texture;

#if Normal == 1
in vec3 normal;
#elif Normal == 2
uniform sampler2D normals;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;
#endif

#if Specularity == 2
uniform sampler2D specular;
#endif

in vec2 texcoord;
in vec2 lmcoord;

in vec4 color;

void main(){
  
}

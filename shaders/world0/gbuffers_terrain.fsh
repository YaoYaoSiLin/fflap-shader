#version 130

uniform sampler2D tex;
uniform sampler2D specular;
uniform sampler2D normals;

uniform vec3 lightVectorView;

uniform vec2 resolution;

uniform ivec2 atlasSize;

in vec2 tileCoord0;
in vec2 tileCoord1;
in vec2 tileCoord2;
in vec2 tileCenter;

in float material_id;

in float solid;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec3 viewVector;
in vec3 lightVector;

in vec4 color;

#define Enabled_Parallax_Self_Shadow
#define Parallax_Mapping_Depth 1.0         //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]
#define Parallax_Mapping_Distance 32.0

//#define Tile_Resolution_Auto_Detect
#define Tile_Resolution 16      //[2 4 8 16 32 64 128 256 512 1024 2048 4096 8192]

vec2 OffsetCoord(in vec2 coord, in vec2 offset, in vec2 size){
	vec2 offsetCoord = coord + mod(offset.xy, size);

	vec2 minCoord = vec2(coord.x - mod(coord.x, size.x), coord.y - mod(coord.y, size.y));
	vec2 maxCoord = minCoord + size;

    vec2 texelSize = vec2(1.0) / vec2(atlasSize);
    
    if(offsetCoord.x < minCoord.x){
        offsetCoord.x += size.x;
    }else if(maxCoord.x < offsetCoord.x){
        offsetCoord.x -= size.x;
    }

    if(offsetCoord.y < minCoord.y){
        offsetCoord.y += size.y;
    }else if(maxCoord.y < offsetCoord.y){
        offsetCoord.y -= size.y;
    }
    
    /*
    offsetCoord.x += offsetCoord.x < minCoord.x ? size.x : 0.0;
    offsetCoord.x -= maxCoord.x < offsetCoord.x ? size.x : 0.0;
    offsetCoord.y += offsetCoord.y < minCoord.y ? size.y : 0.0;
    offsetCoord.y -= maxCoord.y < offsetCoord.y ? size.y : 0.0;
    */
    /*
    offsetCoord.x += size.x * step(offsetCoord.x, minCoord.x);
    offsetCoord.x -= size.x * step(maxCoord.x, offsetCoord.x);
    offsetCoord.y += size.y * step(offsetCoord.y, minCoord.y);
    offsetCoord.y -= size.y * step(maxCoord.y, offsetCoord.y);
    */
	return offsetCoord;
}

vec2 OffsetCoord(in vec2 coord, in vec2 offset, in ivec3 size){
    vec2 tileSize = vec2(size.z) / vec2(size.xy);

	return OffsetCoord(coord, offset, tileSize);
}

vec2 dx = dFdx(texcoord);
vec2 dy = dFdy(texcoord);
float mipmap_level = (log2(max(abs(dx.x) + abs(dx.y), abs(dy.x) + abs(dy.y)) * 200.0 * 2.0)); 

//float lod_level = length(viewVector) * min(resolution.x, resolution.y) / Tile_Resolution / 512.0;

vec4 _texture(in sampler2D sampler, in vec2 coord) {
    //return texture2D(sampler, coord);

    //vec2 atla = vec2(atlasSize) / min(exp2(1.0 + floor(lod_level)), Tile_Resolution / 16.0);

    //coord = floor(coord * atla) / atla;

    return texture2DLod(sampler, coord, mipmap_level);
}

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

#include "/libs/common.inc"

float GetHeightMap(in vec2 coord){
    return _texture(normals, coord).a - 1.0;
    //return texture_(tex, coord).a - 1.0;
}

vec2 ParallaxMapping(in vec2 coord, in vec3 direction, in vec2 resolution, in float dist, inout float parallaxDepth){
    int steps = 16;
    float invsteps = 1.0 / float(steps);

    float fading = saturate((-dist + Parallax_Mapping_Distance - 1.0) / 8.0);

    //invsteps = mix(invsteps * 4.0, invsteps, fading);

    float alpha = GetHeightMap(coord);

    vec2 atla = atlasSize;

    vec2 delta = direction.xy / atla;
         delta *= (1.0 / direction.z) * max(resolution.x, resolution.y) / 16.0 * invsteps * fading * 4.0;

    float layer = -invsteps;
    float layerHeight = layer;

    float height = 0.0;

    if(alpha > -0.001 || dist > Parallax_Mapping_Distance || min(atla.x, atla.y) < 1.0) 
    return coord;
        
    for(int i = 0; i < steps; i++){
        height = GetHeightMap(coord);
        if(layerHeight < height) break;

        coord = OffsetCoord(coord, delta * -1.0, min(resolution.x, resolution.y) / atla);

        layerHeight += layer;
    }
    
    //parallaxDepth = (layerHeight - layer) * (-height) + 1.0;
    parallaxDepth = (layerHeight - layer) + 1.0;

    return coord;
}

float ParallaxSelfShadow(in vec2 coord, in vec3 direction, in vec2 resolution, float dist, float parallaxDepth) {
    int steps = 12;
    float invsteps = 1.0 / float(steps);

    vec2 atla = vec2(atlasSize);

    vec2 delta = direction.xy / atla * max(resolution.x, resolution.y) * (invsteps * 0.25 * 2.0);
    
    float stepLength = -invsteps;
    float layerHeight = 0.0;

    vec2 mipmapAtla = atla / clamp(floor(mipmap_level) * 2.0, 1.0, 4.0);

    coord = floor(coord * mipmapAtla) / mipmapAtla;

    float height = GetHeightMap(coord);
    //height = min(parallaxDepth - 1.0, height);
    height -= stepLength * 0.25;

    if(height > -0.001 || dist > Parallax_Mapping_Distance || min(float(atlasSize.x), float(atlasSize.y)) < 1.0) 
    return 1.0;

    for(int i = 0; i < steps; i++) {
        coord = OffsetCoord(coord, delta, min(resolution.x, resolution.y) / atla);
        float sampleHeight = GetHeightMap(floor(coord * mipmapAtla) / mipmapAtla);

        if(layerHeight < height) break;
        if(height < sampleHeight) return 0.0;

        layerHeight += stepLength;
    }

    return 1.0;
}

void main() {
    vec2 coord = texcoord;

    mat3 tbn = mat3(tangent, binormal, normal);

    vec2 tileResolution = vec2(Tile_Resolution);
    vec2 tileSizeAuto = vec2(round(abs(tileCenter.x - min(tileCoord0.x, min(tileCoord1.x, tileCoord2.x))) * 2.0), 
                             round(abs(tileCenter.y - min(tileCoord0.y, min(tileCoord1.y, tileCoord2.y))) * 2.0));

    //float p0 = length(tileCoord0 - tileCoord1);
    //float p1 = length(tileCoord0 - tileCoord1);
    //vec2 tileSizeAuto = vec2(round(p0), round(p1));

    #ifdef Tile_Resolution_Auto_Detect
        tileResolution = tileSizeAuto;
    #endif

    float heightmap = 1.0;
    float parallaxSelfShadow = 1.0;

    coord = ParallaxMapping(coord, normalize(viewVector * tbn), tileResolution, length(viewVector), heightmap);
    parallaxSelfShadow = ParallaxSelfShadow(coord, normalize(lightVector * tbn), tileResolution, length(viewVector), heightmap);

    vec4 albedo = _texture(tex, coord) * color;
         albedo.a = albedo.a < 0.2 ? 0.0 : 1.0;
         //albedo.a = 1.0;

    vec4 specular0 = _texture(specular, coord);
         specular0.a = texture2DLod(specular, coord, 0).a;

    float material = floor(specular0.b * 255.0);
    float emissive = specular0.a;
    float materialAO = _texture(normals, coord).z;

    vec3 geometryNormal = normal;

    vec3 normal_texture = _texture(normals, coord).xyz * 2.0 - 1.0;

    normal_texture.xy *= 2.0;
    normal_texture.z = 1.0 - sqrt(dot(normal_texture.xy, normal_texture.xy));

    vec3 texturedNormal = normalize(tbn * normalize(normal_texture));

    float tileMaterial = round(material_id) / 255.0;

    //if(dot(-normalize(viewVector), normal) < 1e-5) discard;

    if(!gl_FrontFacing) {
        geometryNormal = -geometryNormal;
        texturedNormal = -texturedNormal;
        //discard;
    }

    gl_FragData[0] = albedo;
    gl_FragData[1] = vec4(pack2x8(min(vec2(1.0), lmcoord.xy * (15.0 / 14.0))), parallaxSelfShadow, pack2x8(vec2(emissive, materialAO)), solid);
    gl_FragData[2] = vec4(normalEncode(geometryNormal), tileMaterial, 1.0);
    gl_FragData[3] = vec4(normalEncode(texturedNormal), pack2x8(specular0.rg), material / 255.0);
}
/* DRAWBUFFERS:0123 */
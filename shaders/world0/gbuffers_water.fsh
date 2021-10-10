#version 130

uniform sampler2D tex;
uniform sampler2D normals;
uniform sampler2D specular;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

in float material_id;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec4 color;

vec2 normalEncode(vec3 n) {
    vec2 enc = normalize(n.xy) * (sqrt(-n.z*0.5+0.5));
    enc = enc*0.5+0.5;
    return enc;
}

#include "/libs/common.inc"

void main() {
    float mask = round(material_id);
    bool isWater      = CalculateMaskID(8.0, mask);
    bool isIce        = CalculateMaskID(79.0, mask);
    bool isEwww       = CalculateMaskID(165.0, mask);

    bool isGlass      = CalculateMaskID(20.0, mask);
    bool isGlassPane = CalculateMaskID(102.0, mask);
    bool isStainedGlass = CalculateMaskID(95.0, mask);
    bool isStainedGlassPane = CalculateMaskID(160.0, mask);
    bool AnyGlass = isGlass || isGlassPane || isStainedGlass || isStainedGlassPane;
    bool AnyClearGlass = isGlass || isGlassPane;
    bool AnyStainedGlass = isStainedGlass || isStainedGlassPane;
    bool AnyGlassBlock = isGlass || isStainedGlass;
    bool AnyGlassPane = isGlassPane || isStainedGlassPane;

    if(!gl_FrontFacing && isWater) discard;

    if(!gl_FrontFacing) {
        gl_FragDepth = gl_FragCoord.z + 1e-5;

        gl_FragData[4] = vec4(gl_FragCoord.z, normalEncode(-normal), 1.0);

        gl_FragData[0] = vec4(0.0);
        gl_FragData[1] = vec4(0.0);
        gl_FragData[2] = vec4(0.0);
        gl_FragData[3] = vec4(0.0);

        return;
    }else{
        gl_FragDepth = gl_FragCoord.z;

        gl_FragData[4] = vec4(0.0);
    }

    vec4 albedo = texture2D(tex, texcoord) * color;

    vec3 normal_texture = texture2D(normals, texcoord).rgb * 2.0 - 1.0;
    mat3 tbn = mat3(tangent, binormal, normal);

    vec4 speculars = texture2D(specular, texcoord);
    bool missingSpecular = maxComponent(speculars.rgb) < 1e-5;

    float smoothness = clamp(speculars.r, 0.001, 0.999);
    float metallic = max(0.02, speculars.g);
    float material = floor(speculars.b * 255.0);
    float absorption = 0.0;
    float ior = 1.333;

    if(isWater) {
        material = 230.0;
        absorption = 0.5;

        smoothness = 0.995;
        metallic = 0.02;

        albedo = color;
    }else
    if(AnyGlass) {
        ior = 1.52;

        material = 255.0;
        absorption = AnyGlassPane ? 4.0 : 2.0;

        if(missingSpecular) {
            metallic = 0.04;
            smoothness = mix(0.99, 0.1, rescale(0.99, 1.0, albedo.a));
        }
    }

    vec3 texturedNormal = normalize(tbn * normal_texture);

    gl_FragData[0] = vec4(albedo.rgb, 1.0);
    gl_FragData[1] = vec4(pack2x8(lmcoord), 1.0, albedo.a, 1.0);
    gl_FragData[2] = vec4(absorption / 255.0, material / 255.0, mask / 255.0, 1.0);
    gl_FragData[3] = vec4(normalEncode(texturedNormal), pack2x8(vec2(smoothness, metallic)), 1.0);
}
/* DRAWBUFFERS:01234 */
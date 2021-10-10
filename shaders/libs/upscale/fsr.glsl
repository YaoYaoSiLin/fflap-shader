// Filtering for a given tap for the scalar.
void FsrEasuTapF(inout vec3 accumulaction, inout float total, vec2 offset, vec2 direction, vec2 len, float lob, float clp, vec3 color) {
    vec2 v = vec2(offset.x * direction.x + offset.y * direction.y, -offset.x * direction.y + offset.y * direction.x);
         v*= len;

    float d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);

    float weightB = pow2((2.0 / 5.0) * d2 + (-1.0));
    float weightA = pow2(lob * d2 + (-1.0));

    weightB = (25.0 / 16.0) * weightB + (-(25.0 / 16.0 - 1.0));
    float weight = weightB * weightA;

    accumulaction += color * weight;
    total += weight;
}

// Accumulate direction and length.
void FsrEasuSetF(inout vec2 direction, inout vec2 len, vec2 pp, bool biS, bool biT, bool biU, bool biV, float lA, float lB, float lC, float lD, float lE){
    float weight = 0.0;

    if(biS) weight = (1.0 - pp.x) * (1.0 - pp.y);
    if(biT) weight = (      pp.x) * (1.0 - pp.y);
    if(biU) weight = (1.0 - pp.x) * (      pp.y);
    if(biV) weight = (      pp.x) * (      pp.y);

    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = pow2(inversesqrt(max(abs(dc), abs(cb))));
    float dirX = lD - lB;
    direction.x += dirX * weight;
    lenX = pow2(saturate(abs(direction.x) * lenX));
    len += lenX * weight;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = pow2(inversesqrt(max(abs(ec), abs(ca))));
    float dirY = lE - lA;
    direction.y += dirY * weight;
    lenY = pow2(saturate(abs(direction.y) * lenY));
    len += lenY * weight;
}

void FsrEasuF(out vec3 pix, in vec2 position, vec4 con0, vec4 con1, vec4 con2, vec4 con3){
    ivec2 ip = ivec2(position * resolution);

    vec2 pp = vec2(ip) * con0.xy + con0.zw;
    vec2 fp = floor(pp);
    pp -= fp;

    vec2 p0 = fp * con1.xy + con1.zw;
}

/*
vec2 inputScale = resolution * 0.5;
vec2 outputScale = resolution;

vec4 con1 = vec4(1.0 / inputScale, 1.0 * (1.0 / inputScale.x), -1.0 * (1.0 / inputScale.y))
*/
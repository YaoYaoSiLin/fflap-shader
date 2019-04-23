#define Enabled_TAA

#if Taa_Support == 1 && defined(Enabled_TAA)
uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;

vec2 resolution = vec2(viewWidth, viewHeight);
vec2 pixel = 1.0 / vec2(viewWidth, viewHeight);
#endif

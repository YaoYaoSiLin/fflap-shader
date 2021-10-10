vec3 ReinhardTonemap(in vec3 x) {
    return color / (color + 1.0);
}

vec3 InverseReinhardTonemap(in vec3 x) {
    return -color / (color - 1.0);
}

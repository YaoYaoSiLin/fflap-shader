#define Enabled_TAA

#define Included_jittering

//
const vec2 haltonSequence_2n3[16] = vec2[16](vec2(0.5    , 0.33333),
                                             vec2(0.25   , 0.66666),
                                             vec2(0.75   , 0.11111),
                                             vec2(0.125  , 0.44444),
                                             vec2(0.625  , 0.77777),
                                             vec2(0.375  , 0.22222),
                                             vec2(0.875  , 0.55555),
                                             vec2(0.0625 , 0.88888),
                                             vec2(0.5625 , 0.03703),
                                             vec2(0.3125 , 0.37037),
                                             vec2(0.8125 , 0.7037 ),
                                             vec2(0.1875 , 0.14814),
                                             vec2(0.6875 , 0.48148),
                                             vec2(0.4375 , 0.81481),
                                             vec2(0.9375 , 0.25925),
                                             vec2(0.03125, 0.59259)
                                           );

//The Unreasonable Effectiveness of Quasirandom Sequences | Extreme Learning
const vec2 R2sq2[16] = vec2[16](vec2(0.2548776662466927, 0.06984029099805333),
                                vec2(0.009755332493385449, 0.6396805819961064),
                                vec2(0.764632998740078, 0.20952087299415956),
                                vec2(0.5195106649867709, 0.7793611639922129),
                                vec2(0.27438833123346384, 0.3492014549902662),
                                vec2(0.029265997480155903, 0.9190417459883191),
                                vec2(0.7841436637268488, 0.48888203698637245),
                                vec2(0.5390213299735418, 0.05872232798442578),
                                vec2(0.29389899622023474, 0.6285626189824791),
                                vec2(0.04877666246692769, 0.19840290998053245),
                                vec2(0.8036543287136197, 0.7682432009785858),
                                vec2(0.5585319949603118, 0.33808349197663823),
                                vec2(0.31340966120700564, 0.9079237829746916),
                                vec2(0.0682873274536977, 0.4777640739727449),
                                vec2(0.8231649937003915, 0.04760436497079823),
                                vec2(0.5780426599470836, 0.6174446559688516)
                                );

#define halton 0
#define R2 1

#define Jitter_Mode R2 //[halton R2]

#if Jitter_Mode == halton
#define jittering (haltonSequence_2n3[int(mod(frameCounter, 16))])
#elif Jitter_Mode == R2
#define jittering (R2sq2[int(mod(frameCounter, 16))])
#endif

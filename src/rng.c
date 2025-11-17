#include <stdlib.h>
#include <math.h>

#include "rng.h"

void rng_seed(unsigned int s) {
    srand(s);
}

float rand_uniform(float low, float high) {
    return ((float)rand() / (float)RAND_MAX) * (high - low) + low;
};

float rand_normal(float mean, float std) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    float z  = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + std * z;
}

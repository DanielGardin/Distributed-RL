#pragma once

#include "policy.h"
#include "utils.h"

void policy_gradient(
    Policy *policy,
    ExperienceBuffer *buffer,
    float gamma,
    float *baseline,
    MLPCache *cache
);

void mean_baseline(ExperienceBuffer *buffer, int gamma, float *baseline);

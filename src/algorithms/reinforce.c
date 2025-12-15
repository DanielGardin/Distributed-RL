#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "algorithms/utils.h"
#include "algorithms/reinforce.h"

void policy_gradient(
    Policy *policy,
    ExperienceBuffer *buffer,
    float gamma,
    float *baseline,
    MLPCache *cache
) {
    int size = buffer->size;
    int out_size = policy->mlp->output_size;

    float *returns = malloc(buffer->size * sizeof(float));
    discounted_cumsum(buffer, gamma, returns);
    
    float *logits = malloc(size * out_size * sizeof(float));
    mlp_forward(policy->mlp, buffer->observations, size, logits, cache);

    float *dlogp = malloc(size * out_size * sizeof(float));
    policy_log_prob_from_logits(policy, logits, buffer->actions, size, NULL, dlogp);

    for (int t = 0; t < size; t++) {
        float advantage = returns[t];
        if (baseline) advantage -= baseline[t];

        for (int j = 0; j < out_size; j++)
            dlogp[t * out_size + j] *= -advantage;
    }

    mlp_backward(policy->mlp, cache, dlogp, NULL);

    free(returns);
    free(logits);
    free(dlogp);
}

void mean_baseline(ExperienceBuffer *buffer, int gamma, float *baseline) {
    float mean_R = mean_return(buffer);

    for (int t = 0; t < buffer->size; t++) baseline[t] = mean_R;
}

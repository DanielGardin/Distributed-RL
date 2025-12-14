#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "algorithms/policy.h"
#include "rng.h"

void policy_log_prob(
    const Policy *policy,
    const float *obs,
    const float *actions,
    int batch_size,
    float *log_prob,
    float *grad_out,
    MLPCache *cache
) {
    float *logits = malloc(batch_size * policy->mlp->layers[policy->mlp->num_layers-1].output_size * sizeof(float));
    mlp_forward(policy->mlp, obs, batch_size, logits, cache);
    policy_log_prob_from_logits(policy, logits, actions, batch_size, log_prob, grad_out);
    free(logits);
}

float log_sum_exp(float *array, int size) {
    float max_value = array[0];
    for (int i = 1; i < size; i++)
        if (array[i] > max_value) max_value = array[i];

    float sum_exp = 0.0f;
    for (int i = 1; i < size; i++)
        sum_exp += expf(array[i] - max_value);
    
    return max_value + logf(sum_exp);
}

/***************************
 * Discrete Policy methods *
 ***************************/

// void sample_discrete_action(const MLP *mlp, const float *obs, int batch_size, float *action, float *log_prob) {
//     int n_actions = mlp->layers[mlp->num_layers-1].output_size;

//     float *logits = malloc(n_actions * sizeof(float));
//     mlp_forward(mlp, obs, batch_size, logits, NULL);

//     float lsexp = log_sum_exp(logits, n_actions);
//     float cum_prob = 0;
//     int act = 0;
//     for (; act < n_actions; act++) {
//         cum_prob += expf(logits[act] - lsexp);
    
//         if (rand_uniform(0, 1) < cum_prob)
//             break;
//     }

//     *action = (float)act;
//     if (log_prob) *log_prob = log_prob[act] - lsexp;

//     free(logits);
// }

// void discrete_log_prob (const MLP *mlp, const float *obs, int batch_size, const float *action, float *log_prob, float *grad_out) {

// }

// Policy create_discrete_policy(MLP *mlp) {
//     return (Policy) {
//         .mlp = mlp,
//         .sample = sample_discrete_action,
//         .log_prob = discrete_log_prob
//     };
// }

/***************************
 *  Binary Policy methods  *
 ***************************/

void binary_log_prob(
        const float *logits,
        const float *actions,
        int batch_size,
        float *log_prob,
        float *grad_out
) {
    for (int b = 0; b < batch_size; b++) {
        float logit = logits[b];
        int act = (int)actions[b];
        
        float neg_logit = -logit;
        float exp_neg = expf(neg_logit);
        float p_one = 1.0f / (1.0f + exp_neg);
        
        if (log_prob)
            log_prob[b] = act ? -logf(1.0f + exp_neg) 
                               : neg_logit - logf(1.0f + exp_neg);

        if (grad_out)
            grad_out[b] = (float)act - p_one;
    }
}

void binary_entropy(
    const float *logits,
    int batch_size,
    float *entropy,
    float *grad_out
) {
    for (int b = 0; b < batch_size; b++) {
        float z = logits[b];
        float p = 1.0f / (1.0f + expf(-z));

        if (entropy) entropy[b] = logf(1.0f + expf(-z)) + (1.0f - p) * z;

        if (grad_out) grad_out[b] = -z * p * (1.0f - p);
    }
}

void sample_binary_action(
    const MLP *mlp,
    const float *obs,
    int batch_size,
    float *actions,
    float *log_prob,
    float *entropy
) {
    float *logits = malloc(batch_size * sizeof(float));

    mlp_forward(mlp, obs, batch_size, logits, NULL);

    for (int b = 0; b < batch_size; b++) {
        float p_one = 1.0f / (1.0f + expf(-logits[b]));
        actions[b] = (rand_uniform(0, 1) < p_one) ? 1.0f : 0.0f;
    }

    if (log_prob) binary_log_prob(logits, actions, batch_size, log_prob, NULL);
    if (entropy) binary_entropy(logits, batch_size, entropy, NULL);
}

Policy create_binary_policy(MLP *mlp) {
    if (mlp->output_size != 1)
        fprintf(stderr,
            "WARNING: Expected a single probability output from the neural network, got size %d.",
            mlp->output_size
        );

    return (Policy) {
        .mlp = mlp,
        .sample = sample_binary_action,
        .log_prob = binary_log_prob,
        .entropy = binary_entropy
    };
}

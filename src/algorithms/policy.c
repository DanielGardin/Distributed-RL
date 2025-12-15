#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "algorithms/policy.h"
#include "rng.h"

void policy_sample_action(
    const Policy *policy,
    const float *obs,
    int batch_size,
    float *actions
) {
    float *logits = malloc(batch_size * policy->mlp->output_size * sizeof(float));
    mlp_forward(policy->mlp, obs, batch_size, logits, NULL);
    policy_sample_action_from_logits(policy, logits, batch_size, actions);
    free(logits);
}

void policy_log_prob(
    const Policy *policy,
    const float *obs,
    const float *actions,
    int batch_size,
    float *log_prob,
    float *grad_out,
    MLPCache *cache
) {
    float *logits = malloc(batch_size * policy->mlp->output_size * sizeof(float));
    mlp_forward(policy->mlp, obs, batch_size, logits, cache);
    policy_log_prob_from_logits(policy, logits, actions, batch_size, log_prob, grad_out);
    free(logits);
}

float log_sum_exp(const float *array, int size) {
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

void sample_discrete_action(
    const Policy *policy,
    const float *logits,
    int batch_size,
    float *actions
) {
    int n_actions = policy->n_actions;
    float cum_prob, lsexp, u;
    int act;

    for (int b = 0; b < batch_size; b++) {
        lsexp = log_sum_exp(logits, n_actions);
        u = rand_uniform(0, 1);

        cum_prob = 0;
        for (act = 0; act < n_actions - 1; act++) {
            cum_prob += expf(logits[act] - lsexp);
            if (u < cum_prob) break;
        }

        actions[b] = (float)act;
    }
}

void discrete_log_prob (
    const struct Policy *policy,
    const float *logits,
    const float *actions,
    int batch_size,
    float *log_prob,
    float *grad_out
) {
    int n_actions = policy->n_actions;
    
    for (int b = 0; b < batch_size; b++) {
        int action = (int)actions[b];
        
        float lse = log_sum_exp(logits, n_actions);
        log_prob[b] = logits[action] - lse;
        
        if (grad_out) {
            for (int a = 0; a < n_actions; a++) {
                float prob = expf(logits[a] - lse);
                grad_out[a] = (a == action) ? (1.0f - prob) : -prob;
            }
            grad_out += n_actions;
        }
        
        logits += n_actions;
    }
}

void discrete_entropy(
    const Policy *policy,
    const float *logits,
    int batch_size,
    float *entropy,
    float *grad_out
) {
    int n_actions = policy->n_actions;
    
    for (int b = 0; b < batch_size; b++) {
        float lse = log_sum_exp(logits, n_actions);
        
        float H = 0.0f;
        if (entropy || grad_out) {
            for (int a = 0; a < n_actions; a++) {
                float log_p = logits[a] - lse;
                float p = expf(log_p);
                H -= p * log_p;
            }
            if (entropy) entropy[b] = H;
        }
        
        if (grad_out) {
            for (int a = 0; a < n_actions; a++) {
                float log_p = logits[a] - lse;
                float p = expf(log_p);
                grad_out[a] = -p * (1.0f + log_p + H);
            }
            grad_out += n_actions;
        }
        
        logits += n_actions;
    }
}

Policy create_discrete_policy(MLP *mlp, int n_actions) {
    if (mlp->output_size != n_actions)
        fprintf(stderr,
            "WARNING: Expected an output of size %d from the neural network, got size %d.",
            n_actions, mlp->output_size
        );

    return (Policy) {
        .mlp = mlp,
        .act_size=1,
        .n_actions=n_actions,
        .sample = sample_discrete_action,
        .log_prob = discrete_log_prob
    };
}

/***************************
 *  Binary Policy methods  *
 ***************************/

void sample_binary_action(
    const Policy *policy,
    const float *logits,
    int batch_size,
    float *actions
) {
    for (int b = 0; b < batch_size; b++) {
        float p_one = 1.0f / (1.0f + expf(-logits[b]));
        actions[b] = (rand_uniform(0, 1) < p_one) ? 1.0f : 0.0f;
    }
}

void binary_log_prob(
    const Policy *policy,
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
    const Policy *policy,
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

Policy create_binary_policy(MLP *mlp) {
    if (mlp->output_size != 1)
        fprintf(stderr,
            "WARNING: Expected a single probability output from the neural network, got size %d.",
            mlp->output_size
        );

    return (Policy) {
        .mlp = mlp,
        .act_size=1,
        .n_actions=2,
        .sample = sample_binary_action,
        .log_prob = binary_log_prob,
        .entropy = binary_entropy
    };
}

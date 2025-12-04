#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "algorithms/common.h"
#include "rng.h"

float log_sum_exp(float *array, int size) {
    float max_value = array[0];
    for (int i = 1; i < size; i++)
        if (array[i] > max_value) max_value = array[i];

    float sum_exp = 0.0f;
    for (int i = 1; i < size; i++)
        sum_exp += expf(array[i] - max_value);
    
    return max_value + logf(sum_exp);
}

void policy_sample_action(const Policy *policy, const float *obs, int batch_size, float *action, float *log_prob) {
    policy->sample(policy->mlp, obs, batch_size, action, log_prob);
}

void policy_log_prob_from_logits(
    const Policy *policy,
    const float *logits,
    const float *actions,
    int batch_size,
    float *log_prob,      // Can be NULL
    float *grad_out       // Can be NULL
) {
    policy->log_prob(logits, actions, batch_size, log_prob, grad_out);
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
    float *logits = malloc(batch_size * policy->mlp->layers[policy->mlp->num_layers-1].output_size * sizeof(float));
    mlp_forward(policy->mlp, obs, batch_size, logits, cache);
    policy_log_prob_from_logits(policy, logits, actions, batch_size, log_prob, grad_out);
    free(logits);
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

void sample_binary_action(const MLP *mlp, const float *obs, int batch_size, float *action, float *log_prob) {
    mlp_forward(mlp, obs, batch_size, action, NULL);

    for (int b = 0; b < batch_size; b++) {
        float p_one = 1.0f / (1.0f + expf(-action[b]));
        action[b] = (rand_uniform(0, 1) < p_one) ? 1.0f : 0.0f;

        if (log_prob) log_prob[b] = action[b] ? logf(p_one) : logf(1.0f - p_one);
    }
}

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
        
        if (log_prob) {
            log_prob[b] = act ? -logf(1.0f + exp_neg) 
                               : neg_logit - logf(1.0f + exp_neg);
        }
        
        if (grad_out) {
            grad_out[b] = act ? (1.0f - p_one) : -p_one;
        }
    }
}

Policy create_binary_policy(MLP *mlp) {
    return (Policy) {
        .mlp = mlp,
        .sample = sample_binary_action,
        .log_prob = binary_log_prob
    };
}

/***************************
 *    Algorithms' utils    *
 ***************************/

int policy_rollout(
    Env *env,
    const Policy *policy,
    int n_steps,
    float *observations,
    float *actions,
    float *rewards,
    bool *dones
) {
    float *log_probs = malloc(env->act_size * sizeof(float));

    float *cur_obs = observations;
    float *cur_act = actions;
    float *cur_rew = rewards;
    bool *cur_done = dones;

    env_reset(env, cur_obs);

    int step_count = 0;
    for (; step_count < n_steps; step_count++) {
        policy_sample_action(policy, cur_obs, 1, cur_act, NULL);

        cur_obs += env->obs_size;
        if (step_count == n_steps-1) cur_obs = NULL;

        env_step(env, cur_act, cur_obs, cur_rew, cur_done);

        if (*cur_done) break;
        cur_act += env->act_size;
        cur_rew += 1;
        cur_done += 1;
    };

    free(log_probs);

    return step_count;
}

void discounted_cumsum_inplace(float *r, int T, float gamma) {
    float running = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        running = r[t] + gamma * running;
        r[t] = running;
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <locale.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include <mpi.h>
#include "raylib.h"

#include "algorithms/reinforce.h"
#include "environments/cartpole.h"
#include "distributed/comm.h"
#include "nn/optimizers.h"
#include "nn/linear.h"
#include "nn/debug.h"
#include "rng.h"
#include "metrics.h"

#define WIDTH 600
#define HEIGHT 200

typedef struct {
    int seed;
    int hidden_size;
    int episodes;
    int max_steps;
    float gamma;
    int grad_steps;
    float learning_rate;
    bool render;
    char *env_name;
    char *output_dir;
} Config;

// Default values
#define DEFAULT_SEED 1
#define DEFAULT_HIDDENSIZE 16
#define DEFAULT_EPISODES 1
#define DEFAULT_MAX_STEPS 500
#define DEFAULT_GAMMA 0.99f
#define DEFAULT_GRAD_STEPS 2500
#define DEFAULT_LEARNING_RATE 1e-2f

void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [Environment] [options]\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  Environment: positional name (Default: cartpole)\n");
    fprintf(stderr, "  -s <int>   RNG seed (Default: %d)\n", DEFAULT_SEED);
    fprintf(stderr, "  -n <float> Neural network's hidden size (Default: %d)\n", DEFAULT_HIDDENSIZE);
    fprintf(stderr, "  -e <int>   Episodes per gradient step (batch size) (Default: %d)\n", DEFAULT_EPISODES);
    fprintf(stderr, "  -m <int>   Max steps per episode (Default: %d)\n", DEFAULT_MAX_STEPS);
    fprintf(stderr, "  -y <float> Discount factor (gamma) (Default: %.2f)\n", DEFAULT_GAMMA);
    fprintf(stderr, "  -k <float> Number of gradient steps to perform (Default: %d)\n", DEFAULT_GRAD_STEPS);
    fprintf(stderr, "  -l <float> Learning rate (Default: %.0e)\n", DEFAULT_LEARNING_RATE);
    fprintf(stderr, "  -o <path>  Output directory for CSV files (Default: disabled)\n");
    fprintf(stderr, "  -r         Render episode using trained policy\n");
    fprintf(stderr, "  -h         Print this help message\n");
}

void parse_arguments(int argc, char *argv[], Config *config) {
    int opt;

    // Set default values first
    config->seed = DEFAULT_SEED;
    config->hidden_size = DEFAULT_HIDDENSIZE;
    config->episodes = DEFAULT_EPISODES;
    config->max_steps = DEFAULT_MAX_STEPS;
    config->gamma = DEFAULT_GAMMA;
    config->grad_steps = DEFAULT_GRAD_STEPS;
    config->learning_rate = DEFAULT_LEARNING_RATE;
    config->env_name = "cartpole";
    config->output_dir = NULL;

    // Use "s:g:n:e:m:y:k:rl:o:h" to specify options that take an argument
    while ((opt = getopt(argc, argv, "s:g:n:e:m:y:k:rl:o:h")) != -1) {
        switch (opt) {
            case 's':
                config->seed = atoi(optarg);
                break;
            case 'n':
                config->hidden_size = atoi(optarg);
                break;
            case 'e':
                config->episodes = atoi(optarg);
                break;
            case 'm':
                config->max_steps = atoi(optarg);
                break;
            case 'y':
                config->gamma = atof(optarg);
                break;
            case 'k':
                config->grad_steps = atoi(optarg);
                break;
            case 'l':
                config->learning_rate = atof(optarg);
                break;
            case 'o':
                config->output_dir = optarg;
                break;
            case 'r':
                config->render = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            case '?':
                fprintf(stderr, "Unknown option or missing argument.\n");
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (optind < argc) {
        config->env_name = argv[optind];
    }
}

void render_episode(Env *env, Policy *policy);
void print_training_summary(TrainingMetrics *metrics, MPIContext *mpi_ctx, Config *config);

static int mkdir_p(const char *path) {
    if (path == NULL || *path == '\0') return -1;

    char tmp[512];
    size_t len = strnlen(path, sizeof(tmp) - 1);
    if (len == 0 || len >= sizeof(tmp)) return -1;
    strncpy(tmp, path, sizeof(tmp));
    tmp[sizeof(tmp) - 1] = '\0';

    // Remove trailing slashes
    while (len > 1 && tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
        len--;
    }

    // Iterate and create each component
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

Env dispatch_environment(char *env_name) {
    if (!env_name || strcmp(env_name, "cartpole") == 0)
        return make_cartpole_env(10.0f, false);

    fprintf(stderr, "ERROR: Could not find environment matching %s\n", env_name);
    exit(1);
}

Policy dispatch_policy(Env *env, int hidden_size) {
    static MLP policynet;
    
    Activation activations[2] = {relu, identity};
    int input_size[2] = {env->obs_size, hidden_size};
    int output_size = 1;

    if (env->act_size == 1) {
        int act_space = *env->act_space;
        output_size = (act_space == 2) ? 1 : act_space;
    }

    policynet = create_mlp(
        input_size,
        output_size,
        2,
        activations
    );
    kaiming_mlp_init(&policynet);

    if (env->act_size == 1) {
        int act_space = *env->act_space;

        if (act_space == 2) return create_binary_policy(&policynet);
        if (act_space > 2) return create_discrete_policy(&policynet, act_space);
    }

    fprintf(stderr, "ERROR: Could not initialize a policy for environment %s", env->name);
    exit(1);
}

int main(int argc, char *argv[]) {
    MPIContext mpi_ctx = mpi_init_context(&argc, &argv);
    double init_start = get_time();

    Config config = {0};
    parse_arguments(argc, argv, &config);

    rng_seed(config.seed + mpi_ctx.rank);

    Env env = dispatch_environment(config.env_name);
    Policy policy = dispatch_policy(&env, config.hidden_size);
    
    AdamState optimizer_state = create_adam_state(policy.mlp);
    ExperienceBuffer buffer = create_buffer(config.max_steps, env.obs_size, env.act_size);
    MLPCache cache = create_mlp_cache(policy.mlp, config.max_steps);
    
    TrainingMetrics metrics = create_metrics(config.grad_steps, config.episodes);

    int capacity = buffer.capacity;
    int out_size = policy.mlp->output_size;

    float *returns = malloc(capacity * sizeof(float));
    float *logits = malloc(capacity * out_size * sizeof(float));
    float *logp = malloc(capacity * sizeof(float));
    float *dlogp = malloc(capacity * out_size * sizeof(float));


    double training_start = get_time();
    for (int grad_step = 0; grad_step < config.grad_steps; grad_step++) {
        double step_start = get_time();
        metrics.step_starts[grad_step] = step_start;

        // Sync model across processes (communication time)
        if (metrics.comm_starts[grad_step] == 0.0) metrics.comm_starts[grad_step] = step_start;
        broadcast_model_weights(policy.mlp, &mpi_ctx, 0);
        metrics.comm_times[grad_step] += (get_time() - step_start);

        mlp_zero_grad(policy.mlp);

        int idx = grad_step * config.episodes;
        for (int ep = 0; ep < config.episodes; ep++) {
            // Rollout
            double rollout_start = get_time();
            if (ep == 0 && metrics.rollout_starts[grad_step] == 0.0)
                metrics.rollout_starts[grad_step] = rollout_start;
            policy_rollout(&env, &policy, config.max_steps, &buffer);
            metrics.rollout_times[grad_step] += (get_time() - rollout_start);

            // policy_gradient(&policy, &buffer, config.gamma, NULL, &cache);
            // ...
            double forward_start = get_time();
            if (ep == 0 && metrics.forward_starts[grad_step] == 0.0)
                metrics.forward_starts[grad_step] = forward_start;
            discounted_cumsum(&buffer, config.gamma, returns);
            
            mlp_forward(policy.mlp, buffer.observations, buffer.size, logits, &cache);
            metrics.forward_times[grad_step] += (get_time() - forward_start);
            
            double backward_start = get_time();
            if (ep == 0 && metrics.backward_starts[grad_step] == 0.0)
                metrics.backward_starts[grad_step] = backward_start;
            policy_log_prob_from_logits(&policy, logits, buffer.actions, buffer.size, logp, dlogp);

            for (int t = 0; t < buffer.size; t++) {
                metrics.loss[idx + ep] += logp[t] * returns[t];

                for (int j = 0; j < out_size; j++) {
                    dlogp[t * out_size + j] *= -returns[t];
                }
            }
            
            mlp_backward(policy.mlp, &cache, dlogp, NULL);
            metrics.backward_times[grad_step] += (get_time() - backward_start);

            metrics.returns[idx + ep] = mean_return(&buffer);
            metrics.steps[idx + ep] = buffer.size;
        }

        // Aggregate gradients (communication time)
        double comm_start = get_time();
        aggregate_gradients(policy.mlp, &mpi_ctx, 0);
        metrics.comm_times[grad_step] += (get_time() - comm_start);

        double update_start = get_time();
        metrics.update_starts[grad_step] = update_start;
        if (mpi_ctx.rank == 0) {
            adam_step(policy.mlp, &optimizer_state, config.learning_rate, 0.9, 0.999, 1e-08);
        }
        metrics.update_times[grad_step] = (get_time() - update_start);

        metrics.step_times[grad_step] = (get_time() - step_start);
    }

    metrics.wall_time_train = (get_time() - training_start);
    metrics.wall_time_total = (get_time() - init_start);

    reduce_metrics(&metrics, &mpi_ctx, 0);

    if (config.output_dir != NULL) {
        if (mpi_ctx.rank == 0) {
            if (mkdir_p(config.output_dir) != 0) {
                fprintf(stderr, "ERROR: Failed to create output directory '%s'\n", config.output_dir);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        char timeline_path[512];
        snprintf(timeline_path, sizeof(timeline_path), "%s/training_timeline_rank%d.csv", config.output_dir, mpi_ctx.rank);
        write_metrics_timeline_csv(&metrics, &mpi_ctx, timeline_path);

        if (mpi_ctx.rank == 0) {
            char results_path[512];
            snprintf(results_path, sizeof(results_path), "%s/training_results.csv", config.output_dir);
            write_metrics_results_csv(&metrics, &mpi_ctx, results_path);

            snprintf(results_path, sizeof(results_path), "%s/weights.bin", config.output_dir);
            save_mlp_weights(policy.mlp, results_path);
        }
    }

    if (mpi_ctx.rank == 0) {
        if (config.render) render_episode(&env, &policy);

        print_training_summary(&metrics, &mpi_ctx, &config);
    }

    free_mlp_cache(&cache);
    free_buffer(&buffer);
    free_adam_state(&optimizer_state);

    free_mlp(policy.mlp);
    env_destroy(&env);

    mpi_finalize();

    return 0;
}

void print_array(float *array, int size) {
    fprintf(stderr, "{ %.3f", array[0]);
    for (int i = 1; i<size; i++)
        fprintf(stderr, ", %.3f", array[i]);

    fprintf(stderr, " }");
}

void render_episode(Env *env, Policy *policy) {
    InitWindow(WIDTH, HEIGHT, "puffer Cartpole");
    SetTargetFPS(50);

    float *obs = malloc(env->obs_size * sizeof(float));
    float *act = malloc(env->act_size * sizeof(float));

    float reward;
    bool done = false;

    env_reset(env, obs);

    int step_count = 0;
    for (; !done && !WindowShouldClose(); step_count++) {
        env_render(env);

        fprintf(stderr, "INFO: Observation ");
        print_array(obs, env->obs_size);
        fprintf(stderr, "\n");

        policy_sample_action(policy, obs, 1, act);

        env_step(env, act, obs, &reward, &done);

        fprintf(stderr, "INFO: Action      ");
        print_array(act, env->act_size);
        fprintf(stderr, "\n");

        fprintf(stderr, "INFO: Reward        %.3f\n", reward);
    };
    CloseWindow();

    fprintf(stderr, "INFO: Finished simulation at step %d\n", step_count);

    free(obs);
    free(act);
}

void print_training_summary(TrainingMetrics *metrics, MPIContext *mpi_ctx, Config *config) {
    if (mpi_ctx->rank != 0) return;
    
    // Set locale for number formatting
    setlocale(LC_NUMERIC, "");

    int updates_total = metrics->updates_capacity;
    int episodes_per_rank = updates_total * metrics->num_episodes;

    double time_rollout_total = 0.0;
    double time_forward_total = 0.0;
    double time_backward_total = 0.0;
    double time_update_total = 0.0;
    double time_comm_total = 0.0;
    double time_step_total = 0.0;
    for (int i = 0; i < updates_total; i++) {
        time_rollout_total += metrics->rollout_times[i];
        time_forward_total += metrics->forward_times[i];
        time_backward_total += metrics->backward_times[i];
        time_update_total += metrics->update_times[i];
        time_comm_total += metrics->comm_times[i];
        time_step_total += metrics->step_times[i];
    }

    double total_compute = time_forward_total + time_backward_total + 
                          time_update_total + time_rollout_total;

    double comm_ratio = metrics->wall_time_train > 0 ? 
                       (time_comm_total / metrics->wall_time_train) * 100 : 0;
    double compute_ratio = metrics->wall_time_train > 0 ? 
                          (total_compute / metrics->wall_time_train) * 100 : 0;

    int steps_total = 0;
    double returns_sum = 0.0;
    double returns_min = metrics->returns[0];
    double returns_max = metrics->returns[0];
    for (int i = 0; i < updates_total * metrics->num_episodes; i++) {
        returns_sum += metrics->returns[i];
        if (metrics->returns[i] < returns_min) returns_min = metrics->returns[i];
        if (metrics->returns[i] > returns_max) returns_max = metrics->returns[i];
        steps_total += metrics->steps[i];
    }

    double avg_return = episodes_per_rank > 0 ? returns_sum / episodes_per_rank : 0.0;
    double variance_acc = 0.0;
    for (int i = 0; i < updates_total * metrics->num_episodes; i++) {
        double diff = metrics->returns[i] - avg_return;
        variance_acc += diff * diff;
    }
    double return_variance = episodes_per_rank > 0 ? variance_acc / episodes_per_rank : 0.0;

    double steps_per_second = metrics->wall_time_train > 0 ? 
                             steps_total / metrics->wall_time_train : 0;
    double episodes_per_second = metrics->wall_time_train > 0 ? 
                                mpi_ctx->world_size * episodes_per_rank / metrics->wall_time_train : 0;
    
    double avg_episode_length = episodes_per_rank > 0 ? 
                               (double)steps_total / (episodes_per_rank * mpi_ctx->world_size) : 0;

    fprintf(stdout, "\n");
    fprintf(stdout, "=========================================================\n");
    fprintf(stdout, "              TRAINING SUMMARY REPORT                    \n");
    fprintf(stdout, "=========================================================\n");
    fprintf(stdout, "Environment:          %s\n", config->env_name);
    fprintf(stdout, "MPI Processes:        %d\n", mpi_ctx->world_size);
    fprintf(stdout, "Gradient Steps:       %d\n", updates_total);
    fprintf(stdout, "Episodes per Step:    %d\n", config->episodes);
    fprintf(stdout, "\n--- WALL TIME BREAKDOWN ---\n");
        fprintf(stdout, "  Total Time:         %.3f s\n", metrics->wall_time_total);
        fprintf(stdout, "  Training:           %.3f s (%.1f%%)\n", 
            metrics->wall_time_train,
            (metrics->wall_time_total > 0 ? (metrics->wall_time_train / metrics->wall_time_total) * 100 : 0));

    fprintf(stdout, "\n--- TRAINING TIME BREAKDOWN ---\n");
        fprintf(stdout, "  Communication:      %.3f s (%.1f%%)\n", 
            time_comm_total, comm_ratio);
        fprintf(stdout, "  Computation:        %.3f s (%.1f%%)\n", 
            total_compute, compute_ratio);
        fprintf(stdout, "    - Rollout:        %.3f s\n", time_rollout_total);
        fprintf(stdout, "    - Forward Pass:   %.3f s\n", time_forward_total);
        fprintf(stdout, "    - Backward Pass:  %.3f s\n", time_backward_total);
        fprintf(stdout, "    - Optimizer:      %.3f s\n", time_update_total);
    
    fprintf(stdout, "\n--- THROUGHPUT METRICS ---\n");
    fprintf(stdout, "  Total Episodes:     %d\n", episodes_per_rank * mpi_ctx->world_size);
    fprintf(stdout, "  Total Steps:        %'d\n", steps_total);
    fprintf(stdout, "  Avg Episode Length: %.1f steps\n", avg_episode_length);
    fprintf(stdout, "  Episodes/second:    %'.2f\n", episodes_per_second);
    fprintf(stdout, "  Steps/second:       %'.2f\n", steps_per_second);
    
    fprintf(stdout, "\n--- LEARNING/CONVERGENCE METRICS ---\n");
    fprintf(stdout, "  Avg Return:         %.2f\n", avg_return);
    fprintf(stdout, "  Min Return:         %.2f\n", returns_min);
    fprintf(stdout, "  Max Return:         %.2f\n", returns_max);
    fprintf(stdout, "  Return Std Dev:     %.2f\n", sqrt(return_variance));
    
    fprintf(stdout, "\n--- SCALABILITY METRICS ---\n");
    fprintf(stdout, "  Comm/Compute Ratio: %.2f%%\n", 
                total_compute > 0 ? (time_comm_total / total_compute) * 100 : 0);
            fprintf(stdout, "  Parallel Efficiency: %.1f%%\n",
                compute_ratio);
    fprintf(stdout, "\n=========================================================\n\n");
}

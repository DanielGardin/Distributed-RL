#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "raylib.h"

#include "algorithms/reinforce.h"
#include "environments/cartpole.h"
#include "nn/optimizers.h"
#include "nn/debug.h"
#include "rng.h"

#define WIDTH 600
#define HEIGHT 200

typedef struct {
    int seed;
    float gravity;
    int hidden_size;
    int episodes;
    int max_steps;
    float gamma;
    int grad_steps;
    float learning_rate;
} Config;

// Default values
#define DEFAULT_SEED 1
#define DEFAULT_GRAVITY 1.0f
#define DEFAULT_HIDDENSIZE 16
#define DEFAULT_EPISODES 1
#define DEFAULT_MAX_STEPS 200
#define DEFAULT_GAMMA 0.99f
#define DEFAULT_GRAD_STEPS 2500
#define DEFAULT_LEARNING_RATE 1e-2f

void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [options]\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s <int>   RNG seed (Default: %d)\n", DEFAULT_SEED);
    fprintf(stderr, "  -g <float> CartPole gravity (Default: %.2f)\n", DEFAULT_GRAVITY);
    fprintf(stderr, "  -n <float> Neural network's hidden size (Default: %d)\n", DEFAULT_HIDDENSIZE);
    fprintf(stderr, "  -e <int>   Episodes per gradient step (batch size) (Default: %d)\n", DEFAULT_EPISODES);
    fprintf(stderr, "  -m <int>   Max steps per episode (Default: %d)\n", DEFAULT_MAX_STEPS);
    fprintf(stderr, "  -y <float> Discount factor (gamma) (Default: %.2f)\n", DEFAULT_GAMMA);
    fprintf(stderr, "  -k <float> Number of gradient steps to perform (Default: %d)\n", DEFAULT_GRAD_STEPS);
    fprintf(stderr, "  -l <float> Learning rate (Default: %.0e)\n", DEFAULT_LEARNING_RATE);
    fprintf(stderr, "  -h         Print this help message\n");
}

void parse_arguments(int argc, char *argv[], Config *config);
void render_episode(Env *env, Policy *policy);

int main(int argc, char *argv[]) {
    Config config;
    parse_arguments(argc, argv, &config);

    rng_seed(config.seed);

    Env env = make_cartpole_env(config.gravity, false);

    Activation activations[2] = {relu, identity};
    int input_size[2] = {env.obs_size, config.hidden_size};
    int output_size = 1;

    MLP policynet = create_mlp(
        input_size,
        output_size,
        2,
        activations
    );

    for (int l=0; l<policynet.num_layers; l++)
        kaiming_init(&policynet.layers[l]);

    Policy policy = create_binary_policy(&policynet);

    for (int grad_step = 0; grad_step < config.grad_steps; grad_step++) {
        // if ((grad_step + 1)%100 == 0) {
            
        // }

        binary_policy_gradient(&env, &policy, config.episodes, config.max_steps, config.gamma, MeanBaseline);
        gd_step(&policynet, config.learning_rate);
        mlp_zero_grad(&policynet);
    }

    print_mlp(&policynet);
    // save_mlp_weights(&policy, "weights.bin");
    
    render_episode(&env, &policy);

    free_mlp(&policynet);
    env_destroy(&env);

    return 0;
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

        policy_sample_action(policy, obs, 1, act, NULL, NULL);

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

void parse_arguments(int argc, char *argv[], Config *config) {
    int opt;

    // Set default values first
    config->seed = DEFAULT_SEED;
    config->gravity = DEFAULT_GRAVITY;
    config->hidden_size = DEFAULT_HIDDENSIZE;
    config->episodes = DEFAULT_EPISODES;
    config->max_steps = DEFAULT_MAX_STEPS;
    config->gamma = DEFAULT_GAMMA;
    config->grad_steps = DEFAULT_GRAD_STEPS;
    config->learning_rate = DEFAULT_LEARNING_RATE;

    // Use "s:g:e:m:y:l:h" to specify options that take an argument
    while ((opt = getopt(argc, argv, "s:g:n:e:m:y:k:l:h")) != -1) {
        switch (opt) {
            case 's':
                config->seed = atoi(optarg);
                break;
            case 'g':
                config->gravity = atof(optarg);
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
            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);
            case '?': // Catch invalid options or missing arguments
                fprintf(stderr, "Unknown option or missing argument.\n");
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }
}
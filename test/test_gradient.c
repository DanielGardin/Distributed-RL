#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "mlp.h"
#include "rng.h"

#include "test_utils.c"

#define EPSILON 1e-3

float numerical_gradient_weight(
    MLP *mlp,
    const float *input,
    int batch_size,
    const float *target,
    int layer_idx,
    int weight_idx,
    float epsilon
) {
    // Compute loss at w + epsilon
    float original = mlp->layers[layer_idx].weights[weight_idx];
    mlp->layers[layer_idx].weights[weight_idx] = original + epsilon;

    float output[batch_size];
    mlp_forward(mlp, input, batch_size, output, NULL);

    float loss_pos = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float diff = output[i] - target[i];
        loss_pos += diff * diff;
    }
    loss_pos /= batch_size;

    // Compute loss at w - epsilon
    mlp->layers[layer_idx].weights[weight_idx] = original - epsilon;

    float output_neg[batch_size];
    mlp_forward(mlp, input, batch_size, output_neg, NULL);

    float loss_neg = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float diff = output_neg[i] - target[i];
        loss_neg += diff * diff;
    }
    loss_neg /= batch_size;

    mlp->layers[layer_idx].weights[weight_idx] = original;

    // Numerical gradient: (f(w+eps) - f(w-eps)) / (2*eps)
    return (loss_pos - loss_neg) / (2.0f * epsilon);
}

float numerical_gradient_bias(
    MLP *mlp,
    const float *input,
    int batch_size,
    const float *target,
    int layer_idx,
    int bias_idx,
    float epsilon
) {
    // Compute loss at b + epsilon
    float original = mlp->layers[layer_idx].biases[bias_idx];
    mlp->layers[layer_idx].biases[bias_idx] = original + epsilon;

    float output[batch_size];
    mlp_forward(mlp, input, batch_size, output, NULL);

    float loss_pos = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float diff = output[i] - target[i];
        loss_pos += diff * diff;
    }
    loss_pos /= batch_size;

    // Compute loss at b - epsilon
    mlp->layers[layer_idx].biases[bias_idx] = original - epsilon;

    float output_neg[batch_size];
    mlp_forward(mlp, input, batch_size, output_neg, NULL);

    float loss_neg = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float diff = output_neg[i] - target[i];
        loss_neg += diff * diff;
    }
    loss_neg /= batch_size;

    mlp->layers[layer_idx].biases[bias_idx] = original;

    // Numerical gradient: (f(b+eps) - f(b-eps)) / (2*eps)
    return (loss_pos - loss_neg) / (2.0f * epsilon);
}

int test_gradient_2layer_all() {
    TEST_START("2-layer MLP all gradients (weights and biases)");
    int input_size = 8;
    int n_layers = 2;
    int batch_size = 3;

    int layer_sizes[] = {input_size, 16, 1};
    Activation acts[] = {relu, identity};

    MLP mlp = create_mlp(layer_sizes, 1, n_layers, acts);
    kaiming_mlp_init(&mlp);

    float input[input_size*batch_size];
    for (int i = 0; i < input_size*batch_size; i++)
        input[i] = rand_uniform(-1.0f, 1.0f);

    float target[batch_size];
    for (int i = 0; i < batch_size; i++)
        target[i] = rand_uniform(-1.0f, 1.0f);

    float output[3];
    MLPCache cache = create_mlp_cache(&mlp, batch_size);
    mlp_forward(&mlp, input, batch_size, output, &cache);

    float out_grad[3];
    for (int i = 0; i < batch_size; i++) {
        out_grad[i] = 2.0f * (output[i] - target[i]) / batch_size; // d(MSE)/d(output)
    }

    mlp_backward(&mlp, &cache, out_grad, NULL);

    printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    printf(  "│ WEIGHT GRADIENTS: ANALYTICAL (autograd) vs NUMERICAL               │\n");
    printf(  "├───────────┬──────────────┬──────────────┬──────────────┬───────────┤\n");
    printf(  "│ param     │ autograd     │ numerical    │ difference   │ rel err   │\n");
    printf(  "├───────────┼──────────────┼──────────────┼──────────────┼───────────┤\n");

    int total_errors = 0;
    for (int l = 0; l < n_layers; l++) {
        int num_weights = mlp.layers[l].output_size * mlp.layers[l].input_size;

        for (int w = 0; w < num_weights; w++) {
            float analytical = mlp.layers[l].weights_grad[w];
            float numerical = numerical_gradient_weight(&mlp, input, batch_size, target, l, w, EPSILON);
            float abs_diff = fabsf(analytical - numerical);
            float denom = fmaxf(fabsf(numerical), 1e-3f);
            float rel_error = abs_diff / (denom + 1e-6f);

            printf("│ L%d.w[%-3d] │ %-+12.8f │ %-+12.8f │ %-+12.8f │ %-+8.4f%% │\n",
                   l, w, analytical, numerical, analytical - numerical, rel_error * 100);

            if (rel_error >= 0.01f && abs_diff >= 1e-3f) {
                total_errors++;
            }
        }
    }

    printf("├───────────┴──────────────┴──────────────┴──────────────┴───────────┤\n");
    printf("│ BIAS GRADIENTS: ANALYTICAL (autograd) vs NUMERICAL                 │\n");
    printf("├───────────┬──────────────┬──────────────┬──────────────┬───────────┤\n");

    for (int l = 0; l < 2; l++) {
        for (int b = 0; b < mlp.layers[l].output_size; b++) {
            float analytical = mlp.layers[l].biases_grad[b];
            float numerical = numerical_gradient_bias(&mlp, input, batch_size, target, l, b, EPSILON);
            float abs_diff = fabsf(analytical - numerical);
            float denom = fmaxf(fabsf(numerical), 1e-3f);
            float rel_error = abs_diff / (denom + 1e-6f);

            printf("│ L%d.b[%-3d] │ %-+12.8f │ %-+12.8f │ %-+12.8f │ %-+8.4f%% │\n",
                   l, b, analytical, numerical, analytical - numerical, rel_error * 100);

            if (rel_error >= 0.01f && abs_diff >= 1e-3f) {
                total_errors++;
            }
        }
    }

    printf("└───────────┴──────────────┴──────────────┴──────────────┴───────────┘\n\n");

    free_mlp_cache(&cache);
    free_mlp(&mlp);

    if (total_errors > 0) {
        printf(RED "[FAIL] %d gradient mismatches\n" RESET, total_errors);
        return 1;
    }

    TEST_END("2-layer MLP all gradients");
    return 0;
}

int main() {
    rng_seed((unsigned int)time(NULL));

    int failures = 0;

    failures += test_gradient_2layer_all();

    if (failures == 0) {
        printf("\n" GRN "=== ALL TESTS PASSED ===" RESET "\n");
    } else {
        printf("\n" RED "=== %d TEST(S) FAILED ===" RESET "\n", failures);
    }

    return failures;
}

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mlp.h"
#include "nn/linear.h"
#include "optimizers.h"

#include "test_utils.c"

int test_mlp_overfit() {
    TEST_START("MLP overfitting on small dataset");

    int layer_sizes[] = {2, 1};
    Activation activations[] = {identity};
    MLP mlp = create_mlp(layer_sizes, 1, 1, activations);

    int batch_size = 32;
    float *input = (float *)malloc(batch_size * 2 * sizeof(float));
    float *targets = (float *)malloc(batch_size * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        input[i * 2 + 0] = (float)(i % 8) / 4.0f - 1.0f;
        input[i * 2 + 1] = (float)(i / 8) / 2.0f - 1.0f;
        targets[i] = 2.0f * input[i * 2 + 0] + 3.0f * input[i * 2 + 1];
    }

    MLPCache cache = create_mlp_cache(&mlp, batch_size);
    int num_epochs = 1000;
    float learning_rate = 0.01f;
    float final_loss = 0.0f;

    float *predictions = (float *)malloc(batch_size * sizeof(float));
    float *output_grad = (float *)malloc(batch_size * sizeof(float));

    Optimizer opt = make_gd(learning_rate);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        mlp_zero_grad(&mlp);
        mlp_forward(&mlp, input, batch_size, predictions, &cache);

        float total_loss = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float error = predictions[i] - targets[i];
            total_loss += error * error;
            output_grad[i] = 2.0f * error / batch_size;
        }
        final_loss = total_loss / batch_size;

        mlp_backward(&mlp, &cache, output_grad, NULL);
        optimizer_step(&opt, &mlp, &cache);

        if (epoch % 250 == 0) {
            printf("  Epoch %d, Loss: %.6f\n", epoch, final_loss);
        }
    }

    printf("  Final Loss: %.6f\n", final_loss);
    ASSERT_TRUE("loss decreased significantly", final_loss < 0.015f);

    // Final evaluation
    mlp_forward(&mlp, input, batch_size, predictions, &cache);

    float max_error = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float error = fabs(predictions[i] - targets[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("  Max prediction error: %.6f\n", max_error);
    ASSERT_TRUE("predictions match targets", max_error < 0.02f);
    
    free(predictions);
    free(output_grad);
    free(input);
    free(targets);
    free_optimizer(&opt);
    free_mlp_cache(&cache);
    free_mlp(&mlp);

    TEST_END("MLP overfitting on small dataset");
    return 0;
}

int test_mlp_2layer_overfit() {
    TEST_START("2-layer MLP overfitting on small dataset");

    int layer_sizes[] = {2, 128, 1};
    Activation activations[] = {relu, identity};
    MLP mlp = create_mlp(layer_sizes, 1, 2, activations);
    
    kaiming_mlp_init(&mlp);

    int batch_size = 32;
    float *input = (float *)malloc(batch_size * 2 * sizeof(float));
    float *targets = (float *)malloc(batch_size * sizeof(float));

    for (int i = 0; i < batch_size; i++) {
        input[i * 2 + 0] = (float)(i % 8) / 4.0f - 1.0f;
        input[i * 2 + 1] = (float)(i / 8) / 2.0f - 1.0f;
        float x1 = input[i * 2 + 0];
        float x2 = input[i * 2 + 1];
        targets[i] = sinf(x1 * 3.14159f) * cosf(x2 * 3.14159f) + (x1*x1*x1 - x2*x2*x2);
    }

    MLPCache cache = create_mlp_cache(&mlp, batch_size);
    Optimizer opt = make_gd(0.05f);

    int num_epochs = 5000;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        mlp_zero_grad(&mlp);
        float *predictions = (float *)malloc(batch_size * sizeof(float));
        mlp_forward(&mlp, input, batch_size, predictions, &cache);

        float total_loss = 0.0f;
        float *output_grad = (float *)malloc(batch_size * sizeof(float));
        for (int i = 0; i < batch_size; i++) {
            float error = predictions[i] - targets[i];
            total_loss += error * error;
            output_grad[i] = 2.0f * error / batch_size;
        }
        final_loss = total_loss / batch_size;

        float *input_grad = (float *)malloc(batch_size * 2 * sizeof(float));
        mlp_backward(&mlp, &cache, output_grad, input_grad);
        optimizer_step(&opt, &mlp, &cache);

        free(predictions);
        free(output_grad);
        free(input_grad);

        if (epoch % 500 == 0) {
            printf("  Epoch %d, Loss: %.6f\n", epoch, final_loss);
        }
    }

    printf("  Final Loss: %.6f\n", final_loss);
    ASSERT_TRUE("loss decreased significantly", final_loss < 0.1f);

    float *final_predictions = (float *)malloc(batch_size * sizeof(float));
    mlp_forward(&mlp, input, batch_size, final_predictions, &cache);

    float max_error = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        float error = fabs(final_predictions[i] - targets[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("  Max prediction error: %.6f\n", max_error);
    ASSERT_TRUE("predictions match targets", max_error < 0.5f);

    free(input);
    free(targets);
    free(final_predictions);
    free_optimizer(&opt);
    free_mlp_cache(&cache);
    free_mlp(&mlp);

    TEST_END("2-layer MLP overfitting on small dataset");
    return 0;
}

int main() {
    if (test_mlp_overfit() != 0) {
        return 1;
    }
    if (test_mlp_2layer_overfit() != 0) {
        return 1;
    }
    return 0;
}

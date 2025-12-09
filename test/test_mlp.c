#include <stdlib.h>
#include <math.h>

#include "mlp.h"

#include "test_utils.c"

int test_forward_correctness() {
    TEST_START("forward correctness");
    int layer_sizes[] = {2, 2, 1};
    Activation acts[] = {relu, identity};

    MLP mlp = create_mlp(layer_sizes, 1, 2, acts);

    // Set layer 0 weights manually: 2x2 matrix
    // [1.0  2.0]
    // [3.0  4.0]
    mlp.layers[0].weights[0] = 1.0f;  // w[0,0]
    mlp.layers[0].weights[1] = 2.0f;  // w[0,1]
    mlp.layers[0].weights[2] = 3.0f;  // w[1,0]
    mlp.layers[0].weights[3] = 4.0f;  // w[1,1]
    mlp.layers[0].biases[0] = 2.0f;
    mlp.layers[0].biases[1] = 0.5f;

    // Set layer 1 weights: 2x1 matrix
    // [2.0]
    // [1.0]
    mlp.layers[1].weights[0] = 2.0f;
    mlp.layers[1].weights[1] = 1.0f;
    mlp.layers[1].biases[0] = 0.0f;

    float input[2] = {1.0f, -1.0f};
    float expected = 2.0f;
    float output;

    MLPCache cache = create_mlp_cache(&mlp, 1);
    mlp_forward(&mlp, input, 1, &output, &cache);

    ASSERT_FLOAT_EQ("forward pass", output, expected, GLOBAL_TOL);
    
    float exp0[2] = { 1.0f, -0.5f };
    ASSERT_FLOAT_EQ_ARR("layer0 pre-activation", cache.layer_caches[0].pre_activations, exp0, 2, GLOBAL_TOL);

    float exp1[1] = { 2.0f };
    ASSERT_FLOAT_EQ_ARR("layer1 pre-activation", cache.layer_caches[1].pre_activations, exp1, 1, GLOBAL_TOL);

    free_mlp_cache(&cache);
    free_mlp(&mlp);

    TEST_END("forward correctness");
    return 0;
}

int test_forward_batched() {
    TEST_START("forward batch correctness");
    int layer_sizes[] = {2, 1};
    Activation acts[] = {identity};

    MLP mlp = create_mlp(layer_sizes, 1, 1, acts);

    // (x, y) -> x + y
    mlp.layers[0].weights[0] = 1.0f;
    mlp.layers[0].weights[1] = 1.0f;
    mlp.layers[0].biases[0] = 0.0f;

    float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, -0.5f, 0.5f, 8.0f, 4.0f};
    float expected[4] = {3.0f, 7.0f, 0.0f, 12.0f};
    float output[4];

    MLPCache cache = create_mlp_cache(&mlp, 4);
    mlp_forward(&mlp, input, 4, output, &cache);

    ASSERT_FLOAT_EQ_ARR("forward batched pass", output, expected, 4, GLOBAL_TOL);

    free_mlp_cache(&cache);
    free_mlp(&mlp);

    TEST_END("forward batch correctness");
    return 0;
}


int test_backward_correctness() {
    TEST_START("backward correctness");

    int layer_sizes[] = {2, 2, 1};
    Activation acts[] = {relu, identity};

    MLP mlp = create_mlp(layer_sizes, 1, 2, acts);

    // Set layer 0 weights manually: 2x2 matrix
    // [1.0  2.0]
    // [3.0  4.0]
    mlp.layers[0].weights[0] = 1.0f;  // w[0,0]
    mlp.layers[0].weights[1] = 2.0f;  // w[0,1]
    mlp.layers[0].weights[2] = 3.0f;  // w[1,0]
    mlp.layers[0].weights[3] = 4.0f;  // w[1,1]
    mlp.layers[0].biases[0] = 2.0f;
    mlp.layers[0].biases[1] = 0.5f;

    // Set layer 1 weights: 2x1 matrix
    // [2.0]
    // [1.0]
    mlp.layers[1].weights[0] = 2.0f;
    mlp.layers[1].weights[1] = 1.0f;
    mlp.layers[1].biases[0] = 0.0f;

    float input[2] = {1.0f, -1.0f};
    float output;

    MLPCache cache = create_mlp_cache(&mlp, 1);
    mlp_forward(&mlp, input, 1, &output, &cache);

    float out_grad = 1.0f;
    float in_grad[2];

    mlp_zero_grad(&mlp);
    mlp_backward(&mlp, &cache, &out_grad, in_grad);

    float expected_in_grad[2] = {2.0f, 4.0f};
    ASSERT_FLOAT_EQ_ARR("input gradient", in_grad, expected_in_grad, 2, GLOBAL_TOL);

    // Layer 1 gradients
    float expected_w1_grad[2] = {1.0f, 0.0f};
    ASSERT_FLOAT_EQ_ARR("layer1 weight gradient", mlp.layers[1].weights_grad, expected_w1_grad, 2, GLOBAL_TOL);

    float expected_b1_grad[1] = {1.0f};
    ASSERT_FLOAT_EQ_ARR("layer1 bias gradient", mlp.layers[1].biases_grad, expected_b1_grad, 1, GLOBAL_TOL);

    // Layer 0 gradients
    float expected_w0_grad[4] = {2.0f, -2.0f, 0.0f, 0.0f};
    ASSERT_FLOAT_EQ_ARR("layer0 weight gradient", mlp.layers[0].weights_grad, expected_w0_grad, 4, GLOBAL_TOL);

    float expected_b0_grad[2] = {2.0f, 0.0f};
    ASSERT_FLOAT_EQ_ARR("layer0 bias gradient", mlp.layers[0].biases_grad, expected_b0_grad, 2, GLOBAL_TOL);

    free_mlp_cache(&cache);
    free_mlp(&mlp);

    TEST_END("backward correctness");
    return 0;
}


int main() {
    int total_tests = 1;
    int failed_tests = 0;

    failed_tests += test_forward_correctness();

    failed_tests += test_forward_batched();

    failed_tests += test_backward_correctness();

    return 0;
}
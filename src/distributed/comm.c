#include <stdlib.h>

#include <mpi.h>

#include "distributed/comm.h"

static void serialize_gradients(const MLP *mlp, float *buffer) {
    int offset = 0;
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = &mlp->layers[i];
        int weights_size = layer->input_size * layer->output_size;
        int biases_size = layer->output_size;

        for (int j = 0; j < weights_size; j++) {
            buffer[offset++] = layer->weights_grad[j];
        }

        for (int j = 0; j < biases_size; j++) {
            buffer[offset++] = layer->biases_grad[j];
        }
    }
}

static void deserialize_gradients(MLP *mlp, const float *buffer) {
    int offset = 0;
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = &mlp->layers[i];
        int weights_size = layer->input_size * layer->output_size;
        int biases_size = layer->output_size;

        for (int j = 0; j < weights_size; j++) {
            layer->weights_grad[j] = buffer[offset++];
        }

        for (int j = 0; j < biases_size; j++) {
            layer->biases_grad[j] = buffer[offset++];
        }
    }
}
static void serialize_weights(const MLP *mlp, float *buffer) {
    int offset = 0;
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = &mlp->layers[i];
        int weights_size = layer->input_size * layer->output_size;
        int biases_size = layer->output_size;

        for (int j = 0; j < weights_size; j++) {
            buffer[offset++] = layer->weights[j];
        }

        for (int j = 0; j < biases_size; j++) {
            buffer[offset++] = layer->biases[j];
        }
    }
}

static void deserialize_weights(MLP *mlp, const float *buffer) {
    int offset = 0;
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = &mlp->layers[i];
        int weights_size = layer->input_size * layer->output_size;
        int biases_size = layer->output_size;

        for (int j = 0; j < weights_size; j++) {
            layer->weights[j] = buffer[offset++];
        }

        for (int j = 0; j < biases_size; j++) {
            layer->biases[j] = buffer[offset++];
        }
    }
}

void broadcast_model_weights(MLP *mlp, const MPIContext *mpi_ctx, int src_rank) {
    int total_params = get_num_params(mlp);
    float *weights_buffer = (float *)malloc(total_params * sizeof(float));

    if (mpi_ctx->rank == src_rank) {
        serialize_weights(mlp, weights_buffer);
    }

    MPI_Bcast(weights_buffer, total_params, MPI_FLOAT, src_rank, mpi_ctx->comm);

    deserialize_weights(mlp, weights_buffer);
    
    free(weights_buffer);
}

void aggregate_gradients(MLP *mlp, const MPIContext *mpi_ctx, int compute_rank) {
    int total_params = get_num_params(mlp);
    float *local_grad_buffer = (float *)malloc(total_params * sizeof(float));
    float *aggregated_grad_buffer = (float *)malloc(total_params * sizeof(float));

    // All ranks serialize their gradients
    serialize_gradients(mlp, local_grad_buffer);
    
    // Sum all gradients to compute_rank using MPI_Reduce
    MPI_Reduce(local_grad_buffer, aggregated_grad_buffer, total_params, MPI_FLOAT, 
               MPI_SUM, compute_rank, mpi_ctx->comm);
    
    // Only compute_rank computes the mean and deserializes the aggregated gradients
    if (mpi_ctx->rank == compute_rank) {
        // float world_size_inv = 1.0f / mpi_ctx->world_size;
        // for (int i = 0; i < total_params; i++) {
        //     aggregated_grad_buffer[i] *= world_size_inv;
        // }
        deserialize_gradients(mlp, aggregated_grad_buffer);
    }
    
    free(local_grad_buffer);
    free(aggregated_grad_buffer);
}

#pragma once

#include "nn/mlp.h"
#include "mpi_utils.h"

void broadcast_model_weights(MLP *mlp, const MPIContext *mpi_ctx, int src_rank);

void aggregate_gradients(MLP *mlp, const MPIContext *mpi_ctx, int compute_rank);

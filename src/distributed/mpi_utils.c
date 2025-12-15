#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include "distributed/mpi_utils.h"

MPIContext mpi_init_context(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    
    MPIContext ctx;
    ctx.comm = MPI_COMM_WORLD;
    
    MPI_Comm_rank(ctx.comm, &ctx.rank);
    MPI_Comm_size(ctx.comm, &ctx.world_size);
    
    return ctx;
}

void main_printf(const MPIContext *ctx, const char *format, ...) {
    if (ctx->rank == 0) {
        va_list args;
        va_start(args, format);
        vfprintf(stderr, format, args);
        va_end(args);
    }
}

void mpi_finalize() {
    MPI_Finalize();
}

double get_time() {
    return MPI_Wtime();
}

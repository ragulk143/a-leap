#include "kernels.h"

namespace aleap {

__global__ void gemv_fp16_kernel(
    const half* __restrict__ activations,
    const half* __restrict__ weights,
    half* __restrict__ outputs,
    int m_rows,
    int n_cols
) {
    int row = blockIdx.x; // Thread block targets specific matrix rows
    int tid = threadIdx.x;

    extern __shared__ half shared_sum[];

    #if __CUDA_ARCH__ >= 530
    half accumulation = __float2half(0.0f);

    // Grid-stride reduction loop over the wide column layout
    for (int col = tid; col < n_cols; col += blockDim.x) {
        half act = activations[col];
        half weight = weights[row * n_cols + col];
        accumulation = __hadd(accumulation, __hmul(act, weight));
    }
    shared_sum[tid] = accumulation;
    __syncthreads();

    // In-place parallel execution tree reduction across shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] = __hadd(shared_sum[tid], shared_sum[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputs[row] = shared_sum[0];
    }
    #endif
}

void launch_gemv(
    const half* d_activations, const half* d_weights, half* d_outputs,
    int m_rows, int n_cols, cudaStream_t stream
) {
    int threads_per_block = 256;
    size_t shared_mem_size = threads_per_block * sizeof(half);

    gemv_fp16_kernel<<<m_rows, threads_per_block, shared_mem_size, stream>>>(
        d_activations, d_weights, d_outputs, m_rows, n_cols
    );
}

} // namespace aleap


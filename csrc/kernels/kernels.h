#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace aleap {

// Inflates bit-packed INT4 weight schemes to FP16 matrix parameters on-the-fly
__global__ void decompress_int4_to_fp16_kernel(
    const uint8_t* __restrict__ packed_weights,
    half* __restrict__ unpacked_outputs,
    const half* __restrict__ scales,
    const half* __restrict__ biases,
    size_t total_elements
);

// High-speed fused Tensor Core Matrix-Vector Multiplication Engine
__global__ void gemv_fp16_kernel(
    const half* __restrict__ activations,
    const half* __restrict__ weights,
    half* __restrict__ outputs,
    int m_rows,
    int n_cols
);

// Host execution wrappers exposed to our runner system
void launch_decompression(
    const uint8_t* d_packed, half* d_unpacked, 
    const half* d_scales, const half* d_biases, 
    size_t total_elements, cudaStream_t stream
);

void launch_gemv(
    const half* d_activations, const half* d_weights, half* d_outputs,
    int m_rows, int n_cols, cudaStream_t stream
);

} // namespace aleap

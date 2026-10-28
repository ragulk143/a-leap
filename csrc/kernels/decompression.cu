#include "kernels.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace aleap {

__global__ void decompress_int4_to_fp16_kernel(
    const uint8_t* __restrict__ packed_weights,
    half* __restrict__ unpacked_outputs,
    const half* __restrict__ scales,
    const half* __restrict__ biases,
    size_t total_elements
) {
    // Coordinate 1D grid layout to map total element counts
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements / 2) return; // Each thread processes 1 byte (2 int4 weights)

    uint8_t packed_val = packed_weights[idx];
    
    #if __CUDA_ARCH__ >= 530
    // Unpack low 4 bits and scale
    int4 low_int = { static_cast<int>(packed_val & 0x0F) };
    half low_fp = __int2half_rn(low_int.x - 8); // Symmetric offset bias shift
    unpacked_outputs[idx * 2] = __hadd(__hmul(low_fp, scales[idx * 2]), biases[idx * 2]);

    // Unpack high 4 bits and scale
    int4 high_int = { static_cast<int>((packed_val >> 4) & 0x0F) };
    half high_fp = __int2half_rn(high_int.x - 8);
    unpacked_outputs[idx * 2 + 1] = __hadd(__hmul(high_fp, scales[idx * 2 + 1]), biases[idx * 2 + 1]);
    #endif
}

void launch_decompression(
    const uint8_t* d_packed, half* d_unpacked, 
    const half* d_scales, const half* d_biases, 
    size_t total_elements, cudaStream_t stream
) {
    int threads_per_block = 256;
    size_t total_bytes = total_elements / 2;
    size_t blocks = (total_bytes + threads_per_block - 1) / threads_per_block;

    decompress_int4_to_fp16_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_packed, d_unpacked, d_scales, d_biases, total_elements
    );
}

} // namespace aleap

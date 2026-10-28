#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace aleap {

class RingBuffer {
private:
    size_t layer_size_;
    size_t total_layers_;
    
    // Dual hardware streams for absolute parallelism
    cudaStream_t stream_compute_;
    cudaStream_t stream_fetch_;
    
    // Ping-Pong hardware memory addresses in VRAM
    void* d_ping_buffer_;
    void* d_pong_buffer_;
    
    // Host pinned memory pointers (Mapped to NVMe via zero-copy virtual memory handles)
    std::vector<void*> h_pinned_layers_;

    void checkCuda(cudaError_t result, const char* message);

public:
    RingBuffer(size_t layer_size, size_t total_layers);
    ~RingBuffer();

    // Prevent implicit copies to safeguard raw VRAM handles
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Registers host-side pointers to our zero-copy NVMe mapped streams
    void register_host_layer(int layer_idx, void* host_ptr);

    // Asynchronously loads a target layer into the designated buffer (Ping or Pong)
    void prefetch_layer_async(int layer_idx, bool use_pong);

    // Synchronizes the fetch thread to guarantee the layer arrived before compute begins
    void synchronize_fetch();

    // Accessor methods for active compute targeting
    void* get_ping_ptr() const { return d_ping_buffer_; }
    void* get_pong_ptr() const { return d_pong_buffer_; }
    cudaStream_t get_compute_stream() const { return stream_compute_; }
    cudaStream_t get_fetch_stream() const { return stream_fetch_; }
};

} // namespace aleap


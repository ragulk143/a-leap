#include "ring_buffer.h"

namespace aleap {

void RingBuffer::checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::string err = std::string(message) + " : " + cudaGetErrorString(result);
        throw std::runtime_error(err);
    }
}

RingBuffer::RingBuffer(size_t layer_size, size_t total_layers)
    : layer_size_(layer_size), total_layers_(total_layers), d_ping_buffer_(nullptr), d_pong_buffer_(nullptr) {
    
    // 1. Initialize dedicated asynchronous hardware tracks
    checkCuda(cudaStreamCreateWithFlags(&stream_compute_, cudaStreamNonBlocking), "Failed to create compute stream");
    checkCuda(cudaStreamCreateWithFlags(&stream_fetch_, cudaStreamNonBlocking), "Failed to create fetch stream");

    // 2. Allocate the high-speed Ping-Pong buffer pools in physical VRAM
    checkCuda(cudaMalloc(&d_ping_buffer_, layer_size_), "Failed to allocate VRAM Ping buffer");
    checkCuda(cudaMalloc(&d_pong_buffer_, layer_size_), "Failed to allocate VRAM Pong buffer");

    h_pinned_layers_.resize(total_layers_, nullptr);
    std::cout << "[A-LEAP Engine] Ring Buffer initialized. Layer size: " 
              << layer_size_ / (1024 * 1024) << " MB. Dual VRAM buffers bound." << std::endl;
}

RingBuffer::~RingBuffer() {
    // Teardown VRAM
    if (d_ping_buffer_) cudaFree(d_ping_buffer_);
    if (d_pong_buffer_) cudaFree(d_pong_buffer_);
    
    // Destroy execution streams
    cudaStreamDestroy(stream_compute_);
    cudaStreamDestroy(stream_fetch_);
}

void RingBuffer::register_host_layer(int layer_idx, void* host_ptr) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(total_layers_)) {
        throw std::out_of_range("Layer index out of bounds during host tracking registration");
    }
    h_pinned_layers_[layer_idx] = host_ptr;
}

void RingBuffer::prefetch_layer_async(int layer_idx, bool use_pong) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(total_layers_)) {
        throw std::out_of_range("Invalid prefetch target index requested");
    }
    
    void* host_src = h_pinned_layers_[layer_idx];
    if (!host_src) {
        throw std::runtime_error("Attempted to prefetch an unregistered host layer address");
    }

    void* vram_dst = use_pong ? d_pong_buffer_ : d_ping_buffer_;

    // Execute absolute asynchronous DMA transfer over PCIe bus without blocking CPU execution
    checkCuda(cudaMemcpyAsync(vram_dst, host_src, layer_size_, cudaMemcpyHostToDevice, stream_fetch_),
              "PCIe Asynchronous DMA transfer initiation failed");
}

void RingBuffer::synchronize_fetch() {
    // Block until the streaming track finishes its specific layer delivery
    checkCuda(cudaStreamSynchronize(stream_fetch_), "Failed to sync memory prefetch track");
}

} // namespace aleap


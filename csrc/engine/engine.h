#pragma once
#include "../memory/ring_buffer.h"
#include "../kernels/kernels.h"
#include <memory>

namespace aleap {

class EnginePipeline {
private:
    std::unique_ptr<RingBuffer> ring_buffer_;
    size_t layer_size_;
    int total_layers_;

public:
    EnginePipeline(size_t layer_size, int total_layers);
    ~EnginePipeline() = default;

    void register_host_layer(int layer_idx, uintptr_t host_ptr);
    void prefetch_layer_async(int layer_idx, bool use_pong);
    void synchronize_fetch();
    
    uintptr_t get_ping_ptr() const;
    uintptr_t get_pong_ptr() const;
    uintptr_t get_compute_stream() const;
    uintptr_t get_fetch_stream() const;
};

} // namespace aleap

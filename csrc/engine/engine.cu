#include "engine.h"

namespace aleap {

EnginePipeline::EnginePipeline(size_t layer_size, int total_layers)
    : layer_size_(layer_size), total_layers_(total_layers) {
    ring_buffer_ = std::make_unique<RingBuffer>(layer_size_, total_layers_);
}

void EnginePipeline::register_host_layer(int layer_idx, uintptr_t host_ptr) {
    ring_buffer_->register_host_layer(layer_idx, reinterpret_cast<void*>(host_ptr));
}

void EnginePipeline::prefetch_layer_async(int layer_idx, bool use_pong) {
    ring_buffer_->prefetch_layer_async(layer_idx, use_pong);
}

void EnginePipeline::synchronize_fetch() {
    ring_buffer_->synchronize_fetch();
}

uintptr_t EnginePipeline::get_ping_ptr() const {
    return reinterpret_cast<uintptr_t>(ring_buffer_->get_ping_ptr());
}

uintptr_t EnginePipeline::get_pong_ptr() const {
    return reinterpret_cast<uintptr_t>(ring_buffer_->get_pong_ptr());
}

uintptr_t EnginePipeline::get_compute_stream() const {
    return reinterpret_cast<uintptr_t>(ring_buffer_->get_compute_stream());
}

uintptr_t EnginePipeline::get_fetch_stream() const {
    return reinterpret_cast<uintptr_t>(ring_buffer_->get_fetch_stream());
}

} // namespace aleap

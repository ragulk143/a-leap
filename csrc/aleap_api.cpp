#include <pybind11/pybind11.h>
#include "engine/engine.h"

namespace py = pybind11;

PYBIND11_MODULE(aleap_api, m) {
    m.doc() = "A-LEAP High Performance Asynchronous Memory Ring Buffer Engine API";

    py::class_<aleap::EnginePipeline>(m, "RingBuffer")
        .def(py::init<size_t, int>(), py::arg("layer_size"), py::arg("total_layers"))
        .def("register_host_layer", &aleap::EnginePipeline::register_host_layer, py::arg("layer_idx"), py::arg("host_ptr"))
        .def("prefetch_layer_async", &aleap::EnginePipeline::prefetch_layer_async, py::arg("layer_idx"), py::arg("use_pong"))
        .def("synchronize_fetch", &aleap::EnginePipeline::synchronize_fetch)
        .def("get_ping_ptr", &aleap::EnginePipeline::get_ping_ptr)
        .def("get_pong_ptr", &aleap::EnginePipeline::get_pong_ptr)
        .def("get_compute_stream", &aleap::EnginePipeline::get_compute_stream)
        .def("get_fetch_stream", &aleap::EnginePipeline::get_fetch_stream);
}

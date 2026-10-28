# Define the core high-performance GPU driver runtime library
add_library(aleap_cuda SHARED
    memory/ring_buffer.cpp
    engine/engine.cu
    kernels/decompression.cu
    kernels/gemm.cu
)

# Route header search boundaries cleanly across targets
target_include_directories(aleap_cuda PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/local/cuda/include
)

# Explicitly link system-level CUDA driver operations to our C++ runtime object
target_link_libraries(aleap_cuda PRIVATE cuda cudart)

# Bind the Python interface entry point library module using Pybind11
pybind11_add_module(aleap_api aleap_api.cpp)
target_link_libraries(aleap_api PRIVATE aleap_cuda)

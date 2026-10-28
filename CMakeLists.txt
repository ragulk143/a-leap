cmake_minimum_required(VERSION 3.18)
project(a_leap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure position independent code for python extensions
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Discover local Pybind11 configurations
find_package(pybind11 REQUIRED)

# Force discovery of raw system-level CUDA paths
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/local/cuda/include
)

link_directories(/usr/local/cuda/lib64)

add_subdirectory(csrc)

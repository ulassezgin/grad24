cmake_minimum_required(VERSION 3.27)
project(spadalib64 LANGUAGES CXX CUDA)

set(
    CUDA_SOURCE_FILES
    spada_1_64.cu
    spada_2_32.cu
    spada_4_16.cu
    spada_8_8.cu
    spada_16_4.cu
    spada_32_2.cu
    spada_64_1.cu
    spada_lib.cu
)

add_library(spadalib64 SHARED
    ${CUDA_SOURCE_FILES}
)

set_target_properties(spadalib64 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(spadalib64
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    # Add more include directories here
)

target_link_libraries(spadalib64 PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUDA_COMPUTE_SANITIZER_LIBRARY}
    ${CUDA_RUNTIME}
    # Add more libraries here
)

target_compile_definitions(spadalib64 PRIVATE
    # Add more definitions here
    -DCUDA_SANITIZER_MEMCHECK_FULL
)



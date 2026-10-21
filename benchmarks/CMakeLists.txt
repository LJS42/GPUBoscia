add_executable(benchmark_multiblock benchmark_multiblock.cu)
target_link_libraries(benchmark_multiblock PRIVATE gpuboscia::gpuboscia)
set_target_properties(benchmark_multiblock PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(benchmark_lmo benchmark_lmo.cu)
target_link_libraries(benchmark_lmo PRIVATE gpuboscia::gpuboscia)
set_target_properties(benchmark_lmo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

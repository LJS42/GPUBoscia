cmake_minimum_required(VERSION 3.20)
project(fastbb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Threads     REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB        REQUIRED)

# Override this at configure time when targeting another GPU, for example
# -DFASTBB_CUDA_ARCH=89.
set(FASTBB_CUDA_ARCH "80" CACHE STRING "CUDA SM architecture for FastBB")

# ---- main binary ----
add_executable(fastbb src/main.cu)
target_include_directories(fastbb PRIVATE include)
target_link_libraries(fastbb Threads::Threads CUDA::cudart ZLIB::ZLIB)
target_compile_definitions(fastbb PRIVATE
    FASTBB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(fastbb PROPERTIES
    CUDA_ARCHITECTURES "${FASTBB_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON)

# ---- discrete sparse Sharpe-ratio portfolio example ----
add_executable(sparse_sharpe src/sparse_sharpe.cu)
target_include_directories(sparse_sharpe PRIVATE include)
target_link_libraries(sparse_sharpe PRIVATE
    Threads::Threads CUDA::cudart ZLIB::ZLIB)
target_compile_definitions(sparse_sharpe PRIVATE
    FASTBB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(sparse_sharpe PROPERTIES
    CUDA_ARCHITECTURES "${FASTBB_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON)

# ---- fixed-cardinality maximum-weight induced-subgraph example ----
add_executable(weighted_clique src/weighted_clique.cu)
target_include_directories(weighted_clique PRIVATE include)
target_link_libraries(weighted_clique PRIVATE Threads::Threads CUDA::cudart)
set_target_properties(weighted_clique PROPERTIES
    CUDA_ARCHITECTURES "${FASTBB_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON)

# ---- tests ----
enable_testing()

# CPU-only unit tests (compiled with the C++ compiler, no CUDA required)
add_executable(test_cpu tests/test_cpu.cpp)
target_include_directories(test_cpu PRIVATE include)
target_link_libraries(test_cpu Threads::Threads)

# GPU pipeline integration tests (compiled with nvcc, requires a GPU at runtime)
add_executable(test_pipeline tests/test_pipeline.cu)
target_include_directories(test_pipeline PRIVATE include)
target_link_libraries(test_pipeline Threads::Threads CUDA::cudart)
set_target_properties(test_pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME cpu_tests      COMMAND test_cpu)
add_test(NAME pipeline_tests COMMAND test_pipeline)

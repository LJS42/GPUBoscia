add_executable(test_cpu test_cpu.cpp)
target_link_libraries(test_cpu PRIVATE gpuboscia::core)
set_target_properties(test_cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_test(NAME cpu_tests COMMAND test_cpu)

# Each CPU header must compile without incidental includes from another header.
file(GLOB _gpuboscia_cpu_headers "${PROJECT_SOURCE_DIR}/include/gpuboscia/*.h")
set(_gpuboscia_header_checks)
foreach(header IN LISTS _gpuboscia_cpu_headers)
    get_filename_component(name "${header}" NAME)
    set(source "${CMAKE_CURRENT_BINARY_DIR}/header_${name}.cpp")
    file(WRITE "${source}" "#include <gpuboscia/${name}>\n")
    list(APPEND _gpuboscia_header_checks "${source}")
endforeach()
add_library(test_header_includes OBJECT ${_gpuboscia_header_checks})
target_link_libraries(test_header_includes PRIVATE gpuboscia::core)

if(GPUBOSCIA_ENABLE_CUDA)
    add_executable(test_multiblock_lmo ../benchmarks/benchmark_lmo.cu)
    target_link_libraries(test_multiblock_lmo PRIVATE gpuboscia::gpuboscia)
    set_target_properties(test_multiblock_lmo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    add_test(NAME multiblock_lmo_tests COMMAND test_multiblock_lmo --test)
    set_tests_properties(multiblock_lmo_tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_executable(test_multiblock test_multiblock.cu)
    target_link_libraries(test_multiblock PRIVATE gpuboscia::gpuboscia)
    set_target_properties(test_multiblock PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    add_test(NAME multiblock_tests COMMAND test_multiblock)
    set_tests_properties(multiblock_tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_executable(test_pipeline test_pipeline.cu)
    target_link_libraries(test_pipeline PRIVATE gpuboscia::gpuboscia)
    set_target_properties(test_pipeline PROPERTIES
        CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    add_test(NAME pipeline_tests COMMAND test_pipeline)
    set_tests_properties(pipeline_tests PROPERTIES SKIP_RETURN_CODE 77)
    if(TARGET gpuboscia_minimal)
        add_test(NAME minimal_example COMMAND gpuboscia_minimal)
    endif()
    set(_gpuboscia_cuda_header_checks)
    file(GLOB _gpuboscia_cuda_headers "${PROJECT_SOURCE_DIR}/include/gpuboscia/*.cuh")
    foreach(header IN LISTS _gpuboscia_cuda_headers)
        get_filename_component(name "${header}" NAME)
        set(source "${CMAKE_CURRENT_BINARY_DIR}/header_${name}.cu")
        file(WRITE "${source}" "#include <gpuboscia/${name}>\n")
        list(APPEND _gpuboscia_cuda_header_checks "${source}")
    endforeach()
    add_library(test_cuda_header_includes OBJECT ${_gpuboscia_cuda_header_checks})
    target_link_libraries(test_cuda_header_includes PRIVATE gpuboscia::gpuboscia)
    set_target_properties(test_cuda_header_includes PROPERTIES
        CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}")
endif()

if(GPUBOSCIA_INSTALL)
    add_test(NAME package_consumer COMMAND ${CMAKE_COMMAND}
        "-DGPUBOSCIA_BUILD_DIR=${PROJECT_BINARY_DIR}"
        "-DGPUBOSCIA_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DGPUBOSCIA_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-test"
        "-DGPUBOSCIA_GENERATOR=${CMAKE_GENERATOR}"
        "-DGPUBOSCIA_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DGPUBOSCIA_CONFIG=$<CONFIG>"
        "-DGPUBOSCIA_TEST_CUDA=${GPUBOSCIA_ENABLE_CUDA}"
        "-DGPUBOSCIA_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        "-DGPUBOSCIA_CUDA_ARCH=${GPUBOSCIA_CUDA_ARCH}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
endif()

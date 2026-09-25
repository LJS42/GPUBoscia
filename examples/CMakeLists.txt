# Dependencies of the applications belong here, not on the solver library.
find_package(ZLIB REQUIRED)

function(gpuboscia_add_example target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE gpuboscia::gpuboscia)
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endfunction()

gpuboscia_add_example(gpuboscia_minimal minimal.cu)

# Preserve ./build/gpuboscia while freeing the gpuboscia target name for the library.
gpuboscia_add_example(gpuboscia_cli main.cu)
set_target_properties(gpuboscia_cli PROPERTIES OUTPUT_NAME gpuboscia)
target_link_libraries(gpuboscia_cli PRIVATE ZLIB::ZLIB)
target_compile_definitions(gpuboscia_cli PRIVATE GPUBOSCIA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

gpuboscia_add_example(sparse_sharpe sparse_sharpe.cu)
target_link_libraries(sparse_sharpe PRIVATE ZLIB::ZLIB)
target_compile_definitions(sparse_sharpe PRIVATE GPUBOSCIA_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

gpuboscia_add_example(weighted_clique weighted_clique.cu)

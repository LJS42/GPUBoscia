# Exercise the exported package after moving it away from the install prefix.
function(checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Command failed: ${ARGV}\n${out}\n${err}")
    endif()
endfunction()

# A unique scratch directory makes repeated/parallel tests independent.
string(RANDOM LENGTH 10 ALPHABET 0123456789abcdef suffix)
set(work "${GPUBOSCIA_TEST_DIR}/${suffix}")
file(MAKE_DIRECTORY "${work}")
checked("${CMAKE_COMMAND}" --install "${GPUBOSCIA_BUILD_DIR}"
    --prefix "${work}/original" --config "${GPUBOSCIA_CONFIG}")
file(RENAME "${work}/original" "${work}/relocated")
checked("${CMAKE_COMMAND}" -S "${GPUBOSCIA_SOURCE_DIR}/tests/consumer" -B "${work}/installed-consumer"
    -G "${GPUBOSCIA_GENERATOR}" "-DCMAKE_CXX_COMPILER=${GPUBOSCIA_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${work}/relocated")
checked("${CMAKE_COMMAND}" --build "${work}/installed-consumer" --config "${GPUBOSCIA_CONFIG}")
find_program(ctest_command NAMES ctest REQUIRED)
checked("${ctest_command}" --test-dir "${work}/installed-consumer" -C "${GPUBOSCIA_CONFIG}" --output-on-failure)

# Also exercise source-tree consumption with library defaults, without CUDA.
checked("${CMAKE_COMMAND}" -S "${GPUBOSCIA_SOURCE_DIR}/tests/consumer" -B "${work}/source-consumer"
    -G "${GPUBOSCIA_GENERATOR}" "-DCMAKE_CXX_COMPILER=${GPUBOSCIA_CXX_COMPILER}"
    "-DGPUBOSCIA_SOURCE_ROOT=${GPUBOSCIA_SOURCE_DIR}" -DGPUBOSCIA_ENABLE_CUDA=OFF)
checked("${CMAKE_COMMAND}" --build "${work}/source-consumer" --config "${GPUBOSCIA_CONFIG}")
checked("${ctest_command}" --test-dir "${work}/source-consumer" -C "${GPUBOSCIA_CONFIG}" --output-on-failure)

if(GPUBOSCIA_TEST_CUDA)
    checked("${CMAKE_COMMAND}" -S "${GPUBOSCIA_SOURCE_DIR}/tests/consumer_cuda" -B "${work}/cuda-consumer"
        -G "${GPUBOSCIA_GENERATOR}" "-DCMAKE_CXX_COMPILER=${GPUBOSCIA_CXX_COMPILER}"
        "-DCMAKE_CUDA_COMPILER=${GPUBOSCIA_CUDA_COMPILER}"
        "-DCMAKE_CUDA_ARCHITECTURES=${GPUBOSCIA_CUDA_ARCH}"
        "-DCMAKE_PREFIX_PATH=${work}/relocated"
        "-DGPUBOSCIA_EXAMPLE_SOURCE=${GPUBOSCIA_SOURCE_DIR}/examples/minimal.cu")
    checked("${CMAKE_COMMAND}" --build "${work}/cuda-consumer" --config "${GPUBOSCIA_CONFIG}")
    checked("${ctest_command}" --test-dir "${work}/cuda-consumer" -C "${GPUBOSCIA_CONFIG}" --output-on-failure)
else()
    execute_process(COMMAND "${CMAKE_COMMAND}"
        -S "${GPUBOSCIA_SOURCE_DIR}/tests/consumer" -B "${work}/missing-cuda"
        "-DCMAKE_PREFIX_PATH=${work}/relocated" -DGPUBOSCIA_CONSUMER_COMPONENT=CUDA
        RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(status EQUAL 0 OR NOT "${out}${err}" MATCHES "without CUDA")
        message(FATAL_ERROR "CPU-only installation must clearly reject CUDA requests: ${out}${err}")
    endif()
endif()

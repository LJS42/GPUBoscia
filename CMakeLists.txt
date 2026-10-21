cmake_minimum_required(VERSION 3.20)
project(GPUBoscia VERSION 0.1.0 LANGUAGES CXX)

# Do not enable CUDA or examples in projects that only need CPU utilities.
set(_gpuboscia_top_level OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(_gpuboscia_top_level ON)
endif()
option(GPUBOSCIA_ENABLE_CUDA "Provide the CUDA solver target" ON)
option(GPUBOSCIA_BUILD_BENCHMARKS "Build optional CUDA performance benchmarks" OFF)
option(GPUBOSCIA_BUILD_TESTS "Build GPUBoscia tests" ${_gpuboscia_top_level})
set(_gpuboscia_examples_default OFF)
if(_gpuboscia_top_level AND GPUBOSCIA_ENABLE_CUDA)
    set(_gpuboscia_examples_default ON)
endif()
option(GPUBOSCIA_BUILD_EXAMPLES "Build standalone optimization examples" ${_gpuboscia_examples_default})
option(GPUBOSCIA_INSTALL "Generate library installation rules" ${_gpuboscia_top_level})

find_package(Threads REQUIRED)
include(GNUInstallDirs)

# Templates are instantiated by the consuming project; there is no fixed .so/.a.
add_library(gpuboscia_core INTERFACE)
add_library(gpuboscia::core ALIAS gpuboscia_core)
set_target_properties(gpuboscia_core PROPERTIES EXPORT_NAME core)
target_compile_features(gpuboscia_core INTERFACE cxx_std_17)
target_include_directories(gpuboscia_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(gpuboscia_core INTERFACE Threads::Threads)

if(GPUBOSCIA_ENABLE_CUDA)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(_gpuboscia_arch_default "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(_gpuboscia_arch_default "80")
    endif()
    set(GPUBOSCIA_CUDA_ARCH "${_gpuboscia_arch_default}" CACHE STRING
        "CUDA architectures for GPUBoscia examples and tests")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${GPUBOSCIA_CUDA_ARCH}")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(gpuboscia INTERFACE)
    add_library(gpuboscia::gpuboscia ALIAS gpuboscia)
    target_compile_features(gpuboscia INTERFACE cuda_std_17)
    target_link_libraries(gpuboscia INTERFACE gpuboscia::core CUDA::cudart)
endif()

if(GPUBOSCIA_BUILD_EXAMPLES)
    if(NOT GPUBOSCIA_ENABLE_CUDA)
        message(FATAL_ERROR "GPUBOSCIA_BUILD_EXAMPLES requires GPUBOSCIA_ENABLE_CUDA=ON")
    endif()
    add_subdirectory(examples)
endif()

if(GPUBOSCIA_BUILD_BENCHMARKS)
    if(NOT GPUBOSCIA_ENABLE_CUDA)
        message(FATAL_ERROR "GPUBOSCIA_BUILD_BENCHMARKS requires GPUBOSCIA_ENABLE_CUDA=ON")
    endif()
    add_subdirectory(benchmarks)
endif()

if(GPUBOSCIA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GPUBOSCIA_INSTALL)
    include(CMakePackageConfigHelpers)
    set(_gpuboscia_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/GPUBoscia")
    install(DIRECTORY include/gpuboscia DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS gpuboscia_core EXPORT GPUBosciaCoreTargets)
    install(EXPORT GPUBosciaCoreTargets FILE GPUBosciaCoreTargets.cmake
        NAMESPACE gpuboscia:: DESTINATION ${_gpuboscia_package_dir})
    if(GPUBOSCIA_ENABLE_CUDA)
        install(TARGETS gpuboscia EXPORT GPUBosciaTargets)
        install(EXPORT GPUBosciaTargets FILE GPUBosciaTargets.cmake
            NAMESPACE gpuboscia:: DESTINATION ${_gpuboscia_package_dir})
    endif()
    configure_package_config_file(cmake/GPUBosciaConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/GPUBosciaConfig.cmake"
        INSTALL_DESTINATION ${_gpuboscia_package_dir})
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/GPUBosciaConfigVersion.cmake"
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GPUBosciaConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/GPUBosciaConfigVersion.cmake"
        DESTINATION ${_gpuboscia_package_dir})
endif()

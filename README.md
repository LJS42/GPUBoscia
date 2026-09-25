# GPUBoscia

GPUBoscia is a C++17/CUDA branch-and-bound library that uses Frank–Wolfe or
Decomposition-Invariant Conditional Gradient (DICG) methods to solve node
relaxations. Supply an objective, variable bounds, integer variable indices, and
a linear minimization oracle (LMO), then call `gpuboscia::solve(...)` to obtain a
solution, objective value, lower bound, and solver statistics.

The solver is a **header-only library**. Link the CMake target
`gpuboscia::gpuboscia` and instantiate its templates in your CUDA translation
units. There is no precompiled shared library or Python binding.

## Quick start

Requirements:

- CMake 3.20 or newer.
- A CUDA Toolkit and host compiler supporting C++17.
- An NVIDIA GPU to run the solver.
- ZLIB when building all examples. The quantization example also uses cuBLAS,
  supplied with the CUDA Toolkit.

Replace `80` with the CUDA architecture of your GPU:

```bash
cmake -S . -B build -DGPUBOSCIA_CUDA_ARCH=80
cmake --build build -j 4
./build/gpuboscia_minimal
```

[examples/minimal.cu](examples/minimal.cu) is a complete two-variable binary
quadratic example. Its expected solution is `[0, 1]`, with objective `0.18`.
The example itself does not require a dataset, PyTorch, or cuBLAS.

## Use in another project

### Add the source directory

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyApp LANGUAGES CXX CUDA)

add_subdirectory(/path/to/GPUBoscia gpuboscia-build)
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpuboscia::gpuboscia)
set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

As a subproject, GPUBoscia disables its examples, tests, and installation rules
by default. Include directories, C++/CUDA 17 requirements, Threads, and the CUDA
runtime are propagated through the library target. Select the application's GPU
architecture when configuring it, for example with
`-DCMAKE_CUDA_ARCHITECTURES=80`.

### Install and use find_package

Configure and install only the library:

```bash
cmake -S . -B build-library \
  -DGPUBOSCIA_BUILD_EXAMPLES=OFF \
  -DGPUBOSCIA_BUILD_TESTS=OFF \
  -DGPUBOSCIA_CUDA_ARCH=80
cmake --install build-library --prefix "$HOME/.local/gpuboscia"
```

As a template library, it can be installed immediately after configuration.
The installation contains headers and CMake package files, without datasets,
example executables, tests, or Python dependencies. Library-only configuration
does not require ZLIB or link cuBLAS.

In the consuming project:

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyApp LANGUAGES CXX CUDA)

find_package(GPUBoscia 0.1 CONFIG REQUIRED)
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpuboscia::gpuboscia)
set_target_properties(my_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

Configure and build the consuming project:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/.local/gpuboscia" \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build
```

Include the public entry point in `main.cu`:

```cpp
#include <gpuboscia/gpuboscia.cuh>

// Construct a Problem, device objective, and LMO as in examples/minimal.cu.
// auto result = gpuboscia::solve(problem, objective, lmo, settings);
```

The entry points are `gpuboscia::Settings`, `gpuboscia::Result`, and
`gpuboscia::solve`. Existing types such as `Problem`, `BoxLMO`, and
`LineSearchKind` retain their global names. GPU solver headers must be compiled
as CUDA code. Objectives still specify dimensions at compile time; packaging
the solver does not remove that constraint.

## Configuration

| Option | Standalone default | Purpose |
|---|---|---|
| `GPUBOSCIA_ENABLE_CUDA` | `ON` | Provide the GPU solver target `gpuboscia::gpuboscia` |
| `GPUBOSCIA_BUILD_EXAMPLES` | `ON` when CUDA is enabled | Build standalone examples |
| `GPUBOSCIA_BUILD_TESTS` | `ON` | Build tests |
| `GPUBOSCIA_INSTALL` | `ON` | Generate installation and package export rules |
| `GPUBOSCIA_CUDA_ARCH` | Existing `CMAKE_CUDA_ARCHITECTURES`, otherwise `80` | GPU architectures for this project's examples and tests |
| `GPUBOSCIA_QUANTIZATION_WEIGHTS` | `4096` | Weights per row in the quantization example; use `576` for ResNet20 |

The `gpuboscia::core` target provides CPU utilities and the Threads dependency.
To build and test without CUDA:

```bash
cmake -S . -B build-cpu -DGPUBOSCIA_ENABLE_CUDA=OFF
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure
```

For a CPU-only installation, use
`find_package(GPUBoscia CONFIG REQUIRED COMPONENTS Core)` and link
`gpuboscia::core`. This target does not provide a CPU replacement for the full
GPU solver. A plain `find_package(GPUBoscia)` requires the CUDA component and
reports an error for CPU-only installations. A CUDA-enabled installation also
supports requesting only `Core`, without looking for a CUDA Toolkit.

## Repository layout

```text
include/gpuboscia/    Solver headers and general utilities
cmake/               Installed CMake package configuration
examples/            Standalone optimization programs
examples/common/     Example-specific command-line parsing
experiments/         ResNet20 data export, solving, and evaluation scripts
tests/               Algorithm, header inclusion, and package consumption tests
DataSets/            Existing experimental datasets
test_instances/      Existing optimization instances
```

The command-line application is built with target `gpuboscia_cli` and runs as
`./build/gpuboscia`. Other example targets include `gpuboscia_minimal`,
`sparse_sharpe`, `weighted_clique`, and `quantization_sparse`:

```bash
cmake --build build --target quantization_sparse
./build/quantization_sparse --help
```

The quantization implementation is currently an example, rather than a separate
library. See [experiments/README.md](experiments/README.md) for the ResNet20 workflow.

## Tests and validation

```bash
ctest --test-dir build --output-on-failure
```

Tests cover the existing CPU and GPU algorithms, independent header inclusion,
and package consumption. Package tests relocate an installation before building
a consumer with two translation units, and also check source-directory
consumption. With CUDA enabled, they build and run the minimal solver example
against the installed package.

CPU builds, CPU tests, header checks, and package consumption were verified in
the development environment. CUDA compilation and GPU execution still need
validation on a CUDA machine.

## Renaming from FastBB

The package is now `GPUBoscia`, the C++ namespace and include directory are
`gpuboscia`, and configuration options use the `GPUBOSCIA_` prefix. Update old
includes, CMake targets, and option names, and configure a fresh build directory.
The previous names are not compatibility aliases. The local checkout directory
can keep its existing name.

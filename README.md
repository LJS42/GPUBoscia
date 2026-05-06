# Boscia_FWonGPU.jl

GPU-accelerated Branch-and-Bound with Frank-Wolfe methods, built on top of [Boscia.jl](https://github.com/ZIB-IOL/Boscia.jl) and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

## Requirements

### Hardware
- An NVIDIA GPU is **required**. There is currently no CPU fallback.

### Software
- Julia ≥ 1.9
- A working NVIDIA driver (check with `nvidia-smi`)
- CUDA toolkit 12.6 (managed automatically by CUDA.jl, but the driver must be installed at the system level)

## Environment Setup

This project requires a conda environment with CUDA toolkit 12.6, because the system may not have CUDA installed globally.

```bash
conda create -n julia_cuda
conda activate julia_cuda
conda install -c nvidia cuda-toolkit=12.6
```

Then set the following environment variables so Julia can find the toolkit:

```bash
export CUDA_HOME=/path/to/conda/envs/julia_cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Always activate this environment before starting Julia:**

```bash
conda activate julia_cuda
julia --project=. --threads=4 -e 'include("examples/DICG_Cube.jl")'
```

## Installation

Clone the repo and instantiate the Julia environment:

```bash
git clone https://github.com/yourname/Boscia_FWonGPU.jl
cd Boscia_FWonGPU.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

On first run, CUDA.jl will download the appropriate CUDA toolkit automatically. You can also set the version explicitly:

```julia
using CUDA
CUDA.set_runtime_version!(v"12.6")
```

## Verify your setup

```julia
using CUDA
CUDA.functional()     # should return true
CUDA.versioninfo()    # shows driver and toolkit versions
```

If `CUDA.functional()` returns `false`, check:
- `nvidia-smi` works in your terminal
- Your NVIDIA driver version is compatible with CUDA 12.6 (driver ≥ 525.60)

## Running the examples

```bash
julia --project=. --threads=4 -e 'include("examples/DICG_Cube.jl")'
```

## Troubleshooting

**`CUDA driver is missing` or `CUDA.functional() == false`**

This means Julia cannot find a working NVIDIA driver. This is a system-level issue, not a Julia package issue. Possible causes:

- No NVIDIA GPU is present
- The NVIDIA driver is not installed — install it from [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers)
- You are running inside a container or VM without GPU passthrough

**Slow first run**

Julia compiles code on first execution (JIT). The first run will be significantly slower than subsequent ones. This is expected.

**Out of memory**

The examples allocate large GPU arrays. If you run out of VRAM, try reducing `m` in the example scripts.

**Out of memory**

The examples allocate large GPU arrays. If you run out of VRAM, try reducing `m` in the example scripts.

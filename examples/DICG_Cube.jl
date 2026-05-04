import Adapt
using FrankWolfe
using LinearAlgebra
using Random
using Boscia
using Bonobo
using CUDA
#CUDA.set_runtime_version!(v"12.6")
using StableRNGs
include("../src/Branching_nodes.jl")
include("../src/BLMO.jl")
include("../src/Managed_blmo.jl")
include("../src/Initialize.jl")
include("../src/StandardFrankWolfe.jl")
include("../src/DICG.jl")

println("\nNonlinear Example - GPU")
seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# ── obj ──────────────────────────────────────────
n = 256
m = 70000

const X = rand(rng, Float32, n, m)
const X_gpu = CuArray(X)
const Y = rand(rng, Float32, n, m)
const Y_gpu = CuArray(Y)

int_vars = collect(1:n*n)
lbs = fill(-128.0, n*n)
ubs = fill(128.0, n*n)

blmo = Boscia.CubeLMO(lbs, ubs, int_vars)
lmo = Boscia.ManagedBoundedLMO(blmo, lbs, ubs, int_vars, n*n)

const F_buf_cpu = zeros(Float32, n, m)

function f(w)
    mul!(F_buf_cpu, reshape(w, n, n), X)
    @. F_buf_cpu -= Y
    return sum(abs2, F_buf_cpu)
end

const R_buf_gpu = CUDA.zeros(Float32, n, m)
const R_buf_cpu = zeros(Float32, n, m)

function grad!(storage, w)
    if w isa CuArray
        mul!(R_buf_gpu, reshape(w, n, n), X_gpu)
        @. R_buf_gpu -= Y_gpu
        mul!(reshape(storage, n, n), R_buf_gpu, X_gpu')
        @. storage *= 2f0
    else
        mul!(R_buf_cpu, reshape(w, n, n), X)
        @. R_buf_cpu -= Y
        mul!(reshape(storage, n, n), R_buf_cpu, X')
        @. storage *= 2f0
    end
end

# ── GPU variant ───────────────────────────────────────
variant = DecompositionInvariantConditionalGradientGPU(256)

# ── Solve ──────────────────────────────────────────────
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 500
settings.branch_and_bound[:time_limit] = 3000
settings.frank_wolfe[:variant] = variant
x, _, _ = Boscia.solve(f, grad!, lmo, settings=settings)

import Adapt
using FrankWolfe
using LinearAlgebra
using Random
using Boscia
using Bonobo
using CUDA
using Printf
#CUDA.set_runtime_version!(v"12.6")
using StableRNGs
include("../src/Branching_nodes.jl")
include("../src/BLMO.jl")
include("../src/Managed_blmo.jl")
include("../src/Initialize.jl")
include("../src/Linesearch.jl")
include("../src/StandardFrankWolfe.jl")
include("../src/DICG.jl")

println("\nNonlinear Example - GPU")
seed = rand(UInt64)
#0x5dd2363d3e3d4329
@show seed
rng = StableRNG(seed)

# ── obj ──────────────────────────────────────────
n = 8
m = 7000
const W_float = randn(rng, Float32, n, n)
const X       = rand(rng, Float32, n, m)
const Y       = W_float * X .+ 0.01f0 * randn(rng, Float32, n, m)
const X_gpu   = CuArray(X)
const Y_gpu   = CuArray(Y)

const QMIN       = 0.0
const QMAX       = 15.0
const NUM_LEVELS = QMAX - QMIN + 1

int_vars = collect(1:n*n)
lbs      = fill(QMIN, n*n)
ubs      = fill(QMAX, n*n)

const w_min = Float64(minimum(W_float))
const w_max = Float64(maximum(W_float))
const scale = (w_max - w_min) / (NUM_LEVELS - 1)

blmo = Boscia.CubeLMO(lbs, ubs, int_vars)
lmo  = Boscia.ManagedBoundedLMO(blmo, lbs, ubs, int_vars, n*n)

const R_buf_f_gpu = CUDA.zeros(Float32, n, m)
const R_buf_g_gpu = CUDA.zeros(Float32, n, m)
const G_buf_gpu   = CUDA.zeros(Float32, n, n)
const R_buf_f = zeros(Float32, n, m)
const R_buf_g = zeros(Float32, n, m)
const G_buf   = zeros(Float32, n, n)

const q_buf_gpu = CUDA.zeros(Float32, n*n)

function dequantize(q_vec)
    if q_vec isa CuArray
        return Float32(scale) .* reshape(q_vec, n, n) .+ Float32(w_min)
    else
        return Float32(scale) .* reshape(q_vec, n, n) .+ Float32(w_min)
    end
end

function f(q)
    W_q = dequantize(q)
    if W_q isa CuArray
        mul!(R_buf_f_gpu, W_q, X_gpu)
        @. R_buf_f_gpu -= Y_gpu
        return Float64(sum(abs2, R_buf_f_gpu)) / (n * m)  
    else
        mul!(R_buf_f, W_q, X)
        @. R_buf_f -= Y
        return Float64(sum(abs2, R_buf_f)) / (n * m)        
    end
end

function grad!(storage, q)
    W_q = dequantize(q)
    if W_q isa CuArray
        mul!(R_buf_g_gpu, W_q, X_gpu)
        @. R_buf_g_gpu -= Y_gpu
        mul!(G_buf_gpu, R_buf_g_gpu, X_gpu')
        @. G_buf_gpu *= Float32(2.0 * scale / (n * m))
        storage .= vec(G_buf_gpu) 
    else
        mul!(R_buf_g, W_q, X)
        @. R_buf_g -= Y
        mul!(G_buf, R_buf_g, X')
        @. G_buf *= Float32(2.0 * scale / (n * m))
        storage .= Float64.(vec(G_buf))
    end
end

# function my_start_point(lmo, active_set)
#     x0  = FrankWolfe.get_active_set_iterate(active_set)
#     dim = length(x0)
#     g   = zeros(Float64, dim)
#     grad!(g, x0)
#     v        = FrankWolfe.compute_extreme_point(lmo, g)
#     dual_gap = dot(g, x0 - v)
#     primal   = f(x0)
#     dual     = primal - dual_gap
#     return v, dual
# end

# ── GPU variant ───────────────────────────────────────
variant = DecompositionInvariantConditionalGradientGPU(256)

# ── Solve ──────────────────────────────────────────────
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 50
settings.branch_and_bound[:time_limit] = 7200
settings.frank_wolfe[:variant] = variant
settings.frank_wolfe[:line_search] = GPUSecant()
settings.frank_wolfe[:max_fw_iter] = 10000
x, _, _ = Boscia.solve(f, grad!, lmo, settings=settings)

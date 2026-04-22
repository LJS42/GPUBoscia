import Adapt
using FrankWolfe
using LinearAlgebra
using Random
using Boscia
using CUDA
#CUDA.set_runtime_version!(v"12.6")
using StableRNGs
include("../src/BLMO.jl")
include("../src/Managed_blmo.jl")
include("../src/StandardFrankWolfe.jl")

println("\nNonlinear Example - GPU")
seed = rand(UInt64)
@show seed
rng = StableRNG(seed)
n = 20000

# ── LMO ──────────────────────────────────────────────
int_vars = collect(1:Int(n/2))

bounds = Boscia.IntegerBounds()

lbs = zeros(Int(n/2))
ubs = ones(Int(n/2))
blmo = Boscia.CubeLMO(zeros(n), ones(n), int_vars)

lmo = Boscia.ManagedBoundedLMO(blmo, lbs[int_vars], ubs[int_vars], int_vars, n)

# ── obj ──────────────────────────────────────────
const A_gpu = let
    M_gpu = CUDA.randn(Float32, n, n)
    A = M_gpu' * M_gpu  
    A
end
const A = Array(Float64.(A_gpu))  

const y = rand(rng, Bool, n) * 0.6 .+ 0.3
const y_gpu = CuArray(Float32.(y))

function f(x)
    d = x .- y
    return dot(d, A * d)
end

x_gpu_buf = CUDA.zeros(Float32, n)  

function grad!(storage, x)
    if x isa CuArray
        mul!(storage, A_gpu, x)
        return mul!(storage, A_gpu, y_gpu, -2f0, 2f0)
    else
        println("warning: compute on cpu")
        mul!(storage, A, x)
        return mul!(storage, A, y, -2.0, 2.0)
    end
end

# ── GPU variant ───────────────────────────────────────
variant = StandardFrankWolfeGPU(256)

# ── Solve ──────────────────────────────────────────────
depth = 5
heu = Boscia.Heuristic(
    (tree, blmo, x) -> Boscia.follow_gradient_heuristic(tree, blmo, x, depth),
    0.8,
    :follow_gradient,
)
heu2 = Boscia.Heuristic(Boscia.rounding_lmo_01_heuristic, 0.8, :lmo_rounding)

heuristics = [heu, heu2]
# heuristics = []

settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 500
settings.branch_and_bound[:time_limit] = 300
settings.heuristic[:custom_heuristics] = heuristics
settings.frank_wolfe[:variant] = variant
x, _, _ = Boscia.solve(f, grad!, lmo, settings=settings)

show(to)
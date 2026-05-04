using LinearAlgebra
using Random
using Boscia
using StableRNGs
using SCIP
using FrankWolfe
import MathOptInterface

const MOI = MathOptInterface


seed = rand(UInt64)
@show seed
rng = StableRNG(seed)

# ── data ──────────────────────────────────────────
n = 8
m = 70000

const X = rand(rng, Float32, n, m)
const Y = rand(rng, Float32, n, m)

int_vars = collect(1:n*n)
lbs = fill(-128.0, n*n)
ubs = fill(128.0, n*n)

blmo = Boscia.CubeLMO(lbs, ubs, int_vars)
lmo = Boscia.ManagedBoundedLMO(blmo, lbs, ubs, int_vars, n*n)

int_vars = collect(1:n)
lbs = fill(0.0, n)
ubs = fill(1.0, n)

blmo = Boscia.CubeLMO(lbs, ubs, int_vars)
lmo = Boscia.ManagedBoundedLMO(blmo, lbs, ubs, int_vars, n)

# ── obj ──────────────────────────────────────
function f(w)
    W = reshape(w, n, n)
    return norm(W * X .- Y)^2
end

const R_buf = zeros(Float32, n, m)

function grad!(storage, w)
    mul!(R_buf, reshape(w, n, n), X)
    @. R_buf -= Y
    mul!(reshape(storage, n, n), R_buf, X')
    @. storage *= 2f0
end

# const diffw = rand(n,)

# function f(x)
#     return sum(0.5*(x.-diffw).^2)
# end

# function grad!(storage, x)
#     @. storage = x-diffw
# end

# variant = Boscia.StandardFrankWolfe()

# ── Solve ──────────────────────────────────────────────
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 500
settings.branch_and_bound[:time_limit] = 1200
settings.frank_wolfe[:variant] = variant

x, _, _ = Boscia.solve(f, grad!, lmo, settings=settings)

# n = 64

# const diffw = 0.5 * ones(n)
# o = SCIP.Optimizer()

# MOI.set(o, MOI.Silent(), true)

# x = MOI.add_variables(o, n)

# for xi in x
#     MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
#     MOI.add_constraint(o, xi, MOI.LessThan(1.0))
#     MOI.add_constraint(o, xi, MOI.ZeroOne())
# end

# lmo = FrankWolfe.MathOptLMO(o)

# function f(x)
#     return sum(0.5*(x.-diffw).^2)
# end

# function grad!(storage, x)
#     @. storage = x-diffw
# end

# settings = Boscia.create_default_settings()
# settings.branch_and_bound[:verbose] = true
# x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)

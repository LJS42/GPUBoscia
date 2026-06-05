using LinearAlgebra
using Random
using Boscia
using StableRNGs
using FrankWolfe


seed = 0x5dd2363d3e3d4329
@show seed
rng = StableRNG(seed)

# ── data ──────────────────────────────────────────
n = 128
m = 700

const X = rand(rng, Float32, n, m)
const Y = rand(rng, Float32, n, m)

int_vars = collect(1:n*n)
lbs = fill(-8.0, n*n)
ubs = fill(7.0, n*n)

blmo = Boscia.CubeLMO(lbs, ubs, int_vars)
lmo = Boscia.ManagedBoundedLMO(blmo, lbs, ubs, int_vars, n*n)

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

variant = Boscia.DecompositionInvariantConditionalGradient()

# ── Solve ──────────────────────────────────────────────
settings = Boscia.create_default_settings()
settings.branch_and_bound[:verbose] = true
settings.branch_and_bound[:print_iter] = 50
settings.branch_and_bound[:time_limit] = 3600
settings.frank_wolfe[:max_fw_iter] = 222222
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

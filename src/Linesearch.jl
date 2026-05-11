"""
    GPUSecant

GPU version of FrankWolfe.Secant. Exact same logic, but:
  - storage .= x .+ gamma .* d  (GPU, no muladd_memory_mode)
  - grad!(grad_storage, storage) (GPU matmul)
  - dot(grad_storage, d)         (GPU reduction → one scalar pull)

All other logic (secant update, convergence check, best tracking) is
identical to the original Secant implementation.
"""
struct GPUSecant <: FrankWolfe.LineSearchMethod
    limit_num_steps :: Int
    tol             :: Float64
    gamma_max       :: Float32
end

function GPUSecant(;
    limit_num_steps = 40,
    tol             = 1e-6,
    gamma_max       = 1f0,
)
    return GPUSecant(limit_num_steps, tol, gamma_max)
end

Base.print(io::IO, ::GPUSecant) = print(io, "GPUSecant")

mutable struct GPUSecantWorkspace
    storage      :: CuArray{Float32, 1}
    grad_storage :: CuArray{Float32, 1}
    last_gamma   :: Float64
end

function FrankWolfe.build_linesearch_workspace(
    ls       :: GPUSecant,
    x        :: CuArray,
    gradient :: CuArray,
)
    dim = length(x)
    return GPUSecantWorkspace(
        CUDA.zeros(Float32, dim),
        CUDA.zeros(Float32, dim),
        1.0,
    )
end

function FrankWolfe.perform_line_search(
    ls        :: GPUSecant,
    _,
    f,
    grad!,
    gradient  :: CuArray,
    x         :: CuArray,
    d         :: CuArray,
    gamma_max :: Real,
    workspace :: GPUSecantWorkspace,
)
    # ── exact copy of Secant, GPU ops only ───────────────────────────────────

    dot_gdir = dot(gradient, d)
    gamma = min(workspace.last_gamma, gamma_max)

    # initial storage = x + gamma * d  (GPU, replaces muladd_memory_mode)
    workspace.storage .= x .- gamma .* d

    best_gamma = gamma
    best_val   = f(workspace.storage)
    i          = 1
    gamma_prev = zero(gamma)
    clamping   = false

    if dot_gdir <= ls.tol
        return zero(gamma)
    end

    while abs(dot_gdir) > ls.tol
        if i > ls.limit_num_steps
            workspace.last_gamma = best_gamma
            break
        end

        # GPU matmul
        grad!(workspace.grad_storage, workspace.storage)

        # GPU reduction → one scalar pull
        dot_gdir_new = dot(workspace.grad_storage, d)

        if dot_gdir_new ≈ dot_gdir
            clamping = true
            workspace.last_gamma = best_gamma
            break
        end

        # secant update — same formula as original
        gamma_new  = gamma - dot_gdir_new * (gamma - gamma_prev) / (dot_gdir_new - dot_gdir)
        gamma_prev = gamma
        gamma      = clamp(gamma_new, 0, gamma_max)

        # GPU update
        workspace.storage .= x .- gamma .* d
        new_val = f(workspace.storage)

        if new_val < best_val
            best_val   = new_val
            best_gamma = gamma
        end

        dot_gdir = dot_gdir_new
        i += 1
    end

    workspace.last_gamma = best_gamma
    return best_gamma
end
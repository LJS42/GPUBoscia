struct DecompositionInvariantConditionalGradientGPU <: Boscia.FrankWolfeVariant
    threads :: Int
end

function Boscia.solve_frank_wolfe(
    frank_wolfe_variant::DecompositionInvariantConditionalGradientGPU,
    f,
    grad!,
    managed_lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=GPUSecant(),
    epsilon=1e-7,
    max_iteration=2222,
    add_dropped_vertices=false,
    use_extra_vertex_storage=false,
    extra_vertex_storage=nothing,
    callback=nothing,
    lazy=false,
    lazy_tolerance=2.0,
    timeout=Inf,
    verbose=false,
    workspace=nothing,
    pre_computed_set=nothing,
    kwargs...,
)
    # If the flag away_steps is set to false, away_frank_wolfe performs Vanilla.
    # Observe that the lazy flag is only observed if away_steps is set to true, so it can neglected. 
    # ── Setup ─────────────────────────────────────────────
    x0 = gpu_start_point_initialize(
        managed_lmo,
        active_set,
        pre_computed_set,
    )
    dim = length(x0)

    pre_computed_set = Vector{Vector{Float64}}()
 
    @assert frank_wolfe_variant.threads % 32 == 0 "threads must be a multiple of 32"
    @assert frank_wolfe_variant.threads <= 1024   "threads cannot exceed 1024"
 
    threads = frank_wolfe_variant.threads
    blocks  = cld(dim, threads)
    n_warps = div(threads + 31, 32)
    shmem   = n_warps * sizeof(Float32)
 
    x_gpu   = CuArray(Float32.(x0))
    g_gpu   = CUDA.zeros(Float32, dim)
    gap_gpu = CUDA.zeros(Float32, 1)
    x_old_gpu = CUDA.zeros(Float32, dim)
    blmo = managed_lmo.lmo
 
    status = :MAXITER
    k = 0

    eff_lb_gpu, eff_ub_gpu = build_effective_bounds(blmo, dim)
    v_gpu = CUDA.zeros(Float32, dim)
    a_gpu = CUDA.zeros(Float32, dim)
    d_gpu = CUDA.zeros(Float32, dim)
    gamma_gpu = CUDA.fill(reinterpret(UInt32, Inf32), 1)

    ls_workspace = FrankWolfe.build_linesearch_workspace(line_search, x_gpu, g_gpu)
 
    # ── Main loop ─────────────────────────────────────────
    diff_gpu = CUDA.zeros(Float32, dim)
    gap_cpu  = Ref(Inf32) 
    check_interval = 111

    prev_gap = Inf32
    gap_ready = false

    t = nothing

    while k < max_iteration
        k += 1
        fill!(gamma_gpu, reinterpret(UInt32, Inf32))
        grad!(g_gpu, x_gpu)
        launch_kernel_DICG!(v_gpu, a_gpu, d_gpu, gamma_gpu, blmo.lmo, g_gpu, x_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
        γ_max = reinterpret(Float32, CUDA.@allowscalar gamma_gpu[1])
        gamma = FrankWolfe.perform_line_search(line_search, k, f, grad!, g_gpu, x_gpu, d_gpu, γ_max, ls_workspace)
        γ = min(gamma, γ_max)
        x_gpu .= x_gpu .- γ .* d_gpu

        if mod(k, check_interval) == 0

            push!(pre_computed_set, Float64.(Array(v_gpu)))
            if gap_ready && prev_gap < Float32(epsilon)
                status = :OPTIMAL
                break
            end

            if gap_ready && prev_gap < Float32(epsilon)
                status = :OPTIMAL
                break
            end

            t = @async begin
                CUDA.@sync begin
                    diff_gpu .= x_gpu .- v_gpu
                    prev_gap  = Float32(dot(diff_gpu, g_gpu))
                end
            end
            gap_ready = true
        end
    end

    if gap_ready
        wait(t)
    end
 
    # ── Final verification ────────────────────────────────
    # Recompute everything once more for accurate final values
    grad!(g_gpu, x_gpu)
    launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
    dual_gap = Float64(dot(x_gpu .- v_gpu, g_gpu))
    x_out    = Float64.(Array(x_gpu))
    map!(v -> v == 0.0 ? 0.0 : v, x_out, x_out)
    primal   = f(x_out)
 
    # ── Return ────────────────────────────────────────────
    # Wrap result in a single-atom active set (no active set management)
    return x_out, primal, dual_gap, status, pre_computed_set
 
end

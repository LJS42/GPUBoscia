struct DecompositionInvariantConditionalGradientGPU <: Boscia.DecompositionInvariant
    threads       :: Int
    check_interval :: Int
    use_strong_lazy::Bool
    use_warm_start::Bool
    use_strong_warm_start::Bool
    DecompositionInvariantConditionalGradientGPU(threads;
        check_interval        = 1,
        use_strong_lazy       = false,
        use_warm_start        = false,
        use_strong_warm_start = false,
    ) = new(threads, check_interval, use_strong_lazy, use_warm_start, use_strong_warm_start)
end

function Boscia.solve_frank_wolfe(
    frank_wolfe_variant::DecompositionInvariantConditionalGradientGPU,
    f,
    grad!,
    managed_lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=GPUSecant(),
    epsilon=1e-4,
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
    # ── Setup ─────────────────────────────────────────────

    x0 = FrankWolfe.get_active_set_iterate(active_set)

    dim = length(x0)

    threads        = frank_wolfe_variant.threads
    check_interval = frank_wolfe_variant.check_interval
    blocks         = cld(dim, threads)

    blmo     = managed_lmo.lmo

    eff_lb_gpu, eff_ub_gpu = build_effective_bounds(blmo, dim)

    x_gpu  = gpu_start_point_initialize(managed_lmo, active_set, pre_computed_set, eff_lb_gpu, eff_ub_gpu, threads, blocks)

    pre_computed_set = Vector{Vector{Float64}}()

    @assert frank_wolfe_variant.threads % 32 == 0 "threads must be a multiple of 32"
    @assert frank_wolfe_variant.threads <= 1024   "threads cannot exceed 1024"
    @assert (frank_wolfe_variant.threads & (frank_wolfe_variant.threads - 1)) == 0 "threads must be a power of 2"

    threads        = frank_wolfe_variant.threads
    check_interval = frank_wolfe_variant.check_interval
    blocks_pow2    = nextpow(2, blocks)

    shmem_reduce = threads     * sizeof(Float32)
    shmem_check  = blocks_pow2 * sizeof(Float32)

    g_gpu    = CUDA.zeros(Float32, dim)
    gap_gpu  = CUDA.zeros(Float32, 1)
    diff_gpu = CUDA.zeros(Float32, dim)

    status = :MAXITER
    k      = 0

    v_gpu     = CUDA.zeros(Float32, dim)
    a_gpu     = CUDA.zeros(Float32, dim)
    d_gpu     = CUDA.zeros(Float32, dim)
    gamma_gpu = CUDA.fill(reinterpret(UInt32, Inf32), 1)

    status_gpu  = CUDA.zeros(Int32,   1)
    partial_gpu = CUDA.zeros(Float32, blocks)

    ls_workspace = FrankWolfe.build_linesearch_workspace(line_search, x_gpu, g_gpu)

    num_pre_compute = 10
    time_start      = time_ns()
    lmo_time_acc    = Ref(0.0)

    # ── Main loop ─────────────────────────────────────────
    while k < max_iteration
        k += 1

        fill!(gamma_gpu, reinterpret(UInt32, Inf32))
        grad!(g_gpu, x_gpu)

        do_check = mod(k, check_interval) == 0

        # ── LMO call ──────────────────────────────────────
        if callback !== nothing
            t_lmo = time_ns()
            launch_kernel_DICG!(
                v_gpu, a_gpu, d_gpu, gamma_gpu,
                blmo.lmo, g_gpu, x_gpu,
                eff_lb_gpu, eff_ub_gpu,
                threads, blocks,
            )
            CUDA.synchronize()
            lmo_time_acc[] += (time_ns() - t_lmo) / 1e9
        else
            launch_kernel_DICG!(
                v_gpu, a_gpu, d_gpu, gamma_gpu,
                blmo.lmo, g_gpu, x_gpu,
                eff_lb_gpu, eff_ub_gpu,
                threads, blocks,
            )
        end
        managed_lmo.ncalls += 2

        γ_max = reinterpret(Float32, CUDA.@allowscalar gamma_gpu[1])
        gamma = FrankWolfe.perform_line_search(
            line_search, k, f, grad!,
            g_gpu, x_gpu, d_gpu, γ_max, ls_workspace,
        )
        γ = min(gamma, γ_max)
        x_gpu .= x_gpu .- γ .* d_gpu

        @show γ

        # ── Gap check ────────────────────────────────────
        if do_check
            diff_gpu .= x_gpu .- v_gpu
            fill!(gap_gpu,    0.0f0)
            fill!(status_gpu, Int32(0))

            @cuda threads=threads     blocks=blocks  shmem=shmem_reduce reduce_dot_kernel!(
                partial_gpu, diff_gpu, g_gpu, dim)
            @cuda threads=blocks_pow2 blocks=1       shmem=shmem_check  check_gap_kernel!(
                status_gpu, gap_gpu, partial_gpu, blocks, blocks_pow2, Float32(epsilon))

            CUDA.@allowscalar begin
                if status_gpu[1] == Int32(1)
                    status = :OPTIMAL
                    break
                end
            end

            # ── Callback (Incumbent update) ───────────────
            if callback !== nothing
                CUDA.@allowscalar gap_val = Float64(gap_gpu[1])
                primal_val = f(x_gpu)
                tot_time   = (time_ns() - time_start) / 1e9
                v_out = Float64.(Array(v_gpu))
                g_out = Float64.(Array(g_gpu))
                state = FrankWolfe.CallbackState(
                    k,
                    primal_val,
                    primal_val - gap_val,
                    gap_val,
                    tot_time,
                    Float64.(Array(x_gpu)),
                    v_out,
                    nothing,
                    0.0,
                    f,
                    grad!,
                    managed_lmo,
                    g_out,
                    FrankWolfe.ST_PAIRWISE,
                )
                if callback(state, pre_computed_set) === false
                    status = :INTERRUPTED
                    break
                end
            end
        end

        # ── pre_computed_set ──────────────────────────────
        if max_iteration - k < num_pre_compute
            push!(pre_computed_set, Float64.(Array(v_gpu)))
        end

    end

    # ── Final verification ────────────────────────────────
    grad!(g_gpu, x_gpu)
    launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)

    diff_gpu .= x_gpu .- v_gpu
    dual_gap  = Float64(dot(diff_gpu, g_gpu))
    x_out     = Float64.(Array(x_gpu))
    map!(v -> v == 0.0 ? 0.0 : v, x_out, x_out)
    primal    = f(x_gpu)

    @show primal
    @show dual_gap

    # ── Final callback ────────────────────────────────────
    if callback !== nothing
        push!(managed_lmo.optimizing_times, lmo_time_acc[])
        tot_time = (time_ns() - time_start) / 1e9
        v_out = Float64.(Array(v_gpu))
        g_out = Float64.(Array(g_gpu))
        state = FrankWolfe.CallbackState(
            k,
            primal,
            primal - dual_gap,
            dual_gap,
            tot_time,
            x_out,
            v_out,
            nothing,
            0.0,
            f,
            grad!,
            managed_lmo,
            g_out,
            FrankWolfe.ST_LAST,
        )
        callback(state, pre_computed_set)
    end

    # lower_bound = primal - dual_gap
    # @show lower_bound

    # ── Return ────────────────────────────────────────────
    return x_out, primal, dual_gap, status, pre_computed_set
end

function reduce_dot_kernel!(partial, diff, g, n)
    tid = threadIdx().x
    idx = (blockIdx().x - Int32(1)) * blockDim().x + tid

    shmem = @cuDynamicSharedMem(Float32, blockDim().x)
    shmem[tid] = idx <= n ? diff[idx] * g[idx] : 0.0f0
    sync_threads()

    stride = blockDim().x >> 1
    while stride > 0
        if tid <= stride
            shmem[tid] += shmem[tid + stride]
        end
        sync_threads()
        stride >>= 1
    end

    if tid == 1
        partial[blockIdx().x] = shmem[1]
    end
    return nothing
end

function check_gap_kernel!(status_gpu, gap_gpu, partial, num_blocks, num_blocks_pow2, epsilon)
    tid = threadIdx().x

    shmem = @cuDynamicSharedMem(Float32, num_blocks_pow2)
    shmem[tid] = tid <= num_blocks ? partial[tid] : 0.0f0
    sync_threads()

    stride = num_blocks_pow2 >> 1
    while stride > 0
        if tid <= stride
            shmem[tid] += shmem[tid + stride]
        end
        sync_threads()
        stride >>= 1
    end

    if tid == 1
        gap_gpu[1] = shmem[1]
        if shmem[1] < epsilon
            status_gpu[1] = Int32(1)
        end
    end
    return nothing
end

# function Boscia.solve_frank_wolfe(
#     frank_wolfe_variant::DecompositionInvariantConditionalGradientGPU,
#     f,
#     grad!,
#     managed_lmo,
#     active_set;
#     line_search::FrankWolfe.LineSearchMethod=GPUSecant(),
#     epsilon=1e-7,
#     max_iteration=2222,
#     add_dropped_vertices=false,
#     use_extra_vertex_storage=false,
#     extra_vertex_storage=nothing,
#     callback=nothing,
#     lazy=false,
#     lazy_tolerance=2.0,
#     timeout=Inf,
#     verbose=false,
#     workspace=nothing,
#     pre_computed_set=nothing,
#     kwargs...,
# )
#     x0 = gpu_start_point_initialize(managed_lmo, active_set, pre_computed_set)
#     dim = length(x0)
#     pre_computed_set = Vector{Vector{Float64}}()

#     @assert frank_wolfe_variant.threads % 32 == 0 "threads must be a multiple of 32"
#     @assert frank_wolfe_variant.threads <= 1024   "threads cannot exceed 1024"

#     threads = frank_wolfe_variant.threads
#     blocks  = cld(dim, threads)
#     n_warps = div(threads + 31, 32)
#     shmem   = n_warps * sizeof(Float32)

#     x_gpu     = CuArray(Float32.(x0))
#     g_gpu     = CUDA.zeros(Float32, dim)
#     x_old_gpu = CUDA.zeros(Float32, dim)
#     blmo      = managed_lmo.lmo
#     status    = :MAXITER
#     k         = 0

#     eff_lb_gpu, eff_ub_gpu = build_effective_bounds(blmo, dim)
#     v_gpu     = CUDA.zeros(Float32, dim)
#     a_gpu     = CUDA.zeros(Float32, dim)
#     d_gpu     = CUDA.zeros(Float32, dim)
#     gamma_gpu = CUDA.fill(reinterpret(UInt32, Inf32), 1)
#     ls_workspace = FrankWolfe.build_linesearch_workspace(line_search, x_gpu, g_gpu)

#     diff_gpu       = CUDA.zeros(Float32, dim)
#     check_interval = 111

#     while k < max_iteration
#         k += 1
#         fill!(gamma_gpu, reinterpret(UInt32, Inf32))
#         grad!(g_gpu, x_gpu)
#         launch_kernel_DICG!(v_gpu, a_gpu, d_gpu, gamma_gpu, blmo.lmo, g_gpu, x_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
#         γ_max = reinterpret(Float32, CUDA.@allowscalar gamma_gpu[1])
#         gamma = FrankWolfe.perform_line_search(line_search, k, f, grad!, g_gpu, x_gpu, d_gpu, γ_max, ls_workspace)
#         γ = min(gamma, γ_max)
#         x_gpu .= x_gpu .- γ .* d_gpu

#         if mod(k, check_interval) == 0
#             push!(pre_computed_set, Float64.(Array(v_gpu)))
#             diff_gpu .= x_gpu .- v_gpu
#             gap = Float32(dot(diff_gpu, g_gpu))
#             if gap < Float32(epsilon)
#                 status = :OPTIMAL
#                 break
#             end
#         end
#     end

#     grad!(g_gpu, x_gpu)
#     launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
#     dual_gap = Float64(dot(x_gpu .- v_gpu, g_gpu))
#     x_out    = Float64.(Array(x_gpu))
#     map!(v -> v == 0.0 ? 0.0 : v, x_out, x_out)
#     primal   = f(x_out)

#     return x_out, primal, dual_gap, status, pre_computed_set
# end
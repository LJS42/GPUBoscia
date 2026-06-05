struct StandardFrankWolfeGPU <: Boscia.FrankWolfeVariant
    threads :: Int
end

function Boscia.solve_frank_wolfe(
    frank_wolfe_variant::StandardFrankWolfeGPU,
    f,
    grad!,
    managed_lmo,
    active_set;
    line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
    epsilon=1e-7,
    max_iteration=10000,
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
    x0 = gpu_start_point_initialize(
        managed_lmo,
        active_set,
        pre_computed_set,
        grad!,
    )
    dim = length(x0)
    pre_computed_set = Vector{Vector{Float64}}()

    @assert frank_wolfe_variant.threads % 32 == 0 "threads must be a multiple of 32"
    @assert frank_wolfe_variant.threads <= 1024   "threads cannot exceed 1024"

    threads = frank_wolfe_variant.threads
    blocks  = cld(dim, threads)
    n_warps = div(threads + 31, 32)
    shmem   = n_warps * sizeof(Float32)

    x_gpu     = CuArray(Float32.(x0))
    g_gpu     = CUDA.zeros(Float32, dim)
    gap_gpu   = CUDA.zeros(Float32, 1)
    x_old_gpu = CUDA.zeros(Float32, dim)
    blmo      = managed_lmo.lmo

    status = :MAXITER
    k = 0

    eff_lb_gpu, eff_ub_gpu = build_effective_bounds(blmo, dim)
    v_gpu = CUDA.zeros(Float32, dim)

    # ── Main loop ─────────────────────────────────────────
    diff_gpu       = CUDA.zeros(Float32, dim)
    check_interval = 100

    while k < max_iteration
        k += 1
        grad!(g_gpu, x_gpu)
        launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
        γ = 2f0 / Float32(k + 2)
        x_gpu .= (1f0 - γ) .* x_gpu .+ γ .* v_gpu

        if mod(k, check_interval) == 0
            push!(pre_computed_set, Float64.(Array(v_gpu)))

            diff_gpu .= x_gpu .- v_gpu
            gap = Float32(dot(diff_gpu, g_gpu))

            if gap < Float32(epsilon)
                status = :OPTIMAL
                break
            end
        end
    end

    # ── Final verification ────────────────────────────────
    grad!(g_gpu, x_gpu)
    launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
    dual_gap = Float64(dot(x_gpu .- v_gpu, g_gpu))
    x_out    = Float64.(Array(x_gpu))
    map!(v -> v == 0.0 ? 0.0 : v, x_out, x_out)
    primal   = f(x_out)

    # ── Return ────────────────────────────────────────────
    return x_out, primal, dual_gap, status, pre_computed_set
end

# using TimerOutputs
# const to = TimerOutput()

# struct StandardFrankWolfeGPU <: Boscia.FrankWolfeVariant
#     threads :: Int
# end

# function Boscia.solve_frank_wolfe(
#     frank_wolfe_variant::StandardFrankWolfeGPU,
#     f,
#     grad!,
#     managed_lmo,
#     active_set;
#     line_search::FrankWolfe.LineSearchMethod=FrankWolfe.Secant(),
#     epsilon=1e-7,
#     max_iteration=10000,
#     add_dropped_vertices=false,
#     use_extra_vertex_storage=false,
#     extra_vertex_storage=nothing,
#     callback=nothing,
#     lazy=false,
#     lazy_tolerance=2.0,
#     timeout=Inf,
#     verbose=false,
#     workspace=nothing,
#     kwargs...,
# )
#     # ── Setup ─────────────────────────────────────────────
#     x0  = Float64.(FrankWolfe.get_active_set_iterate(active_set))
#     dim = length(x0)

#     @assert frank_wolfe_variant.threads % 32 == 0 "threads must be a multiple of 32"
#     @assert frank_wolfe_variant.threads <= 1024   "threads cannot exceed 1024"

#     threads = frank_wolfe_variant.threads
#     blocks  = cld(dim, threads)

#     x_gpu    = CuArray(Float32.(x0))
#     g_gpu    = CUDA.zeros(Float32, dim)
#     v_gpu    = CUDA.zeros(Float32, dim)
#     diff_gpu = CUDA.zeros(Float32, dim)

#     blmo = managed_lmo.lmo
#     eff_lb_gpu, eff_ub_gpu = build_effective_bounds(blmo, dim)

#     # ── Async setup ───────────────────────────────────────
#     prev_gap   = Inf32
#     gap_ready  = false
#     gap_task   = nothing
#     check_interval = 10000
#     status = :MAXITER
#     k = 0

#     # ── Main loop ─────────────────────────────────────────
#     while k < max_iteration
#         k += 1
#         @timeit to "grad"     grad!(g_gpu, x_gpu)
#         @timeit to "kernel"   launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
#         @timeit to "update_x" begin
#             γ = 2f0 / Float32(k + 2)
#             x_gpu .= (1f0 - γ) .* x_gpu .+ γ .* v_gpu
#         end

#         if mod(k, check_interval) == 0
#             @timeit to "dual_gap_read" begin
#                 if gap_ready
#                     wait(gap_task) 
#                     if prev_gap < Float32(epsilon)
#                         status = :OPTIMAL
#                         break
#                     end
#                 end
#             end

#             @timeit to "dual_gap_submit" begin
#                 gap_task = @async begin
#                     CUDA.@sync begin
#                         diff_gpu .= x_gpu .- v_gpu
#                         prev_gap  = Float32(dot(diff_gpu, g_gpu))
#                     end
#                 end
#                 gap_ready = true
#             end
#         end
#     end

#     if gap_ready && gap_task !== nothing
#         wait(gap_task)
#     end

#     # ── Final verification ────────────────────────────────
#     @timeit to "final_grad"   grad!(g_gpu, x_gpu)
#     @timeit to "final_kernel" launch_kernel!(v_gpu, blmo.lmo, g_gpu, eff_lb_gpu, eff_ub_gpu, threads, blocks)
#     diff_gpu .= x_gpu .- v_gpu
#     dual_gap = Float64(dot(diff_gpu, g_gpu))
#     x_out    = Float64.(Array(x_gpu))
#     primal   = f(x_out)

#     # ── Return ────────────────────────────────────────────
#     return x_out, primal, dual_gap, status, x_out
# end
function gpu_start_point_initialize(
    lmo::Boscia.TimeTrackingLMO,
    active_set::FrankWolfe.ActiveSet,
    pre_computed_set,
    lb,
    ub,
    threads,
    blocks
)
    x0 = FrankWolfe.get_active_set_iterate(active_set)
    x0_gpu = CuArray(Float32.(x0))
    dim = length(x0)
    g_gpu = CUDA.zeros(Float32, dim)
    v_gpu = CUDA.zeros(Float32, dim)
    grad!(g_gpu, x0_gpu)
    launch_kernel!(v_gpu, lmo.lmo.lmo, g_gpu, lb, ub, threads, blocks)
    # dual_gap = dot(g, x0 - v)
    # primal = f(x0)
    # dual = primal - dual_gap
    # @info "primal: $primal, dual gap: $dual_gap, lower bound: $dual"
    return v_gpu
end
# struct CubeLMO <: FrankWolfe.LinearMinimizationOracle
#     lower_bounds::Vector{Float64}
#     upper_bounds::Vector{Float64}
#     int_vars::Vector{Int}
# end

function launch_kernel!(v_gpu, lmo::Boscia.CubeLMO, g_gpu, lb, ub, threads, blocks)
    @cuda threads=threads blocks=blocks bounded_compute_extreme_point_cube!(v_gpu, g_gpu, lb, ub)
end

struct CubeLMO_GPU
    lower_bounds :: CuDeviceVector{Float32, 1}
    upper_bounds :: CuDeviceVector{Float32, 1}
    int_vars     :: CuDeviceVector{Int32, 1}
end

"""
     bounded_compute_extreme_point(sblmo::CubeSimpleLMO, d, lb, ub, int_vars; kwargs...)

If the entry is positve, choose the lower bound. Else, choose the upper bound.
"""
function bounded_compute_extreme_point_cube!(
    v        :: CuDeviceVector{Float32},
    g        :: CuDeviceVector{Float32},
    lb       :: CuDeviceVector{Float32},
    ub       :: CuDeviceVector{Float32},
)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if tid <= length(v)
        v[tid] = g[tid] > 0f0 ? lb[tid] : ub[tid]
    end
    return
end

function build_effective_bounds(managed_lmo::Boscia.ManagedLMO, dim)
    eff_lb = Float32.(managed_lmo.lmo.lower_bounds)
    eff_ub = Float32.(managed_lmo.lmo.upper_bounds)
    
    for (i, var) in enumerate(managed_lmo.int_vars)
        eff_lb[var] = max(eff_lb[var], Float32(managed_lmo.lower_bounds[i]))
        eff_ub[var] = min(eff_ub[var], Float32(managed_lmo.upper_bounds[i]))
    end
    
    return CuArray(eff_lb), CuArray(eff_ub)
end
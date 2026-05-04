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

function launch_kernel_DICG!(v_gpu, a_gpu, d_gpu, gamma_max, lmo::Boscia.CubeLMO, g_gpu, x_gpu, lb_gpu, ub_gpu, threads, blocks)
    @cuda threads=threads blocks=blocks bounded_compute_extreme_point_cube!(v_gpu, g_gpu, lb_gpu, ub_gpu)
    @cuda threads=threads blocks=blocks bounded_compute_inface_extreme_point_cube!(a_gpu, g_gpu, x_gpu, lb_gpu, ub_gpu)
    d_gpu .= a_gpu .- v_gpu
    @cuda threads=threads blocks=blocks dicg_maximum_step_kernel_cube!(gamma_max, d_gpu, x_gpu, lb_gpu, ub_gpu)
end

function bounded_compute_inface_extreme_point_cube!(
    a   :: CuDeviceVector{Float32},
    g   :: CuDeviceVector{Float32},
    x   :: CuDeviceVector{Float32},
    lb  :: CuDeviceVector{Float32},
    ub  :: CuDeviceVector{Float32},
)
    atol=1f-6
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if tid <= length(a)
        if x[tid] >= ub[tid] - atol && x[tid] <= ub[tid] + atol
            a[tid] = ub[tid]
        elseif x[tid] >= lb[tid] - atol && x[tid] <= lb[tid] + atol
            a[tid] = lb[tid]
        else
            a[tid] = g[tid] > 0f0 ? ub[tid] : lb[tid]  
        end
    end
    return
end

function warp_reduce_min(val::Float32)
    val = min(val, shfl_down_sync(0xffffffff, val, 16))
    val = min(val, shfl_down_sync(0xffffffff, val, 8))
    val = min(val, shfl_down_sync(0xffffffff, val, 4))
    val = min(val, shfl_down_sync(0xffffffff, val, 2))
    val = min(val, shfl_down_sync(0xffffffff, val, 1))
    return val
end

function dicg_maximum_step_kernel_cube!(gamma_max, d, x, lb, ub)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    local_min = Inf32
    if tid <= length(d)
        di = d[tid]
        if di < 0f0
            local_min = (ub[tid] - x[tid]) / -di
        elseif di > 0f0
            local_min = (x[tid] - lb[tid]) / di
        end
    end
    local_min = warp_reduce_min(local_min)
    if threadIdx().x % UInt32(32) == UInt32(1)
        i = reinterpret(UInt32, local_min)
        CUDA.atomic_min!(reinterpret(Core.LLVMPtr{UInt32, 1}, pointer(gamma_max, 1)), i)
    end
    return
end
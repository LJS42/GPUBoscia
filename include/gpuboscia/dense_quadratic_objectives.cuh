#pragma once
// Device-resident dense quadratic objectives.
//
// Objective contract consumed by DeviceResidentHybridSolver:
//   static constexpr int n_vars;
//   static constexpr int block_threads; // warp-aligned launch upper bound
//   static constexpr int shared_bytes;
//   __device__ double value_local(int tid, const double* x_node,
//                                 unsigned char* smem) const;
//   __device__ void prepare_block(int tid, unsigned char* smem) const;
//   __device__ void grad(int tid, const double* x_node,
//                        double* grad,
//                        unsigned char* smem) const;

#include <cuda_runtime.h>
#include <cstdint>

template<int Vars, int Threads = 64>
struct GlobalCachedDenseQuadraticObjective {
    static constexpr bool supports_quadratic_exact = true;
    static constexpr int n_vars = Vars;
    static constexpr int block_threads = Threads;
    static constexpr int shared_bytes = Vars * (int)sizeof(double) + 16;

    // Device matrix is stored transposed: d_Qt[k * Vars + j] == Q[j, k].
    // Threads computing adjacent output rows j issue contiguous loads at k.
    const double* d_Qt;
    const double* d_linear;
    double        grad_scale;
    double        linear_scale;
    double        constant;

    __host__ __device__
    GlobalCachedDenseQuadraticObjective(const double* Q_transposed = nullptr,
                                        const double* linear = nullptr,
                                        double grad_scale_ = 1.0,
                                        double linear_scale_ = 1.0,
                                        double constant_ = 0.0)
        : d_Qt(Q_transposed), d_linear(linear)
        , grad_scale(grad_scale_), linear_scale(linear_scale_)
        , constant(constant_) {}

    __device__ double value_local(int tid, const double* x_node,
                                  unsigned char* /*smem*/) const {
        double quad = 0.0;
        double lin = 0.0;

        for (int j = tid; j < Vars; j += blockDim.x) {
            double qx_j = 0.0;
            for (int k = 0; k < Vars; k++)
                qx_j += d_Qt[(size_t)k * Vars + j] * x_node[k];

            quad += x_node[j] * qx_j;
            if (d_linear)
                lin += d_linear[j] * x_node[j];
        }

        double val = 0.5 * grad_scale * quad
                   + grad_scale * linear_scale * lin;
        if (tid == 0)
            val += constant;
        return val;
    }

    __device__ double quadratic_curvature_local(
        int tid, const double* dir, unsigned char* smem) const {
        double* ds = shared_x(smem);
        for (int k = tid; k < Vars; k += blockDim.x) ds[k] = dir[k];
        __syncthreads();
        double curvature = 0.0;
        for (int j = tid; j < Vars; j += blockDim.x) {
            double product = 0.0;
            for (int k = 0; k < Vars; ++k)
                product += d_Qt[(size_t)k * Vars + j] * ds[k];
            curvature += grad_scale * ds[j] * product;
        }
        __syncthreads();
        return curvature;
    }

    const void* l2_window_base() const { return d_Qt; }
    size_t l2_window_bytes() const { return (size_t)Vars * Vars * sizeof(double); }

    __device__ double* shared_x(unsigned char* smem) const {
        const uintptr_t addr =
            (reinterpret_cast<uintptr_t>(smem) + 15u) & ~uintptr_t(15u);
        return reinterpret_cast<double*>(addr);
    }

    __device__ void prepare_block(int /*tid*/, unsigned char* /*smem*/) const {}

    __device__ void grad(int tid, const double* x_node,
                         double* grad,
                         unsigned char* smem) const {
        double* x_s = shared_x(smem);

        for (int k = tid; k < Vars; k += blockDim.x)
            x_s[k] = x_node[k];
        __syncthreads();

        for (int j = tid; j < Vars; j += blockDim.x) {
            double acc = 0.0;
            for (int k = 0; k < Vars; k++)
                acc += d_Qt[(size_t)k * Vars + j] * x_s[k];

            const double linear =
                d_linear ? linear_scale * d_linear[j] : 0.0;
            grad[j] = grad_scale * (acc + linear);
        }
        __syncthreads();
    }
};

template<int Vars, int Threads = 256>
struct ILSGlobalCachedObjective
    : public GlobalCachedDenseQuadraticObjective<Vars, Threads> {
    using Base = GlobalCachedDenseQuadraticObjective<Vars, Threads>;

    __host__ __device__
    ILSGlobalCachedObjective(const double* d_AtA_transposed = nullptr,
                             const double* d_Atb = nullptr,
                             double btb = 0.0)
        : Base(d_AtA_transposed, d_Atb, 2.0, -1.0, btb) {}
};

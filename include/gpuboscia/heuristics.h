#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
namespace gpuboscia {

template<typename LMO, typename = void>
struct DRSHasCandidateHeuristic : std::false_type {};
template<typename LMO>
struct DRSHasCandidateHeuristic<LMO, std::void_t<decltype(
    std::declval<const LMO&>().integer_candidate_device(0, 0,
        (const double*)nullptr, (const double*)nullptr,
        (const double*)nullptr, (const std::uint8_t*)nullptr,
        (double*)nullptr))>> : std::true_type {};

// Cooperative GPU heuristic: every thread participates and initializes its
// strided coordinates. The caller synchronizes before validating/scoring.
// Store scalars or device pointers; the heuristic is copied into each kernel.
struct RoundingHeuristic {
    template<typename LMO>
    __device__ void generate(const LMO& lmo, int tid, int n,
        const double* x, const double* lb, const double* ub,
        const std::uint8_t* flags, double* candidate) const {
        if constexpr (DRSHasCandidateHeuristic<LMO>::value) {
            lmo.integer_candidate_device(tid, n, x, lb, ub, flags, candidate);
        } else {
            for (int j = tid; j < n; j += blockDim.x)
                candidate[j] = flags[j] ? round(x[j]) : x[j];
        }
    }
};

} // namespace gpuboscia
#endif

#pragma once

#include <cmath>

enum class LineSearchKind {
    OpenLoop,
    ScaledOpenLoop,
    SafeguardedScaledOpenLoop,
    Secant,
    QuadraticExact
};

enum class FWAlgorithmKind {
    FrankWolfe,
    DICG
};

struct LineSearchOptions {
    LineSearchKind kind = LineSearchKind::Secant;
    int    secant_iters = 4;
    double secant_tol = 1e-10;
    double secant_denom_eps = 1e-14;
};

inline const char* fw_algorithm_name(FWAlgorithmKind kind)
{
    return kind == FWAlgorithmKind::DICG
        ? "Decomposition-Invariant Frank-Wolfe"
        : "Frank-Wolfe";
}

inline const char* line_search_name(LineSearchKind kind)
{
    switch (kind) {
    case LineSearchKind::ScaledOpenLoop:
        return "scaled open-loop gamma_max * 2/(t+2)";
    case LineSearchKind::SafeguardedScaledOpenLoop:
        return "scaled open-loop with full limiting steps";
    case LineSearchKind::QuadraticExact:
        return "quadratic exact";
    case LineSearchKind::Secant:
        return "secant on phi'(gamma)";
    default:
        return "open-loop min(gamma_max, 2/(t+2))";
    }
}

// Minimize -gap*t + 0.5*curvature*t*t on [0, maximum].
// curvature is d^T Hessian(f) d, including the objective's scaling.
#ifdef __CUDACC__
__host__ __device__
#endif
inline double quadratic_exact_step(double gap, double curvature, double maximum) {
    if (maximum <= 0.0) return 0.0;
    if (curvature > 0.0) {
        if (gap <= 0.0) return 0.0;
        const double step = gap / curvature;
        return step < maximum ? step : maximum;
    }
    // Linear or concave restriction: compare endpoints. This also avoids
    // dividing by zero for PSD null directions; ties keep the current point.
    return -gap + 0.5 * curvature * maximum < 0.0 ? maximum : 0.0;
}

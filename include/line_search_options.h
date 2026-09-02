#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>

enum class LineSearchKind {
    OpenLoop,
    ScaledOpenLoop,
    SafeguardedScaledOpenLoop,
    Secant
};

enum class FWAlgorithmKind {
    FrankWolfe,
    DICG
};

struct LineSearchOptions {
    LineSearchKind kind = LineSearchKind::Secant;
    int    secant_iters = 8;
    double secant_tol = 1e-3;
    double secant_denom_eps = 1e-4;
};

struct FastBBExampleOptions {
    FWAlgorithmKind algorithm = FWAlgorithmKind::DICG;
    LineSearchOptions line_search;
    char oed_criterion = 'd';
    std::uint64_t random_seed = 1;
    bool random_seed_explicit = false;
    double gpu_eps = 1e-3;
    double gpu_eps_decay = 0.8;
    double gpu_eps_min = 1e-6;
    double abs_gap_tol = 1e-6;
    double rel_gap_tol = 1e-2;
    double time_limit_s = 3600.0;
    double natural_lower_bound = -std::numeric_limits<double>::infinity();
    int log_interval_nodes = 100;
    std::string instance_path;
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
    case LineSearchKind::Secant:
        return "secant on phi'(gamma)";
    default:
        return "open-loop min(gamma_max, 2/(t+2))";
    }
}

inline void print_fastbb_example_usage(const char* prog)
{
    std::printf("Usage: %s [--algorithm fw|dicg]\n"
                "          [--criterion a|d] [--seed N]\n"
                "          [--line-search open-loop|scaled-open-loop|"
                "safeguarded-scaled-open-loop|secant]\n"
                "          [--secant-iters N] [--secant-tol VALUE]\n"
                "          [--secant-denom-eps VALUE]\n"
                "          [--gpu-eps VALUE] [--gpu-eps-decay VALUE]\n"
                "          [--gpu-eps-min VALUE] [--abs-gap-tol VALUE]\n"
                "          [--rel-gap-tol VALUE] [--time-limit SECONDS]\n"
                "          [--natural-lower-bound VALUE]\n"
                "          [--log-interval-nodes N] [--instance PATH]\n", prog);
}

inline const char* require_option_value(int argc, char** argv, int& i)
{
    if (i + 1 < argc)
        return argv[++i];

    std::fprintf(stderr, "Missing value for argument: %s\n", argv[i]);
    return nullptr;
}

// Returns -1 to continue, otherwise the process exit code.
inline int parse_fastbb_example_options(
    int argc, char** argv, FastBBExampleOptions& opts)
{
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--help") == 0) {
            print_fastbb_example_usage(argv[0]);
            return 0;
        }

        const char* value = nullptr;
        if (std::strcmp(arg, "--algorithm") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            if (std::strcmp(value, "fw") == 0) {
                opts.algorithm = FWAlgorithmKind::FrankWolfe;
            } else if (std::strcmp(value, "dicg") == 0) {
                opts.algorithm = FWAlgorithmKind::DICG;
            } else {
                std::fprintf(stderr,
                    "--algorithm must be either 'fw' or 'dicg'.\n");
                return 1;
            }
        } else if (std::strcmp(arg, "--criterion") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            if (std::strcmp(value, "a") == 0
                    || std::strcmp(value, "A") == 0) {
                opts.oed_criterion = 'a';
            } else if (std::strcmp(value, "d") == 0
                    || std::strcmp(value, "D") == 0) {
                opts.oed_criterion = 'd';
            } else {
                std::fprintf(stderr, "--criterion must be either 'a' or 'd'.\n");
                return 1;
            }
        } else if (std::strcmp(arg, "--seed") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (!end || *end != '\0') {
                std::fprintf(stderr, "--seed must be an unsigned integer.\n");
                return 1;
            }
            opts.random_seed = static_cast<std::uint64_t>(parsed);
            opts.random_seed_explicit = true;
        } else if (std::strcmp(arg, "--log-interval-nodes") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (!end || *end != '\0' || parsed <= 0
                    || parsed > std::numeric_limits<int>::max()) {
                std::fprintf(stderr,
                    "--log-interval-nodes must be a positive integer.\n");
                return 1;
            }
            opts.log_interval_nodes = static_cast<int>(parsed);
        } else if (std::strcmp(arg, "--time-limit") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            char* end = nullptr;
            const double parsed = std::strtod(value, &end);
            if (!end || *end != '\0' || !std::isfinite(parsed)
                    || parsed <= 0.0) {
                std::fprintf(stderr,
                    "--time-limit must be a positive finite number of seconds.\n");
                return 1;
            }
            opts.time_limit_s = parsed;
        } else if (std::strcmp(arg, "--instance") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.instance_path = value;
        } else if (std::strcmp(arg, "--natural-lower-bound") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            char* end = nullptr;
            const double parsed = std::strtod(value, &end);
            if (!end || *end != '\0' || !std::isfinite(parsed)) {
                std::fprintf(stderr,
                    "--natural-lower-bound must be a finite number.\n");
                return 1;
            }
            opts.natural_lower_bound = parsed;
        } else if (std::strcmp(arg, "--line-search") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            if (std::strcmp(value, "open-loop") == 0) {
                opts.line_search.kind = LineSearchKind::OpenLoop;
            } else if (std::strcmp(value, "scaled-open-loop") == 0) {
                opts.line_search.kind = LineSearchKind::ScaledOpenLoop;
            } else if (std::strcmp(
                           value, "safeguarded-scaled-open-loop") == 0) {
                opts.line_search.kind =
                    LineSearchKind::SafeguardedScaledOpenLoop;
            } else if (std::strcmp(value, "secant") == 0) {
                opts.line_search.kind = LineSearchKind::Secant;
            } else {
                std::fprintf(stderr,
                    "--line-search must be 'open-loop', "
                    "'scaled-open-loop', 'safeguarded-scaled-open-loop', "
                    "or 'secant'.\n");
                return 1;
            }
        } else if (std::strcmp(arg, "--secant-iters") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.line_search.secant_iters = std::atoi(value);
        } else if (std::strcmp(arg, "--secant-tol") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.line_search.secant_tol = std::strtod(value, nullptr);
        } else if (std::strcmp(arg, "--secant-denom-eps") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.line_search.secant_denom_eps = std::strtod(value, nullptr);
        } else if (std::strcmp(arg, "--gpu-eps") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.gpu_eps = std::strtod(value, nullptr);
        } else if (std::strcmp(arg, "--gpu-eps-decay") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.gpu_eps_decay = std::strtod(value, nullptr);
        } else if (std::strcmp(arg, "--gpu-eps-min") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.gpu_eps_min = std::strtod(value, nullptr);
        } else if (std::strcmp(arg, "--abs-gap-tol") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.abs_gap_tol = std::strtod(value, nullptr);
        } else if (std::strcmp(arg, "--rel-gap-tol") == 0) {
            if (!(value = require_option_value(argc, argv, i))) return 1;
            opts.rel_gap_tol = std::strtod(value, nullptr);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg);
            print_fastbb_example_usage(argv[0]);
            return 1;
        }
    }

    const auto& ls = opts.line_search;
    if (opts.algorithm == FWAlgorithmKind::FrankWolfe
            && (ls.kind == LineSearchKind::ScaledOpenLoop
                || ls.kind == LineSearchKind::SafeguardedScaledOpenLoop)) {
        std::fprintf(stderr,
            "scaled-open-loop is currently available only for DICG.\n");
        return 1;
    }
    if (ls.secant_iters <= 0
            || !std::isfinite(ls.secant_tol) || ls.secant_tol <= 0.0
            || !std::isfinite(ls.secant_denom_eps)
            || ls.secant_denom_eps <= 0.0) {
        std::fprintf(stderr,
            "Secant parameters must be positive finite values.\n");
        return 1;
    }
    if (!std::isfinite(opts.gpu_eps) || opts.gpu_eps <= 0.0
            || !std::isfinite(opts.gpu_eps_min) || opts.gpu_eps_min <= 0.0
            || !std::isfinite(opts.gpu_eps_decay)
            || opts.gpu_eps_decay <= 0.0 || opts.gpu_eps_decay > 1.0
            || !std::isfinite(opts.abs_gap_tol) || opts.abs_gap_tol < 0.0
            || !std::isfinite(opts.rel_gap_tol) || opts.rel_gap_tol < 0.0) {
        std::fprintf(stderr, "Tolerance parameters are invalid.\n");
        return 1;
    }

    return -1;
}

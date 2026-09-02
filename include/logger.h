#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <limits>
#include <string>
#include <vector>
#include "gap_utils.h"

// ================================================================
//  Logger — toggleable solver progress output for FastBB
//
//  Usage:
//    Logger log(true);              // false = silent
//    log.print_header(params);      // greeting + settings + column header
//    log.print_row(iter, open, bound, incumbent, is_new_incumbent);
//    log.print_footer(stats);       // solution + search summary
// ================================================================

struct LoggerParams {
    std::string traversal_strategy = "best_first";
    std::string branching_strategy = "branch_pow2";
    std::string algorithm          = "Frank-Wolfe";
    std::string line_search_method = "open-loop";
    int         n_vars             = 0;
    int         n_int_vars         = 0;
    double      abs_gap_tol        = 1e-6;
    double      rel_gap_tol        = 1e-2;
    double      time_limit_s       = std::numeric_limits<double>::infinity();
    double      natural_lower_bound = -std::numeric_limits<double>::infinity();
    std::uint64_t random_seed       = 0;
    bool        print_random_seed   = false;
};

struct SolutionStats {
    std::string solution_status  = "Unknown";
    double      primal_objective = std::numeric_limits<double>::infinity();
    double      dual_bound       = -std::numeric_limits<double>::infinity();
    int         total_nodes      = 0;
    double      total_time_s     = 0.0;
    std::vector<double> solution;
    int         solution_preview_size = 12;
};

class Logger {
public:
    bool enabled;

    explicit Logger(bool on = true, int node_interval = 1000)
        : enabled(on), node_interval_(node_interval > 0 ? node_interval : 1) {}

    void print_header(const LoggerParams& p) {
        if (!enabled) return;
        start_ = Clock::now();
        last_printed_iter_ = 0;
        printf("Hello:)\n");
        printf("Parameter settings.\n");
        printf("  Tree traversal strategy:     %s\n",  p.traversal_strategy.c_str());
        printf("  Branching strategy:          %s\n",  p.branching_strategy.c_str());
        printf("  Algorithm:                   %s\n",  p.algorithm.c_str());
        printf("  Line search method:          %s\n",  p.line_search_method.c_str());
        if (p.print_random_seed)
            printf("  Random seed:                 %llu\n",
                   static_cast<unsigned long long>(p.random_seed));
        printf("  Termination:                 abs gap <= %.3e or rel gap <= %.3e\n",
               p.abs_gap_tol, p.rel_gap_tol);
        if (std::isfinite(p.time_limit_s))
            printf("  Time limit (s):              %.3f\n", p.time_limit_s);
        if (std::isfinite(p.natural_lower_bound))
            printf("  Natural lower bound:         %.6e\n",
                   p.natural_lower_bound);
        printf("  Total number of variables:   %d\n",  p.n_vars);
        printf("  Number of integer variables: %d\n",  p.n_int_vars);
        print_sep();
        printf("  Iter   Open           Bound       Incumbent       Gap (abs)       Gap (rel)        Time (s)\n");
        print_sep();
    }

    void print_stage(const char* stage) {
        if (!enabled) return;
        printf("--- [%.3f s] %s ---\n", elapsed_s(), stage);
    }

    // Call once per processed node.
    // is_new_incumbent: set true when this node improved the primal bound.
    void print_row(int    iter,
                   int    open_nodes,
                   double dual_bound,
                   double incumbent,
                   bool   is_new_incumbent = false)
    {
        if (!enabled) return;
        if (iter != 1 && !is_new_incumbent
                && iter - last_printed_iter_ < node_interval_)
            return;
        last_printed_iter_ = iter;
        double t       = elapsed_s();
        double gap_abs = fastbb::gap_absolute(dual_bound, incumbent);
        double gap_rel = fastbb::gap_relative(dual_bound, incumbent);
        printf("%c%5d%7d  %s  %s  %s  %s  %s\n",
               is_new_incumbent ? '*' : ' ',
               iter,
               open_nodes,
               fmt_sci(dual_bound),
               fmt_sci(incumbent),
               fmt_sci(gap_abs),
               fmt_sci(gap_rel),
               fmt_sci(t));
    }

    void print_footer(const SolutionStats& s) {
        if (!enabled) return;
        print_sep();
        printf("Solution Statistics.\n");
        printf("  Solution Status:              %s\n",  s.solution_status.c_str());
        printf("  Primal Objective:             %s\n",  fmt_sci(s.primal_objective));
        printf("  Dual Bound:                   %s\n",  fmt_sci(s.dual_bound));
        double rel_gap = fastbb::gap_relative(s.dual_bound, s.primal_objective);
        printf("  Dual Gap (relative):          %s\n",  fmt_sci(rel_gap));
        if (!s.solution.empty()) {
            const int preview = std::min(
                (int)s.solution.size(), std::max(0, s.solution_preview_size));
            int nonzeros = 0;
            for (double value : s.solution)
                nonzeros += std::fabs(value) > 1e-12;
            printf("  Solution dimension:           %zu\n", s.solution.size());
            printf("  Solution nonzeros:             %d\n", nonzeros);
            printf("  Solution preview x[0:%d]:      [", preview);
            for (int i = 0; i < preview; ++i)
                printf("%s%.6g", i ? ", " : "", s.solution[i]);
            printf("]%s\n", preview < (int)s.solution.size() ? " ..." : "");
        } else {
            printf("  Solution:                      not found\n");
        }
        printf("Search Statistics.\n");
        printf("  Total number of nodes processed: %d\n", s.total_nodes);
        printf("  Total time (s):               %s\n",  fmt_sci(s.total_time_s));
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_ = Clock::now();
    int node_interval_ = 100;
    int last_printed_iter_ = 0;

    static constexpr double INF = std::numeric_limits<double>::infinity();

    double elapsed_s() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    // Returns a 14-char right-aligned scientific-notation string,
    // substituting "Inf" / "-Inf" / "NaN" for special values.
    // Uses a rotating pool of 8 buffers so multiple calls are safe
    // within a single printf argument list.
    static const char* fmt_sci(double v) {
        static thread_local char bufs[8][32];
        static thread_local int  slot = 0;
        char* buf = bufs[slot++ % 8];
        if (std::isnan(v))
            snprintf(buf, 32, "%14s", "NaN");
        else if (v >= INF)
            snprintf(buf, 32, "%14s", "Inf");
        else if (v <= -INF)
            snprintf(buf, 32, "%14s", "-Inf");
        else
            snprintf(buf, 32, "%14.6e", v);
        return buf;
    }

    static void print_sep() {
        printf("---------------------------------------------"
               "------------------------------------------------\n");
    }
};

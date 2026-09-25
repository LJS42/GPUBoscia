#pragma once
#include <atomic>
#include <vector>
#include <mutex>
#include <limits>

// Thread-safe incumbent using CAS on the value and a separate small mutex
// only for copying the solution vector.  Hot path (pruning check) is a
// single atomic load — no lock at all.
struct AtomicIncumbent {
    std::atomic<double> value{std::numeric_limits<double>::infinity()};
    std::vector<double> x;
    mutable std::mutex  x_mtx;

    // Returns true if val improves the incumbent.
    bool try_update(double val, const std::vector<double>& new_x) {
        double old = value.load(std::memory_order_relaxed);
        while (val < old) {
            if (value.compare_exchange_weak(old, val,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lk(x_mtx);
                x = new_x;
                return true;
            }
        }
        return false;
    }

    double get() const { return value.load(std::memory_order_acquire); }

    std::vector<double> get_x() const {
        std::lock_guard<std::mutex> lk(x_mtx);
        return x;
    }
};

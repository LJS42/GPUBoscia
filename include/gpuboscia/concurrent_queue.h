#pragma once
#include <queue>
#include <vector>
#include <mutex>
#include <set>
#include "node.h"

// Thread-safe min-heap by dual_bound.
// Its own mutex is independent of everything else — no cross-lock dependency.
struct ConcurrentQueue {
    void push(Node nd) {
        std::lock_guard<std::mutex> lk(mtx_);
        pq_.push(std::move(nd));
    }

    // Pops up to n nodes.  Nodes already pruned by cutoff are discarded here
    // so callers never see stale work.
    std::vector<Node> pop_batch(int n, double cutoff) {
        std::lock_guard<std::mutex> lk(mtx_);
        std::vector<Node> batch;
        batch.reserve(n);
        while (!pq_.empty() && (int)batch.size() < n) {
            if (pq_.top().dual_bound >= cutoff) { pq_.pop(); continue; }
            batch.push_back(pq_.top());
            pq_.pop();
        }
        return batch;
    }

    // Pops exactly n live nodes, or none when there is not enough work to
    // fill a GPU batch. Because the queue is ordered by dual bound, once the
    // top node is cut off every remaining node is cut off as well.
    std::vector<Node> pop_full_batch(int n, double cutoff) {
        std::lock_guard<std::mutex> lk(mtx_);
        while (!pq_.empty() && pq_.top().dual_bound >= cutoff)
            pq_.pop();

        if ((int)pq_.size() < n)
            return {};

        std::vector<Node> batch;
        batch.reserve(n);
        for (int i = 0; i < n && !pq_.empty(); i++) {
            if (pq_.top().dual_bound >= cutoff) {
                // The remaining heap is all cut off, even if the first
                // nodes in this batch were live.
                while (!pq_.empty()) pq_.pop();
                break;
            }
            batch.push_back(pq_.top());
            pq_.pop();
        }
        if ((int)batch.size() < n) {
            for (auto& nd : batch) pq_.push(std::move(nd));
            return {};
        }
        return batch;
    }

    int  size()  const { std::lock_guard<std::mutex> lk(mtx_); return (int)pq_.size(); }

    // Early-dispatched children may outlive their parent's host validation.
    void discard_children(int parent_id) {
        std::lock_guard<std::mutex> lk(mtx_);
        decltype(pq_) kept;
        while (!pq_.empty()) {
            if (pq_.top().parent_id != parent_id) kept.push(pq_.top());
            pq_.pop();
        }
        pq_.swap(kept);
    }
    bool empty() const { std::lock_guard<std::mutex> lk(mtx_); return pq_.empty(); }

    double best_bound(double cutoff) const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (pq_.empty()) return cutoff;
        const double bound = pq_.top().dual_bound;
        return bound < cutoff ? bound : cutoff;
    }

private:
    struct MinHeap {
        bool operator()(const Node& a, const Node& b) const {
            return a.dual_bound > b.dual_bound;
        }
    };

    mutable std::mutex mtx_;
    std::priority_queue<Node, std::vector<Node>, MinHeap> pq_;
};

// Thread-safe lower-bound set for nodes that have been removed from the
// frontier queue and are currently being relaxed/expanded.
struct ConcurrentBoundSet {
    void insert(double bound) {
        std::lock_guard<std::mutex> lk(mtx_);
        bounds_.insert(bound);
    }

    void insert(const std::vector<Node>& nodes) {
        std::lock_guard<std::mutex> lk(mtx_);
        for (const Node& nd : nodes)
            bounds_.insert(nd.dual_bound);
    }

    void erase_one(double bound) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = bounds_.find(bound);
        if (it != bounds_.end())
            bounds_.erase(it);
    }

    double best_bound(double cutoff) const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (bounds_.empty()) return cutoff;
        const double bound = *bounds_.begin();
        return bound < cutoff ? bound : cutoff;
    }

private:
    mutable std::mutex mtx_;
    std::multiset<double> bounds_;
};

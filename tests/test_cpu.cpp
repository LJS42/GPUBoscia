// CPU-only unit tests — compiled with g++, no CUDA required.
// Tests all non-GPU components: is_integer_feasible, BnBTree,
// node selectors, branching strategies.
//
// Build:  cmake --build build --target test_cpu
// Run:    ./build/test_cpu

#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

#include <gpuboscia/line_search_options.h>
#include "../examples/common/options.h"
#include <gpuboscia/bnb.h>
#include <gpuboscia/lmo_dicg.h>
#include <gpuboscia/node.h>
#include <gpuboscia/node_select.h>
#include <gpuboscia/branching.h>
#include <gpuboscia/gap_utils.h>
#include <gpuboscia/problem.h>
#include <gpuboscia/tree.h>
#include <gpuboscia/concurrent_queue.h>

// ============================================================
//  Minimal test harness
// ============================================================
static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, msg) \
    do { \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++g_pass; } \
        else      { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_fail; } \
    } while (0)

#define EXPECT_EQ(a, b, msg)          EXPECT((a) == (b),               msg)
#define EXPECT_NE(a, b, msg)          EXPECT((a) != (b),               msg)
#define EXPECT_NEAR(a, b, tol, msg)   EXPECT(std::abs((a)-(b)) < (tol), msg)

void test_certified_fw_lower_bound() {
    std::cout << "\n[certified_fw_lower_bound]\n";
    const double inherited = -7.0;
    EXPECT_NEAR(gpuboscia::certified_fw_lower_bound(inherited, 10.0, 3.0),
                7.0, 1e-12, "valid certificate advances lower bound");
    EXPECT_NEAR(gpuboscia::certified_fw_lower_bound(inherited, 10.0, -1e-12),
                inherited, 1e-12, "negative gap preserves inherited bound");
    EXPECT_NEAR(gpuboscia::certified_fw_lower_bound(
                    inherited, 10.0, std::numeric_limits<double>::quiet_NaN()),
                inherited, 1e-12, "NaN gap preserves inherited bound");
    EXPECT_NEAR(gpuboscia::certified_fw_lower_bound(
                    inherited, std::numeric_limits<double>::infinity(), 1.0),
                inherited, 1e-12, "non-finite primal preserves inherited bound");
}

// Shared branch strategy for tests: 2^3 branching on top-3 infeasible vars.
static BranchStrategy make_branch_fn() {
    return [](const Node& p, int& nid,
              const std::vector<double>& x,
              const std::vector<int>& iv) {
        return branch_pow2(p, nid, x, iv, 3);
    };
}

// ============================================================
//  is_integer_feasible
// ============================================================
void test_is_integer_feasible() {
    std::cout << "\n[is_integer_feasible]\n";

    const std::vector<int> vars = {0, 1, 2};

    EXPECT( is_integer_feasible({0.0, 1.0, 2.0},  vars), "all integer");
    EXPECT(!is_integer_feasible({0.5, 1.0, 2.0},  vars), "x[0]=0.5 fractional");
    EXPECT(!is_integer_feasible({0.0, 0.999, 2.0}, vars), "x[1]=0.999 fractional");
    EXPECT(!is_integer_feasible({1.5, 1.5, 1.5},  vars), "all fractional");
    EXPECT( is_integer_feasible({-3.0, 0.0, 4.0}, vars), "negative integer");

    EXPECT( is_integer_feasible({1.0 + 5e-7, 2.0}, {0,1}, 1e-6), "within default tol");
    EXPECT(!is_integer_feasible({1.0 + 5e-5, 2.0}, {0,1}, 1e-6), "outside default tol");
    EXPECT( is_integer_feasible({1.5}, {99}), "OOB int_var index skipped → feasible");
}

// ============================================================
//  is_node_feasible
// ============================================================
void test_is_node_feasible() {
    std::cout << "\n[is_node_feasible]\n";

    EXPECT(is_node_feasible(Node(1, -1, 0, {0.0, 1.0}, {3.0, 2.0}), {0, 1}),
           "non-empty integer box is feasible");
    EXPECT(!is_node_feasible(Node(2, -1, 0, {2.0}, {1.0}), {0}),
           "lb > ub is infeasible");
    EXPECT(!is_node_feasible(Node(3, -1, 0, {1.2}, {1.8}), {0}),
           "integer var with no integer in interval is infeasible");
    EXPECT(is_node_feasible(Node(4, -1, 0, {1.2}, {1.8}), {}),
           "continuous var may use a fractional interval");
    EXPECT(!is_node_feasible(Node(5, -1, 0, {0.0, 1.0}, {1.0}), {0}),
           "mismatched bound vector sizes are infeasible");


}



// ============================================================
//  Dual-gap helper functions
// ============================================================
void test_gap_tolerance() {
    std::cout << "\n[gap helpers]\n";

    EXPECT_NEAR(gpuboscia::gap_absolute(9.0, 10.0), 1.0, 1e-12,
                "absolute gap matches logger value");
    EXPECT_NEAR(gpuboscia::gap_relative(9.0, 10.0), 1.0 / 9.0, 1e-12,
                "relative gap uses min(|bound|, |incumbent|)");
    EXPECT_NEAR(gpuboscia::gap_relative(0.25, 0.75), 2.0, 1e-12,
                "relative gap does not floor the denominator at one");
    EXPECT_NEAR(gpuboscia::gap_relative(-10.0, -9.0), 1.0 / 9.0, 1e-12,
                "relative gap denominator is symmetric in both bounds");
    EXPECT(std::isinf(gpuboscia::gap_relative(0.0, 0.0)),
           "both zero bounds have infinite relative gap");
    EXPECT(std::isinf(gpuboscia::gap_relative(0.0, 1.0)),
           "zero lower bound has infinite relative gap");
    EXPECT(std::isinf(gpuboscia::gap_relative(-1.0, 0.0)),
           "zero incumbent has infinite relative gap");
    EXPECT(gpuboscia::gap_within_tolerance(0.0, 0.0, 0.0, 0.01),
           "closed zero gap terminates via absolute tolerance");
    EXPECT(!gpuboscia::gap_within_tolerance(9.0, 10.0, 0.0, 0.105),
           "relative stopping uses the smaller magnitude denominator");
    EXPECT(!gpuboscia::gap_within_tolerance(0.0, 1.0, 0.01, 0.01),
           "zero lower bound does not terminate an open search");
    EXPECT(!gpuboscia::gap_within_tolerance(-1.0, 0.0, 0.01, 0.01),
           "zero incumbent does not terminate an open search");
    for (double invalid : {std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT(std::isinf(gpuboscia::gap_relative(invalid, 1.0))
                   && std::isinf(gpuboscia::gap_relative(1.0, invalid)),
               "non-finite bounds have infinite relative gap");
        EXPECT(!gpuboscia::gap_within_tolerance(invalid, 1.0, 0.01, 0.01)
                   && !gpuboscia::gap_within_tolerance(1.0, invalid, 0.01, 0.01),
               "non-finite bounds cannot terminate the search");
    }

    EXPECT(gpuboscia::gap_within_tolerance(9.99, 10.0, 0.02, 1e-12),
           "within absolute tolerance");
    EXPECT(gpuboscia::gap_within_tolerance(99.0, 100.0, 1e-12, 0.011),
           "within relative tolerance");
    EXPECT(!gpuboscia::gap_within_tolerance(-1.0, 1.0, 0.1, 0.1),
           "opposite signs do not satisfy relative tolerance");

    EXPECT_NEAR(gpuboscia::depth_dependent_tolerance(1e-3, 0.8, 1e-6, 0),
                1e-3, 1e-15, "root uses configured tolerance");
    EXPECT_NEAR(gpuboscia::depth_dependent_tolerance(1e-3, 0.8, 1e-6, 2),
                6.4e-4, 1e-15, "child tolerance decays with depth");
    EXPECT_NEAR(gpuboscia::depth_dependent_tolerance(1e-3, 0.8, 1e-6, 1000),
                1e-6, 1e-15, "child tolerance stops at configured minimum");
}

// ============================================================
//  BnBTree — push, pop_batch, empty, frontier_size
// ============================================================
void test_tree_push_pop() {
    std::cout << "\n[BnBTree push / pop_batch]\n";

    BnBTree tree(2, best_first, make_branch_fn());

    Node n1(1, -1, 0, {0.0,0.0}, {3.0,3.0}); n1.dual_bound = 1.0;
    Node n2(2, -1, 0, {0.0,0.0}, {3.0,3.0}); n2.dual_bound = 0.5;
    Node n3(3, -1, 0, {0.0,0.0}, {3.0,3.0}); n3.dual_bound = 2.0;

    tree.push(n1); tree.push(n2); tree.push(n3);
    EXPECT_EQ(tree.frontier_size(), 3, "3 nodes after push");

    auto batch = tree.pop_batch(2);
    EXPECT_EQ((int)batch.size(), 2, "pop_batch(2) returns 2 nodes");
    EXPECT_EQ(batch[0].id, 2,       "first popped: lowest dual_bound (n2)");
    EXPECT_EQ(batch[1].id, 1,       "second popped: next lowest (n1)");
    EXPECT_EQ(tree.frontier_size(), 1, "1 node remains");

    auto rest = tree.pop_batch(100);
    EXPECT_EQ((int)rest.size(), 1, "pop capped at remaining");
    EXPECT(tree.empty(),            "tree empty after drain");
}

// ============================================================
//  BnBTree — branch_multi (2^k children)
// ============================================================
void test_tree_branch_multi() {
    std::cout << "\n[BnBTree branch_multi — 2^k branching]\n";

    BnBTree tree(2, best_first, make_branch_fn());

    Node parent(1, -1, 0, {0.0, 0.0}, {4.0, 4.0});
    parent.dual_bound = 0.5;

    // x[1]=2.5 → dist=0.5 (most infeasible)
    // x[0]=1.7 → dist=0.3
    // top_k(k=3) → actual_k=2 (only 2 fractional), 2^2 = 4 children
    std::vector<double> x = {1.7, 2.5};
    auto children = tree.branch_multi(parent, x, {0, 1});

    EXPECT_EQ((int)children.size(), 4, "4 children from 2 fractional vars");

    for (const auto& c : children) {
        EXPECT_EQ(c.parent_id, 1, "parent_id correct");
        EXPECT_EQ(c.depth,     1, "depth = parent.depth + 1");
    }
    EXPECT_NE(children[0].id, children[1].id, "children have distinct IDs");

    // mask=0 (00): ub[1]=floor(2.5)=2, ub[0]=floor(1.7)=1
    EXPECT_NEAR(children[0].ub[1], 2.0, 1e-12, "mask=0: ub[1] = floor(2.5) = 2");
    EXPECT_NEAR(children[0].ub[0], 1.0, 1e-12, "mask=0: ub[0] = floor(1.7) = 1");

    // mask=3 (11): lb[1]=ceil(2.5)=3, lb[0]=ceil(1.7)=2
    EXPECT_NEAR(children[3].lb[1], 3.0, 1e-12, "mask=3: lb[1] = ceil(2.5) = 3");
    EXPECT_NEAR(children[3].lb[0], 2.0, 1e-12, "mask=3: lb[0] = ceil(1.7) = 2");
}

// ============================================================
//  BnBTree — should_prune, update_incumbent
// ============================================================
void test_tree_prune_incumbent() {
    std::cout << "\n[BnBTree prune / incumbent]\n";

    BnBTree tree(2, best_first, make_branch_fn());

    tree.update_incumbent(5.0, {1.0, 2.0});
    EXPECT_NEAR(tree.incumbent_value, 5.0, 1e-12, "incumbent set to 5.0");

    Node below(1,-1,0,{},{}); below.dual_bound = 4.99;
    Node equal(2,-1,0,{},{}); equal.dual_bound = 5.0;
    Node above(3,-1,0,{},{}); above.dual_bound = 5.01;

    EXPECT(!tree.should_prune(below), "dual_bound < incumbent → keep");
    EXPECT( tree.should_prune(equal), "dual_bound == incumbent → prune");
    EXPECT( tree.should_prune(above), "dual_bound > incumbent → prune");

    tree.update_incumbent(9.0, {0.0, 0.0});
    EXPECT_NEAR(tree.incumbent_value, 5.0, 1e-12, "worse update ignored");

    tree.update_incumbent(3.0, {0.5, 0.5});
    EXPECT_NEAR(tree.incumbent_value, 3.0, 1e-12, "better update accepted");
    EXPECT_NEAR(tree.incumbent_x[0], 0.5, 1e-12, "incumbent_x updated");
}

// ============================================================
//  best_first node selector
// ============================================================
void test_best_first() {
    std::cout << "\n[best_first]\n";

    std::vector<Node> nodes = {
        Node(0,-1,0,{},{}),   // dual=3.0
        Node(1,-1,0,{},{}),   // dual=1.0  ← lowest
        Node(2,-1,0,{},{}),   // dual=2.0
    };
    nodes[0].dual_bound = 3.0;
    nodes[1].dual_bound = 1.0;
    nodes[2].dual_bound = 2.0;

    auto idx = best_first(nodes, 2);
    EXPECT_EQ((int)idx.size(), 2, "returns 2 indices");
    EXPECT_EQ(idx[0], 1,          "lowest dual_bound first");
    EXPECT_EQ(idx[1], 2,          "second-lowest next");

    auto all = best_first(nodes, 99);
    EXPECT_EQ((int)all.size(), 3, "capped at node count");
}

// ============================================================
//  depth_first node selector
// ============================================================
void test_depth_first() {
    std::cout << "\n[depth_first]\n";

    std::vector<Node> nodes = {
        Node(0,-1,1,{},{}),   // depth 1
        Node(1,-1,3,{},{}),   // depth 3  ← deepest
        Node(2,-1,2,{},{}),   // depth 2
    };

    auto idx = depth_first(nodes, 2);
    EXPECT_EQ((int)idx.size(), 2, "returns 2 indices");
    EXPECT_EQ(idx[0], 1,          "deepest node first");
    EXPECT_EQ(idx[1], 2,          "second-deepest next");
}

// ============================================================
//  most_infeasible utility
// ============================================================
void test_most_infeasible() {
    std::cout << "\n[most_infeasible]\n";

    std::vector<double> x = {1.1, 2.5, 3.9};
    EXPECT_EQ(most_infeasible(x, {0,1,2}), 1, "x[1]=2.5 is most infeasible");

    std::vector<double> x2 = {1.5, 2.5, 3.0};
    int v = most_infeasible(x2, {0,1,2});
    EXPECT(v == 0 || v == 1, "tie: picks one of the most-infeasible vars");

    std::vector<double> x3 = {1.0, 2.0, 3.0};
    EXPECT_EQ(most_infeasible(x3, {0,1,2}), 0, "all-integer: picks var 0");

    EXPECT_EQ(most_infeasible({2.7}, {0}), 0, "single var");
}

// ============================================================
//  top_k_infeasible utility
// ============================================================
void test_top_k_infeasible() {
    std::cout << "\n[top_k_infeasible]\n";

    // x[1]=2.5 (dist=0.5), x[0]=1.1 (dist=0.1), x[2]=3.0 (integer, excluded)
    std::vector<double> x = {1.1, 2.5, 3.0};
    auto top2 = top_k_infeasible(x, {0,1,2}, 2);
    EXPECT_EQ((int)top2.size(), 2, "2 fractional vars returned");
    EXPECT_EQ(top2[0], 1, "x[1]=2.5 is most infeasible (sorted first)");
    EXPECT_EQ(top2[1], 0, "x[0]=1.1 is second");

    auto top5 = top_k_infeasible(x, {0,1,2}, 5);
    EXPECT_EQ((int)top5.size(), 2, "k capped at fractional count");

    // All integer → empty result
    auto none = top_k_infeasible({1.0, 2.0}, {0,1}, 3);
    EXPECT_EQ((int)none.size(), 0, "all integer → empty");

    auto within_tol = top_k_infeasible({1.0 + 5e-5}, {0}, 1, 1e-4);
    EXPECT_EQ((int)within_tol.size(), 0,
              "custom integer tolerance excludes near-integer variable");

    auto outside_tol = top_k_infeasible({1.0 + 5e-5}, {0}, 1, 1e-6);
    EXPECT_EQ((int)outside_tol.size(), 1,
              "custom integer tolerance keeps fractional variable");
}

// ============================================================
//  branch_pow2 — 2^k children
// ============================================================
void test_branch_pow2() {
    std::cout << "\n[branch_pow2]\n";

    int next_id = 10;
    Node parent(5, -1, 0, {0.0, 0.0, 0.0}, {5.0, 5.0, 5.0});

    // x[0]=1.5 (dist=0.5), x[1]=2.5 (dist=0.5), x[2]=3.9 (dist=0.1)
    // top_k(k=3) → all 3 fractional, actual_k=3 → 8 children
    std::vector<double> x = {1.5, 2.5, 3.9};
    auto ch = branch_pow2(parent, next_id, x, {0,1,2}, 3);
    EXPECT_EQ((int)ch.size(), 8, "3 fractional vars → 2^3 = 8 children");

    for (auto& c : ch) {
        EXPECT_EQ(c.parent_id, 5, "parent_id");
        EXPECT_EQ(c.depth,     1, "depth");
    }

    // All-integer: no children
    std::vector<double> xi = {1.0, 2.0, 3.0};
    int nid2 = 1;
    auto empty = branch_pow2(parent, nid2, xi, {0,1,2}, 3);
    EXPECT_EQ((int)empty.size(), 0, "all-integer → 0 children");

    // Only 1 fractional var → 2^1 = 2 children
    std::vector<double> x1 = {1.0, 2.5, 3.0};
    auto two = branch_pow2(parent, nid2, x1, {0,1,2}, 3);
    EXPECT_EQ((int)two.size(), 2, "1 fractional var → 2 children");
    // x[1]=2.5 selected; mask=0: ub[1]=2; mask=1: lb[1]=3
    EXPECT_NEAR(two[0].ub[1], 2.0, 1e-12, "mask=0: ub[1]=floor(2.5)=2");
    EXPECT_NEAR(two[1].lb[1], 3.0, 1e-12, "mask=1: lb[1]=ceil(2.5)=3");
}

void test_node_resume_queue() {
    Node node(7, 3, 2, {0.0, 0.0}, {1.0, 1.0});
    node.dual_bound = -0.25;
    node.relaxation_x = {0.2, 0.8};
    node.relaxation_iterations = 17;
    ConcurrentQueue queue;
    queue.push(node);
    const auto batch = queue.pop_batch(1, 1.0);
    EXPECT_EQ(batch.size(), size_t(1), "resumed node remains queued");
    EXPECT_EQ(batch[0].id, 7, "resumption keeps node identity");
    EXPECT_EQ(batch[0].depth, 2, "resumption keeps node depth");
    EXPECT_EQ(batch[0].relaxation_x, node.relaxation_x, "queue preserves warm start");
    EXPECT_EQ(batch[0].relaxation_iterations, 17, "queue preserves iteration schedule");
    EXPECT_EQ(queue.best_bound(1.0), 1.0, "dispatch removes queued bound");
    EXPECT_NEAR(batch[0].dual_bound, -0.25, 1e-12, "resumption keeps certificate");
    queue.push(node);
    EXPECT_NEAR(queue.best_bound(1.0), -0.25, 1e-12, "queued resume counts in global bound");
    int next_id = 20;
    const auto children = branch_pow2(node, next_id, {0.2, 0.8}, {0, 1}, 1);
    EXPECT(!children.empty() && children[0].relaxation_x.empty()
               && children[0].relaxation_iterations == 0,
           "new branch children do not inherit an infeasible warm start");
}

// ============================================================
//  BoxLMO DICG primitives
// ============================================================
void test_box_lmo_dicg_primitives() {
    std::cout << "\n[BoxLMO DICG primitives]\n";

    const std::vector<double> lb   = {0.0, -2.0, 1.0, 0.0};
    const std::vector<double> ub   = {4.0,  3.0, 5.0, 2.0};
    const std::vector<double> x    = {0.0,  3.0, 3.0, 1.0};
    const std::vector<double> grad = {-2.0,  4.0, 1.0, -1.0};
    std::vector<double> v(4), a(4);

    for (size_t j = 0; j < grad.size(); ++j)
        v[j] = grad[j] > 0.0 ? lb[j] : ub[j];
    EXPECT_NEAR(v[0], 4.0, 1e-12, "forward LMO chooses upper bound for negative gradient");
    EXPECT_NEAR(v[1], -2.0, 1e-12, "forward LMO chooses lower bound for positive gradient");

    // Pass -grad so this minimization oracle implements the DICG away
    // maximization over the minimal face of x.
    std::vector<double> away_direction(grad.size());
    for (size_t j = 0; j < grad.size(); j++)
        away_direction[j] = -grad[j];
    box_inface_lmo_cpu(away_direction, x, lb, ub, a);
    EXPECT_NEAR(a[0], 0.0, 1e-12, "lower-active coordinate stays fixed in minimal face");
    EXPECT_NEAR(a[1], 3.0, 1e-12, "upper-active coordinate stays fixed in minimal face");
    EXPECT_NEAR(a[2], 5.0, 1e-12, "free coordinate maximizes the original gradient");
    EXPECT_NEAR(a[3], 0.0, 1e-12, "free coordinate uses the opposite bound when maximizing");

    std::vector<double> direction(4);
    for (size_t j = 0; j < direction.size(); j++)
        direction[j] = a[j] - v[j];
    const double gamma_max =
        box_dicg_maximum_step_cpu(direction, x, lb, ub);
    EXPECT_NEAR(gamma_max, 0.5, 1e-12,
                "maximum step keeps x - gamma*(a-v) inside the box");

    for (size_t j = 0; j < x.size(); j++) {
        const double next = x[j] - gamma_max * direction[j];
        EXPECT(next >= lb[j] - 1e-12 && next <= ub[j] + 1e-12,
               "maximum-step DICG update is box feasible");
    }
}

void test_unit_simplex_lmo_dicg_primitives() {
    std::cout << "\n[UnitSimplexLMO DICG primitives]\n";

    const double rhs = 5.0;
    const std::vector<unsigned char> is_int = {1, 1, 0, 0};
    const std::vector<double> lb   = {1.0, 0.0, 0.0, 0.0};
    const std::vector<double> ub   = {3.0, 4.0, 5.0, 5.0};
    const std::vector<double> x    = {1.0, 2.0, 0.0, 1.0};
    const std::vector<double> grad = {-2.0, 1.0, 3.0, -4.0};
    const std::vector<double> away_direction = {2.0, -1.0, -3.0, 4.0};
    std::vector<double> a(4);

    unit_simplex_inface_lmo_cpu(
        rhs, is_int, away_direction, x, lb, ub, a);
    EXPECT_NEAR(a[0], 1.0, 1e-12,
                "simplex integer variable at lower bound remains fixed");
    EXPECT_NEAR(a[1], 4.0, 1e-12,
                "simplex free integer variable receives bounded mass");
    EXPECT_NEAR(a[2], 0.0, 1e-12,
                "simplex continuous variable at zero remains fixed");
    EXPECT_NEAR(a[3], 0.0, 1e-12,
                "positive in-face direction receives no mass");
    EXPECT(unit_simplex_inface_feasible_cpu(rhs, a, x),
           "simplex in-face point is feasible");

    const std::vector<double> forward = {3.0, 0.0, 0.0, 2.0};
    std::vector<double> direction(4);
    for (size_t j = 0; j < direction.size(); j++)
        direction[j] = a[j] - forward[j];
    const double gamma_max = unit_simplex_dicg_maximum_step_cpu(
        rhs, is_int, direction, x, lb, ub);
    EXPECT_NEAR(gamma_max, 0.5, 1e-12,
                "simplex maximum step respects integer lower bound");

    const std::vector<double> equality_x = {1.0, 2.0, 0.0, 2.0};
    const std::vector<double> leaves_equality = {1.0, 2.0, 0.0, 1.0};
    EXPECT(!unit_simplex_inface_feasible_cpu(
               rhs, leaves_equality, equality_x),
           "active simplex inequality must remain active in its minimal face");

    std::vector<double> equality_away(4);
    const std::vector<double> all_positive = {4.0, 3.0, 2.0, 1.0};
    unit_simplex_inface_lmo_cpu(
        rhs, is_int, all_positive, equality_x, lb, ub, equality_away);
    EXPECT(unit_simplex_inface_feasible_cpu(rhs, equality_away, equality_x),
           "active simplex face forces full mass even for positive direction");
    EXPECT(unit_simplex_linear_feasible_cpu(rhs, {1.0, 2.0, 0.0, 1.0}),
           "nonnegative point below simplex rhs is linearly feasible");
    EXPECT(!unit_simplex_linear_feasible_cpu(rhs, {1.0, -0.1, 0.0, 1.0}),
           "negative simplex coordinate is infeasible");
}

void test_equality_simplex_lmo_primitives() {
    std::cout << "\n[EqualitySimplexLMO primitives]\n";

    const std::vector<double> lb = {1.0, 0.0, 0.0, 0.0};
    const std::vector<double> ub = {3.0, 4.0, 5.0, 5.0};
    EXPECT(equality_simplex_linear_feasible_cpu(
               5.0, {1.0, 0.0, 0.0, 4.0}),
           "full-mass equality simplex vertex is feasible");
    EXPECT(!equality_simplex_linear_feasible_cpu(
               5.0, {1.0, 0.0, 0.0, 3.0}),
           "point below rhs is infeasible for equality simplex");

    const std::vector<unsigned char> is_int = {1, 1, 0, 0};
    const std::vector<double> x = {1.0, 2.0, 0.0, 2.0};
    std::vector<double> a(4);
    unit_simplex_inface_lmo_cpu(
        5.0, is_int, {4.0, 3.0, 2.0, 1.0}, x, lb, ub, a,
        1e-6, 1e-4, true);
    EXPECT_NEAR(std::accumulate(a.begin(), a.end(), 0.0), 5.0, 1e-12,
                "equality in-face oracle always preserves full mass");

    EXPECT_NEAR(unit_simplex_dicg_maximum_step_cpu(
                    5.0, is_int, {0.0, 0.0, 1.0, 0.0}, x, lb, ub,
                    1e-6, true),
                0.0, 1e-12,
                "equality simplex rejects directions outside affine hull");
}

// ============================================================
//  main
// ============================================================
void test_queue_deferred_pruning() {
    ConcurrentQueue queue;
    for (int i = 0; i < 4; ++i) {
        Node nd(i, i < 2 ? 10 : 20, 1, {0.0}, {1.0});
        nd.dual_bound = i;
        queue.push(nd);
    }
    EXPECT(queue.pop_full_batch(3, 1.5).empty(),
           "full batch does not submit nodes already cut off");
    EXPECT(queue.size() == 2, "live nodes retained when full batch unavailable");
    Node other(4, 20, 1, {0.0}, {1.0});
    other.dual_bound = 0.5;
    queue.push(other);
    queue.discard_children(10);
    auto batch = queue.pop_batch(3, 1.5);
    EXPECT(batch.size() == 1 && batch[0].id == 4,
           "deferred parent pruning removes only its queued children");
}



static void test_quadratic_exact_step() {
    EXPECT_NEAR(quadratic_exact_step(3, 6, 1), 0.5, 1e-14, "quadratic interior minimum");
    EXPECT_EQ(quadratic_exact_step(3, 2, 0.25), 0.25, "quadratic step obeys feasible upper bound");
    EXPECT_EQ(quadratic_exact_step(-3, 6, 1), 0, "quadratic uphill direction stays put");
    EXPECT_EQ(quadratic_exact_step(3, 0, 1), 1, "linear descent takes endpoint");
    EXPECT_EQ(quadratic_exact_step(0, 0, 1), 0, "flat direction stays put");
    EXPECT_EQ(quadratic_exact_step(-1, -4, 1), 1, "concave restriction compares endpoints");
    EXPECT_EQ(quadratic_exact_step(3, 6, 0), 0, "zero feasible step");
    EXPECT_NEAR(quadratic_exact_step(3e-100, 6e-100, 1), 0.5, 1e-14,
                "small positive curvature is not discarded");
    char prog[] = "test", flag[] = "--line-search", value[] = "quadratic-exact";
    char* argv[] = {prog, flag, value};
    GPUBosciaExampleOptions opts;
    EXPECT_EQ(parse_gpuboscia_example_options(3, argv, opts), -1, "exact CLI option accepted");
    EXPECT(opts.line_search.kind == LineSearchKind::QuadraticExact, "exact CLI policy selected");
}

int main() {
    test_quadratic_exact_step();
    std::cout << "=== GPUBoscia CPU Unit Tests ===\n";

    test_is_integer_feasible();
    test_queue_deferred_pruning();
    test_is_node_feasible();
    test_gap_tolerance();
    test_certified_fw_lower_bound();
    test_tree_push_pop();
    test_tree_branch_multi();
    test_tree_prune_incumbent();
    test_best_first();
    test_depth_first();
    test_most_infeasible();
    test_top_k_infeasible();
    test_branch_pow2();
    test_node_resume_queue();
    test_box_lmo_dicg_primitives();
    test_unit_simplex_lmo_dicg_primitives();
    test_equality_simplex_lmo_primitives();

    std::cout << "\n=== Results: "
              << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}

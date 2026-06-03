// main.cpp
// Entry point and benchmarking harness for Lie-Algebraic Asymptotic
// Mesh Optimization (LAAMO). Measures computational throughput and
// energy-related metrics for core linear-algebraic kernels using OpenMP.
// The benchmarks collect repeated timings and hardware counters to
// compute mean/standard-deviation and precision-per-watt (PPW) metrics.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <string_view>
#include <optional>
#include <omp.h>

#include "core/spatial_lsh.hpp"
#include "core/lie_operator.hpp"
#include "fem/hardware_metrics.hpp"

// Compile-time selection: enable invariant monitoring in non-release builds
#ifdef NDEBUG
  using Monitor = atlas::InvariantMonitorOFF;
#else
  using Monitor = atlas::InvariantMonitorON;
#endif

namespace {

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
inline TimePoint now() noexcept { return Clock::now(); }
inline double elapsed_ms(TimePoint t0, TimePoint t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1-t0).count();
}

constexpr uint64_t thread_seed(uint64_t base, int tid) noexcept {
    return base ^ (static_cast<uint64_t>(tid)*UINT64_C(6364136223846793005)
                   + UINT64_C(1442695040888963407));
}

// Compute the arithmetic mean of vector elements
// Precondition: container may be empty; returns 0 for empty input.
double mean(const std::vector<double>& v) noexcept {
    if (v.empty()) return 0.0;
    double s = 0; for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}
double stddev(const std::vector<double>& v, double m) noexcept {
    if (v.size() < 2) return 0.0;
    double s = 0; for (double x : v) s += (x-m)*(x-m);
    return std::sqrt(s / static_cast<double>(v.size()));
}

} // anonymous namespace

// ===========================================================================
// Phase runners (each returns elapsed ms, accepts repeat count)
// ===========================================================================

// Phase: Populate node positions and compute 3D Morton keys.
// Returns elapsed time in milliseconds.
static double phase_morton(std::vector<atlas::Node3D>& nodes,
                            std::size_t N, uint64_t seed = 1337ULL) {
    const auto t0 = now();
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        std::mt19937_64 rng(thread_seed(seed, tid));
        std::uniform_real_distribution<double> dist(-100, 100);
        #pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < N; ++i)
            nodes[i] = atlas::Node3D::make(dist(rng),dist(rng),dist(rng),
                                           static_cast<uint32_t>(i/8));
        #pragma omp for schedule(static)
        for (std::size_t i = 0; i < N; ++i)
            nodes[i].morton_key = atlas::compute_morton_3d(
                nodes[i].x, nodes[i].y, nodes[i].z, 10.0, 100.0);
    }
    return elapsed_ms(t0, now());
}

// Phase: Compute matrix exponential for each input using a fixed
// generator; validate via the invariant monitor.
// Returns elapsed time in milliseconds.
static double phase_exponential(std::vector<atlas::Matrix3x3>& out, std::size_t N,
                                 Monitor& mon) {
    const atlas::Matrix3x3 xi{
         0.10, 0.02, 0.05, -0.02, 0.15,-0.03, -0.05,-0.03,-0.25};
    const auto t0 = now();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = atlas::exponential_map(xi);
        mon.check_sl3_element(out[i]);
    }
    return elapsed_ms(t0, now());
}

// Phase: Apply Lie transport operation to a base matrix for each element.
// This measures throughput of the Lie-transport kernel.
static double phase_transport(std::vector<atlas::Matrix3x3>& out, std::size_t N,
                               Monitor& mon) {
    const atlas::Matrix3x3 xi{
         0.10, 0.02, 0.05, -0.02, 0.15,-0.03, -0.05,-0.03,-0.25};
    const atlas::Matrix3x3 G = atlas::exponential_map(xi);
    const auto t0 = now();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = atlas::lie_transport(G, xi);
        mon.check_sl3_element(out[i]);
    }
    return elapsed_ms(t0, now());
}

// Phase: Compute SL(3) retraction for small tangent vectors.
// The operation is representative of retraction kernels used in geometry-aware solvers.
static double phase_retraction(std::vector<atlas::Matrix3x3>& out, std::size_t N,
                                Monitor& mon) {
    const atlas::Matrix3x3 xi{
         0.05, 0.01, 0.02, -0.01, 0.07,-0.01, -0.02,-0.01,-0.12};
    const atlas::Matrix3x3 G = atlas::exponential_map(xi);
    const atlas::Matrix3x3 V = atlas::project_to_sl3(
        atlas::Matrix3x3{0.01,0.005,0, 0,0.01,0.002, 0,0,-0.02});
    const auto t0 = now();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = atlas::sl3_retraction(G, V);
        mon.check_sl3_element(out[i]);
    }
    return elapsed_ms(t0, now());
}

// Repeated timing helper: execute phase function multiple times and
// return mean and sample standard deviation of the measured durations.
template<typename PhaseF>
std::pair<double,double> timed_mean(PhaseF&& f, int repeats) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r) samples.push_back(f());
    const double m = mean(samples);
    return {m, stddev(samples, m)};
}

// Program entry: configure benchmarks, run kernel phases, and report
// timing, hardware counters, and derived precision/energy metrics.
int main() {
    constexpr std::size_t N_NODES = 500'000ULL;
    constexpr int         REPEATS = 5;

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  FlexMesh LAAMO   — Production Engine                     ║\n"
              << "║  Lie-Algebraic Asymptotic Mesh Optimization                  ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "  Threads            : " << omp_get_max_threads() << "\n"
              << "  Nodes              : " << N_NODES << "\n"
              << "  Timing repeats     : " << REPEATS << "\n";

    std::cout << "  Hardware counters  : "
              << "RAPL/perf_event_open when available, otherwise unavailable\n";

#ifndef NDEBUG
    std::cout << "  Invariant monitor  : ON (debug build)\n\n";
#else
    std::cout << "  Invariant monitor  : OFF (release build)\n\n";
#endif

    Monitor mon;
    std::vector<atlas::Node3D>    nodes(N_NODES);
    std::vector<atlas::Matrix3x3> outputs(N_NODES);

    // ── Condition estimate on reference matrix ─────────────────────────────
    const atlas::Matrix3x3 xi_ref{
         0.10, 0.02, 0.05, -0.02, 0.15,-0.03, -0.05,-0.03,-0.25};
    std::cout << std::fixed << std::setprecision(3)
              << "  κ(ξ_ref)           : "
              << atlas::matrix_condition_estimate(xi_ref) << "\n\n";

    auto phase_profile = [&](auto&& phase_fn) {
        atlas::hpc::ScopedHardwareProfiler profiler;
        profiler.start();
        const auto result = phase_fn();
        auto metrics = profiler.stop();
        if (metrics.energy_joules) {
            std::cout << "  Energy (J)         : " << *metrics.energy_joules << "\n";
        } else {
            std::cout << "  Energy (J)         : n/a\n";
        }
        std::cout << "  Cycles             : " << metrics.cpu_cycles << "\n"
                  << "  Instructions       : " << metrics.instructions << "\n"
                  << "  LLC misses         : " << metrics.llc_misses << "\n"
                  << "  IPC                : " << metrics.ipc() << "\n";
        return std::pair{result, metrics};
    };

    // ── Phase 1: Morton ────────────────────────────────────────────────────
    std::cout << "─── Phase 1: Spatial LSH Morton Key Generation ──────────────────\n";
    const auto [phase1_time, phase1_metrics] = phase_profile([&]{ return timed_mean([&]{ return phase_morton(nodes, N_NODES); }, REPEATS); });
    const auto [dt1, sd1] = phase1_time;
    const std::optional<double> e1_j = phase1_metrics.energy_joules;
    std::cout << std::fixed << std::setprecision(3)
              << "  Mean time          : " << dt1 << " ms  (σ=" << sd1 << " ms)\n"
              << "  Throughput         : " << N_NODES/(dt1*1e3) << " MNodes/s\n"
              << "  Energy             : " << (e1_j ? std::to_string(*e1_j) : std::string("n/a")) << " J\n\n";

    // ── Phase 2: Exponential map ───────────────────────────────────────────
    std::cout << "─── Phase 2: Lie Exponential Map (Padé [3/3]) ───────────────────\n";
    const auto [phase2_time, phase2_metrics] = phase_profile([&]{ return timed_mean([&]{ return phase_exponential(outputs, N_NODES, mon); }, REPEATS); });
    const auto [dt2, sd2] = phase2_time;
    const std::optional<double> e2_j = phase2_metrics.energy_joules;
    std::cout << "  Mean time          : " << dt2 << " ms  (σ=" << sd2 << " ms)\n"
              << "  Throughput         : " << N_NODES/(dt2*1e3) << " MOps/s\n"
              << "  Energy             : " << (e2_j ? std::to_string(*e2_j) : std::string("n/a")) << " J\n\n";

    // ── Phase 3: Lie transport (NEW) ───────────────────────────────────────
    std::cout << "─── Phase 3: Geodesic Lie Transport (     ) ─────────────────\n";
    const auto [phase3_time, phase3_metrics] = phase_profile([&]{ return timed_mean([&]{ return phase_transport(outputs, N_NODES, mon); }, REPEATS); });
    const auto [dt3, sd3] = phase3_time;
    const std::optional<double> e3_j = phase3_metrics.energy_joules;
    std::cout << "  Mean time          : " << dt3 << " ms  (σ=" << sd3 << " ms)\n"
              << "  Throughput         : " << N_NODES/(dt3*1e3) << " MOps/s\n"
              << "  Energy             : " << (e3_j ? std::to_string(*e3_j) : std::string("n/a")) << " J\n\n";

    // ── Phase 4: SL3 retraction (NEW) ─────────────────────────────────────
    std::cout << "─── Phase 4: SL(3) Retraction (     ) ──────────────────────\n";
    const auto [phase4_time, phase4_metrics] = phase_profile([&]{ return timed_mean([&]{ return phase_retraction(outputs, N_NODES, mon); }, REPEATS); });
    const auto [dt4, sd4] = phase4_time;
    const std::optional<double> e4_j = phase4_metrics.energy_joules;
    std::cout << "  Mean time          : " << dt4 << " ms  (σ=" << sd4 << " ms)\n"
              << "  Throughput         : " << N_NODES/(dt4*1e3) << " MOps/s\n"
              << "  Energy             : " << (e4_j ? std::to_string(*e4_j) : std::string("n/a")) << " J\n\n";

    // ── Precision metrics ──────────────────────────────────────────────────
    std::cout << "─── Precision & Efficiency Metrics ──────────────────────────────\n";
    const atlas::Matrix3x3 A = atlas::exponential_map(xi_ref);
    const atlas::Matrix3x3 diff = atlas::matrix_sub(
        atlas::exponential_map(atlas::matrix_logarithm(A)), A);
    const double rte = atlas::matrix_frobenius_norm(diff)
                     / atlas::matrix_frobenius_norm(A);
    const bool energy_available = e1_j.has_value() && e2_j.has_value() && e3_j.has_value() && e4_j.has_value();
    const double total_J = energy_available ? (*e1_j + *e2_j + *e3_j + *e4_j) : std::numeric_limits<double>::quiet_NaN();
    const double ppw = (energy_available && rte > 0.0 && total_J > 0.0)
                     ? 1.0 / (rte * total_J)
                     : std::numeric_limits<double>::quiet_NaN();

    std::cout << std::scientific << std::setprecision(4)
              << "  Round-trip error             : " << rte << "\n"
              << std::fixed << std::setprecision(4)
              << "  Total energy                 : " << (energy_available ? std::to_string(total_J) : std::string("n/a")) << " J\n"
              << std::scientific << std::setprecision(4)
              << "  Precision-per-Watt (PPW)     : " << (energy_available ? std::to_string(ppw) : std::string("n/a")) << "\n\n";

    // ── Invariant monitor report ───────────────────────────────────────────
    mon.report();

    // ── Summary table ─────────────────────────────────────────────────────
    std::cout << "\n┌───────────────────────────────────┬──────────────────┐\n"
              << "│ Metric                            │  FlexMesh LAAMO  │\n"
              << "├───────────────────────────────────┼──────────────────┤\n";
    std::cout << std::fixed << std::setprecision(2)
              << "│ Phase1 Morton    (ms, mean±σ)     │ "
              << std::setw(7) << dt1 << " ± " << std::setw(5) << sd1 << "  │\n"
              << "│ Phase2 ExpMap    (ms, mean±σ)     │ "
              << std::setw(7) << dt2 << " ± " << std::setw(5) << sd2 << "  │\n"
              << "│ Phase3 Transport (ms, mean±σ)     │ "
              << std::setw(7) << dt3 << " ± " << std::setw(5) << sd3 << "  │\n"
              << "│ Phase4 Retract   (ms, mean±σ)     │ "
              << std::setw(7) << dt4 << " ± " << std::setw(5) << sd4 << "  │\n";
    std::cout << std::fixed << std::setprecision(4)
              << "│ Total energy (J)                  │ "
              << std::setw(14) << (energy_available ? total_J : 0.0) << "  │\n";
    std::cout << std::scientific << std::setprecision(4)
              << "│ Round-trip error                  │ "
              << std::setw(14) << rte << "  │\n"
              << "│ Precision-per-Watt                │ "
              << std::setw(14) << (energy_available ? ppw : 0.0) << "  │\n"
              << "└───────────────────────────────────┴──────────────────┘\n\n";

    volatile double sink = outputs[N_NODES/2].data[0]; (void)sink;
    std::cout << "[FlexMesh LAAMO  ] Execution complete.\n";
    return 0;
}

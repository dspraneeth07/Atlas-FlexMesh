// =============================================================================
// src/pathological_suite.cpp
// Numerical stress-test harness for small-matrix Lie-algebra routines.
// Generates randomized 3×3 traceless matrices and validates core
// numerical invariants used by the LAAMO pipeline. Tests are lightweight
// algebraic checks intended for CI and benchmarking (not PDE validation).
//
// Invariants verified:
//  - I1: determinant preservation under Lie exponential: det(exp(ξ)) == 1
//  - I2: logarithm/exponential round-trip fidelity: exp(log(A)) ≈ A
//  - I3: tracelessness of Lie elements: tr(ξ) == 0
//
// The harness exercises regimes that stress conditioning, near-singularity,
// and long-horizon chaining to reveal drift or numerical instability.
// =============================================================================

#include "core/lie_operator.hpp"
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <algorithm>

namespace atlas::pathological {

struct TestResult {
    uint64_t n_tested{0};
    uint64_t n_pass_i1{0};  ///< I1: det(exp(ξ)) = 1
    uint64_t n_pass_i2{0};  ///< I2: round-trip error ≤ tol
    uint64_t n_pass_i3{0};  ///< I3: traceless algebra
    double   max_det_err{0.0};
    double   max_rte{0.0};
    double   max_trace_err{0.0};
};

// ---------------------------------------------------------------------------
// run_pathological_suite
//  Randomized test driver that constructs traceless matrices in several
//  regimes (near-identity, large-norm, nilpotent-like, symmetric,
//  near-singular) and evaluates the numerical invariants with
//  conservative tolerances chosen for robustness in CI.
// ---------------------------------------------------------------------------
[[nodiscard]]
TestResult run_pathological_suite(uint64_t n_trials = 1'000'000ULL,
                                  uint64_t seed     = 0xDEADBEEFCAFEULL,
                                  bool     verbose  = false) noexcept {
    TestResult res{};
    std::mt19937_64 rng(seed);

    // Reusable distributions
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> small(-0.3, 0.3);  // near-identity
    std::uniform_real_distribution<double> large(-5.0, 5.0);  // large norm
    std::uniform_int_distribution<int>     regime(0, 4);      // test regime

    for (uint64_t t = 0; t < n_trials; ++t) {
        const int r = regime(rng);

        // Build a test matrix ξ ∈ sl(3) (traceless) in selected regime.
        double a,b,c,d,e,f,g,h;
        switch (r) {
        case 0: // Near-identity deviatoric
            a=small(rng); b=small(rng); c=small(rng);
            d=small(rng); e=small(rng); f=small(rng);
            g=small(rng); h=small(rng);
            break;
        case 1: // Large-norm deviatoric
            a=large(rng); b=unit(rng);  c=unit(rng);
            d=unit(rng);  e=large(rng); f=unit(rng);
            g=unit(rng);  h=unit(rng);
            break;
        case 2: // Near-nilpotent (off-diagonals only)
            a=0.0;        b=unit(rng);  c=unit(rng);
            d=unit(rng);  e=0.0;        f=unit(rng);
            g=unit(rng);  h=unit(rng);
            break;
        case 3: // Symmetric deviatoric
            a=unit(rng);  b=unit(rng);  c=unit(rng);
            d=b;          e=unit(rng);  f=unit(rng);
            g=c;          h=f;
            break;
        case 4: // Near-singular element
            a=1.0e-6*unit(rng); b=1.0e-6*unit(rng); c=1.0e-6*unit(rng);
            d=1.0e-6*unit(rng); e=1.0e-6*unit(rng); f=1.0e-6*unit(rng);
            g=1.0e-6*unit(rng); h=1.0e-6*unit(rng);
            break;
        default:
            a=b=c=d=e=f=g=h=0.0;
        }

        // Enforce tracelessness: set third diagonal to preserve trace=0
        const double k_diag = -(a + e);
        const Matrix3x3 xi{
             a, b, c,
             d, e, f,
             g, h, k_diag
        };

        // I3: verify tracelessness (numerical tolerance accounts for rng and
        // construction rounding). Threshold is set near machine precision.
        const double tr_err = std::abs(matrix_trace(xi));
        res.max_trace_err = std::max(res.max_trace_err, tr_err);
        const bool i3 = (tr_err < 1.0e-12);
        if (i3) ++res.n_pass_i3;

        // I1: determinant preservation under exponential. We allow a small
        // multiplicative drift tolerance to account for conditioning and
        // scaling in pathological inputs.
        const Matrix3x3 expXi = exponential_map(xi);
        const double det_err  = std::abs(matrix_determinant(expXi) - 1.0);
        res.max_det_err = std::max(res.max_det_err, det_err);
        const bool i1 = (det_err < 1.0e-7);
        if (i1) ++res.n_pass_i1;
        else if (verbose) {
            std::printf("[FAIL I1] trial=%lu  regime=%d  det_err=%.3e\n",
                        (unsigned long)t, r, det_err);
        }

        // I2: logarithm ↔ exponential round-trip fidelity. Compute
        // normalized Frobenius relative error and compare to a conservative
        // tolerance; this exercises both log and exp numerical stability.
        const Matrix3x3 logA     = matrix_logarithm(expXi);
        const Matrix3x3 expLogA  = exponential_map(logA);
        const double frob_A      = matrix_frobenius_norm(expXi);
        const double rte = (frob_A > 1.0e-30)
            ? matrix_frobenius_norm(matrix_sub(expLogA, expXi)) / frob_A
            : matrix_frobenius_norm(matrix_sub(expLogA, expXi));
        res.max_rte = std::max(res.max_rte, rte);
        const bool i2 = (rte < 1.0e-6);
        if (i2) ++res.n_pass_i2;
        else if (verbose) {
            std::printf("[FAIL I2] trial=%lu  regime=%d  rte=%.3e\n",
                        (unsigned long)t, r, rte);
        }

        ++res.n_tested;
    }
    return res;
}

// ---------------------------------------------------------------------------
// run_long_horizon_drift
//  Repeatedly apply log then exp to a fixed SL(3) element to measure
//  cumulative determinant drift over many iterations. Useful for detecting
//  slow multiplicative errors that accumulate in long adaptation horizons.
// ---------------------------------------------------------------------------
[[nodiscard]]
double run_long_horizon_drift(uint64_t n_steps = 100'000ULL) noexcept {
    // Start with a well-conditioned SL(3) element.
    const Matrix3x3 xi_ref{
         0.10,  0.02,  0.05,
        -0.02,  0.15, -0.03,
        -0.05, -0.03, -0.25
    };

    Matrix3x3 G = exponential_map(xi_ref);
    double max_det_drift = 0.0;

    for (uint64_t step = 0; step < n_steps; ++step) {
        const Matrix3x3 logG = matrix_logarithm(G);
        G = exponential_map(logG);

        const double det = matrix_determinant(G);
        const double drift = std::abs(det - 1.0);
        if (drift > max_det_drift) max_det_drift = drift;
    }
    return max_det_drift;
}

// ---------------------------------------------------------------------------
// print_suite_report
// ---------------------------------------------------------------------------
void print_suite_report(const TestResult& r) noexcept {
    const double pct_i1 = 100.0 * r.n_pass_i1 / r.n_tested;
    const double pct_i2 = 100.0 * r.n_pass_i2 / r.n_tested;
    const double pct_i3 = 100.0 * r.n_pass_i3 / r.n_tested;

    std::printf("\n══ Algebraic Stress-Test Suite ══════════════════════════════\n");
    std::printf("  Trials tested      : %lu\n", (unsigned long)r.n_tested);
    std::printf("  I1 pass (det=1)    : %lu / %lu  (%.3f%%)\n",
                (unsigned long)r.n_pass_i1, (unsigned long)r.n_tested, pct_i1);
    std::printf("  I2 pass (round-trip): %lu / %lu  (%.3f%%)\n",
                (unsigned long)r.n_pass_i2, (unsigned long)r.n_tested, pct_i2);
    std::printf("  I3 pass (traceless) : %lu / %lu  (%.3f%%)\n",
                (unsigned long)r.n_pass_i3, (unsigned long)r.n_tested, pct_i3);
    std::printf("  Max det error      : %.3e\n", r.max_det_err);
    std::printf("  Max round-trip err : %.3e\n", r.max_rte);
    std::printf("  Max trace error    : %.3e\n", r.max_trace_err);

    const bool all_ok = (pct_i1 > 99.0 && pct_i2 > 99.0 && pct_i3 > 99.99);
    std::printf("  Overall            : %s\n",
                all_ok ? "✓ PASS" : "✗ SOME FAILURES — inspect above");
    std::printf("═════════════════════════════════════════════════════════════\n");
}

// ===========================================================================
// CI INTEGRATION & STATISTICAL ANALYSIS FRAMEWORK
// ===========================================================================

/// Statistical histogram for error distribution analysis
struct ErrorHistogram {
    std::vector<uint64_t> bins;           ///< Bin counts (log10 scale)
    double                bin_min;        ///< log10(min_error)
    double                bin_max;        ///< log10(max_error)
    double                mean_error;
    double                std_error;
    double                median_error;
};

[[nodiscard]]
ErrorHistogram build_error_histogram(const std::vector<double>& errors) noexcept {
    ErrorHistogram hist;
    if (errors.empty()) return hist;
    
    static constexpr int N_BINS = 16;
    hist.bins.resize(N_BINS, 0);
    
    // Find error range
    double min_err = 1.0e30, max_err = 1.0e-30;
    double sum = 0.0, sum_sq = 0.0;
    for (double e : errors) {
        if (e > 1.0e-30 && e < 1.0e30) {
            min_err = std::min(min_err, e);
            max_err = std::max(max_err, e);
            sum += e;
            sum_sq += e*e;
        }
    }
    
    hist.bin_min = std::log10(std::max(min_err, 1.0e-30));
    hist.bin_max = std::log10(std::min(max_err, 1.0e30));
    hist.mean_error = sum / errors.size();
    hist.std_error = std::sqrt(std::max(sum_sq / errors.size() - 
                                         hist.mean_error * hist.mean_error, 0.0));
    
    // Populate bins (log10 scale)
    const double bin_width = (hist.bin_max - hist.bin_min) / N_BINS;
    for (double e : errors) {
        if (e > 1.0e-30 && e < 1.0e30) {
            const int bin = static_cast<int>(
                (std::log10(e) - hist.bin_min) / bin_width);
            if (bin >= 0 && bin < N_BINS) {
                hist.bins[bin]++;
            }
        }
    }
    
    // Median
    std::vector<double> sorted_errors = errors;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    const size_t mid = sorted_errors.size() / 2;
    hist.median_error = sorted_errors[mid];
    
    return hist;
}

/// Print histogram to stdout
void print_error_histogram(const ErrorHistogram& hist) noexcept {
    std::printf("\n┌─ Error Distribution (log10 scale) ───────────────────┐\n");
    static constexpr int N_BINS = 16;
    for (int b = 0; b < N_BINS; ++b) {
        const double log_val = hist.bin_min + 
            (hist.bin_max - hist.bin_min) * (b + 0.5) / N_BINS;
        const uint64_t count = hist.bins[b];
        const int bar_len = static_cast<int>(count / 10);  // Scale for display
        
        std::printf("  [%.2e] |", std::pow(10.0, log_val));
        for (int i = 0; i < std::min(bar_len, 50); ++i) std::printf("█");
        std::printf("\n");
    }
    std::printf("  Mean:   %.3e\n", hist.mean_error);
    std::printf("  Median: %.3e\n", hist.median_error);
    std::printf("  StdDev: %.3e\n", hist.std_error);
    std::printf("└─────────────────────────────────────────────────────────┘\n");
}

/// CSV export for external analysis/plotting
void export_test_results_csv(const std::string& filename,
                             const TestResult& res,
                             uint64_t seed) noexcept {
    std::FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return;
    
    std::fprintf(f, "metric,value,seed,timestamp\n");
    std::fprintf(f, "n_tested,%lu,%lu,0\n", (unsigned long)res.n_tested, seed);
    std::fprintf(f, "n_pass_i1,%lu,%lu,0\n", (unsigned long)res.n_pass_i1, seed);
    std::fprintf(f, "n_pass_i2,%lu,%lu,0\n", (unsigned long)res.n_pass_i2, seed);
    std::fprintf(f, "n_pass_i3,%lu,%lu,0\n", (unsigned long)res.n_pass_i3, seed);
    std::fprintf(f, "max_det_err,%.6e,%lu,0\n", res.max_det_err, seed);
    std::fprintf(f, "max_rte,%.6e,%lu,0\n", res.max_rte, seed);
    std::fprintf(f, "max_trace_err,%.6e,%lu,0\n", res.max_trace_err, seed);
    
    const double pct_i1 = 100.0 * res.n_pass_i1 / res.n_tested;
    const double pct_i2 = 100.0 * res.n_pass_i2 / res.n_tested;
    const double pct_i3 = 100.0 * res.n_pass_i3 / res.n_tested;
    std::fprintf(f, "pct_pass_i1,%.3f,%lu,0\n", pct_i1, seed);
    std::fprintf(f, "pct_pass_i2,%.3f,%lu,0\n", pct_i2, seed);
    std::fprintf(f, "pct_pass_i3,%.3f,%lu,0\n", pct_i3, seed);
    
    std::fclose(f);
}

/// AUTOMATED REPRODUCIBILITY HARNESS
/// Run full test suite with multiple random seeds, aggregate statistics
struct ReproducibilityReport {
    uint64_t n_runs;
    double   mean_pass_rate_i1;
    double   mean_pass_rate_i2;
    double   mean_pass_rate_i3;
    double   std_pass_rate_i1;
    double   std_pass_rate_i2;
    double   std_pass_rate_i3;
    double   max_observed_det_error;
    double   max_observed_rte;
};

[[nodiscard]]
ReproducibilityReport run_reproducibility_suite(
    uint64_t n_runs = 10,
    uint64_t trials_per_run = 100'000ULL,
    bool export_csv = false) noexcept
{
    ReproducibilityReport report{};
    report.n_runs = n_runs;
    
    std::vector<double> pass_rates_i1, pass_rates_i2, pass_rates_i3;
    
    std::printf("\n╔══ REPRODUCIBILITY HARNESS ═════════════════════════════╗\n");
    std::printf("║ Running %lu independent test suites...\n", (unsigned long)n_runs);
    std::printf("║ Trials per run: %lu\n", (unsigned long)trials_per_run);
    std::printf("╚═════════════════════════════════════════════════════════╝\n\n");
    
    for (uint64_t run = 0; run < n_runs; ++run) {
        const uint64_t seed = 0xDEADBEEFCAFEULL + run;
        const TestResult res = run_pathological_suite(trials_per_run, seed, false);
        
        const double pct_i1 = 100.0 * res.n_pass_i1 / res.n_tested;
        const double pct_i2 = 100.0 * res.n_pass_i2 / res.n_tested;
        const double pct_i3 = 100.0 * res.n_pass_i3 / res.n_tested;
        
        pass_rates_i1.push_back(pct_i1);
        pass_rates_i2.push_back(pct_i2);
        pass_rates_i3.push_back(pct_i3);
        
        report.max_observed_det_error = std::max(report.max_observed_det_error, res.max_det_err);
        report.max_observed_rte = std::max(report.max_observed_rte, res.max_rte);
        
        if (export_csv) {
            char fname[256];
            std::snprintf(fname, sizeof(fname), "pathological_run_%lu.csv", (unsigned long)run);
            export_test_results_csv(fname, res, seed);
        }
        
        std::printf("Run %2lu: I1=%.3f%% I2=%.3f%% I3=%.3f%%\n",
                    (unsigned long)run, pct_i1, pct_i2, pct_i3);
    }
    
    // Compute statistics
    auto compute_mean_std = [](const std::vector<double>& vals) 
        -> std::pair<double, double> {
        if (vals.empty()) return {0.0, 0.0};
        double sum = 0.0, sum_sq = 0.0;
        for (double v : vals) { sum += v; sum_sq += v*v; }
        const double mean = sum / vals.size();
        const double var = (sum_sq / vals.size()) - mean * mean;
        return {mean, std::sqrt(std::max(var, 0.0))};
    };
    
    auto [mean_i1, std_i1] = compute_mean_std(pass_rates_i1);
    auto [mean_i2, std_i2] = compute_mean_std(pass_rates_i2);
    auto [mean_i3, std_i3] = compute_mean_std(pass_rates_i3);
    
    report.mean_pass_rate_i1 = mean_i1;
    report.mean_pass_rate_i2 = mean_i2;
    report.mean_pass_rate_i3 = mean_i3;
    report.std_pass_rate_i1 = std_i1;
    report.std_pass_rate_i2 = std_i2;
    report.std_pass_rate_i3 = std_i3;
    
    std::printf("\n┌─ STATISTICAL SUMMARY ────────────────────────────────┐\n");
    std::printf("│ I1 (det=1):       %.3f%% ± %.3f%%\n", mean_i1, std_i1);
    std::printf("│ I2 (round-trip):  %.3f%% ± %.3f%%\n", mean_i2, std_i2);
    std::printf("│ I3 (traceless):   %.3f%% ± %.3f%%\n", mean_i3, std_i3);
    std::printf("│ Max det error:    %.3e\n", report.max_observed_det_error);
    std::printf("│ Max round-trip:   %.3e\n", report.max_observed_rte);
    std::printf("└──────────────────────────────────────────────────────┘\n");
    
    return report;
}

} // namespace atlas::pathological

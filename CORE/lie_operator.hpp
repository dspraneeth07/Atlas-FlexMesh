#pragma once
// =============================================================================
// atlas/lie_operator.hpp
// Core linear-algebra utilities and Lie-group operations used by the
// mesh-adaptation and optimization pipeline. Implements robust, numerically
// stable 3×3 matrix primitives and Lie-algebraic operators on SL(3):
//   - high-accuracy scaling-and-squaring matrix exponential (Padé [3/3])
//   - robust matrix logarithm via repeated square roots + Gregory series
//   - Denman–Beavers square-root with stabilization and fallbacks
//   - adjoint-action transport and SL(3) retraction preserving det=1
// Comments focus on algorithmic invariants and numerical guards.
// =============================================================================

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <limits>
#include <atomic>

namespace atlas {

// ---------------------------------------------------------------------------
// Column-major 3×3 double-precision matrix stored as 9 contiguous doubles.
// Layout: `data[col * 3 + row]` matches LAPACK/Fortran column-major convention.
// The struct is padded to two 64-byte cache lines (128 B) to enable
// aligned vector loads on wide SIMD ISAs without unaligned tail handling.
// ---------------------------------------------------------------------------
struct alignas(64) Matrix3x3 {
    std::array<double, 9>  data{};   ///< 72 B  — actual matrix payload
    std::array<double, 7>  _pad{};   ///< 56 B  — alignment filler (never touched)

    // -----------------------------------------------------------------------
    constexpr Matrix3x3() noexcept = default;

    /// @brief Row-major initialiser (transposed into column-major storage).
    constexpr Matrix3x3(double a00, double a01, double a02,
                        double a10, double a11, double a12,
                        double a20, double a21, double a22) noexcept
        : _pad{}
    {
        data[0]=a00; data[1]=a10; data[2]=a20;   // col 0
        data[3]=a01; data[4]=a11; data[5]=a21;   // col 1
        data[6]=a02; data[7]=a12; data[8]=a22;   // col 2
    }

    // ── Element access ────────────────────────────────────────────────────
    [[nodiscard]] inline double  operator()(std::size_t r, std::size_t c) const noexcept { return data[c*3+r]; }
    [[nodiscard]] inline double& operator()(std::size_t r, std::size_t c)       noexcept { return data[c*3+r]; }
    [[nodiscard]] inline double  operator[](std::size_t i)                const noexcept { return data[i]; }
    [[nodiscard]] inline double& operator[](std::size_t i)                      noexcept { return data[i]; }

    // ── Factories ─────────────────────────────────────────────────────────
    [[nodiscard]] static constexpr Matrix3x3 identity() noexcept {
        return {1,0,0, 0,1,0, 0,0,1};
    }
    [[nodiscard]] static constexpr Matrix3x3 zero() noexcept { return Matrix3x3{}; }
};

static_assert(sizeof(Matrix3x3)  == 128, "Matrix3x3 must be 128 B");
static_assert(alignof(Matrix3x3) ==  64, "Matrix3x3 must be 64-B aligned");

// ===========================================================================
// Element-wise and small-matrix linear-algebra primitives. Implementations
// are deliberately simple and fully unrolled to enable compilers to emit
// efficient vectorized code and to expose independent FMAs for scheduling.
// ===========================================================================

[[nodiscard]] inline Matrix3x3
matrix_add(const Matrix3x3& A, const Matrix3x3& B) noexcept {
    Matrix3x3 C;
    for (std::size_t i = 0; i < 9; ++i) C.data[i] = A.data[i] + B.data[i];
    return C;
}

[[nodiscard]] inline Matrix3x3
matrix_sub(const Matrix3x3& A, const Matrix3x3& B) noexcept {
    Matrix3x3 C;
    for (std::size_t i = 0; i < 9; ++i) C.data[i] = A.data[i] - B.data[i];
    return C;
}

[[nodiscard]] inline Matrix3x3
matrix_scale(const Matrix3x3& A, double s) noexcept {
    Matrix3x3 C;
    for (std::size_t i = 0; i < 9; ++i) C.data[i] = A.data[i] * s;
    return C;
}

// ---------------------------------------------------------------------------
// Fully unrolled 3×3 DGEMM: explicit accumulation for each output element.
// The implementation exposes 27 independent multiply-add operations so the
// backend can schedule FMAs efficiently and vectorize across contiguous data.
// ---------------------------------------------------------------------------
[[nodiscard]] inline Matrix3x3
matrix_multiply(const Matrix3x3& A, const Matrix3x3& B) noexcept {
    Matrix3x3 C;
    // col 0
    C.data[0] = A.data[0]*B.data[0] + A.data[3]*B.data[1] + A.data[6]*B.data[2];
    C.data[1] = A.data[1]*B.data[0] + A.data[4]*B.data[1] + A.data[7]*B.data[2];
    C.data[2] = A.data[2]*B.data[0] + A.data[5]*B.data[1] + A.data[8]*B.data[2];
    // col 1
    C.data[3] = A.data[0]*B.data[3] + A.data[3]*B.data[4] + A.data[6]*B.data[5];
    C.data[4] = A.data[1]*B.data[3] + A.data[4]*B.data[4] + A.data[7]*B.data[5];
    C.data[5] = A.data[2]*B.data[3] + A.data[5]*B.data[4] + A.data[8]*B.data[5];
    // col 2
    C.data[6] = A.data[0]*B.data[6] + A.data[3]*B.data[7] + A.data[6]*B.data[8];
    C.data[7] = A.data[1]*B.data[6] + A.data[4]*B.data[7] + A.data[7]*B.data[8];
    C.data[8] = A.data[2]*B.data[6] + A.data[5]*B.data[7] + A.data[8]*B.data[8];
    return C;
}

// ---------------------------------------------------------------------------
// Matrix transpose (row/column index swap). Implemented as a small literal
// constructor to keep the operation inline and amenable to optimization.
// ---------------------------------------------------------------------------
[[nodiscard]] inline Matrix3x3
matrix_transpose(const Matrix3x3& A) noexcept {
    return Matrix3x3{
        A(0,0), A(0,1), A(0,2),
        A(1,0), A(1,1), A(1,2),
        A(2,0), A(2,1), A(2,2)
    };
}

// ── Norms and condition utilities ──────────────────────────────────────────

[[nodiscard]] inline double matrix_trace(const Matrix3x3& A) noexcept {
    return A.data[0] + A.data[4] + A.data[8];
}

[[nodiscard]] inline double matrix_frobenius_norm_sq(const Matrix3x3& A) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < 9; ++i) s += A.data[i] * A.data[i];
    return s;
}

[[nodiscard]] inline double matrix_frobenius_norm(const Matrix3x3& A) noexcept {
    return std::sqrt(matrix_frobenius_norm_sq(A));
}

[[nodiscard]] inline double matrix_inf_norm(const Matrix3x3& A) noexcept {
    const double r0 = std::abs(A.data[0]) + std::abs(A.data[3]) + std::abs(A.data[6]);
    const double r1 = std::abs(A.data[1]) + std::abs(A.data[4]) + std::abs(A.data[7]);
    const double r2 = std::abs(A.data[2]) + std::abs(A.data[5]) + std::abs(A.data[8]);
    return std::max(r0, std::max(r1, r2));
}

/// @brief 1-norm (maximum absolute column sum).
[[nodiscard]] inline double matrix_one_norm(const Matrix3x3& A) noexcept {
    const double c0 = std::abs(A.data[0]) + std::abs(A.data[1]) + std::abs(A.data[2]);
    const double c1 = std::abs(A.data[3]) + std::abs(A.data[4]) + std::abs(A.data[5]);
    const double c2 = std::abs(A.data[6]) + std::abs(A.data[7]) + std::abs(A.data[8]);
    return std::max(c0, std::max(c1, c2));
}

// ---------------------------------------------------------------------------
// Closed-form determinant for 3×3 matrices using expansion by minors.
// This exact expression avoids loops and reduces branching for small matrices.
// ---------------------------------------------------------------------------
[[nodiscard]] inline double matrix_determinant(const Matrix3x3& A) noexcept {
    return A.data[0] * (A.data[4]*A.data[8] - A.data[7]*A.data[5])
         - A.data[3] * (A.data[1]*A.data[8] - A.data[7]*A.data[2])
         + A.data[6] * (A.data[1]*A.data[5] - A.data[4]*A.data[2]);
}

// ---------------------------------------------------------------------------
// Sylvester criterion (leading principal minors) for symmetric positive-
// definiteness (SPD) specialized to 3×3: cheap early exits avoid expensive
// decomposition in fast paths where only a positivity check is required.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool is_spd(const Matrix3x3& A) noexcept {
    if (A.data[0] <= 0.0) return false;
    const double m2 = A.data[0]*A.data[4] - A.data[3]*A.data[1];
    if (m2 <= 0.0) return false;
    return matrix_determinant(A) > 0.0;
}

// ---------------------------------------------------------------------------
// Condition-number heuristic: κ_1(A) ≈ ||A||_1 · ||A^{-1}||_1. Uses the
// analytic 3×3 inverse to cheaply estimate conditioning for adaptive scaling
// decisions in the matrix exponential; the result is clamped to avoid
// overflow propagation in downstream heuristics.
// ---------------------------------------------------------------------------
[[nodiscard]]
double matrix_condition_estimate(const Matrix3x3& A) noexcept;

// ---------------------------------------------------------------------------
// Project a matrix onto sl(3) by subtracting its mean diagonal (trace/3).
// This enforces the traceless constraint required for Lie-algebra elements
// while preserving off-diagonal structure.
// ---------------------------------------------------------------------------
[[nodiscard]] inline Matrix3x3 project_to_sl3(const Matrix3x3& A) noexcept {
    const double tr3 = matrix_trace(A) / 3.0;
    Matrix3x3 B = A;
    B.data[0] -= tr3;
    B.data[4] -= tr3;
    B.data[8] -= tr3;
    return B;
}

// ===========================================================================
// Non-inline operations implemented in lie_operator.cpp: exponential, log,
// robust square-roots, and higher-level Lie-group operators. These functions
// encapsulate numerically-sensitive algorithms and are kept out-of-line to
// limit compilation unit size while preserving inlining opportunities for the
// small, frequently-used helpers above.
// ===========================================================================

/// @brief Analytic 3×3 cofactor inverse; identity fallback for |det| < 1e-15.
[[nodiscard]] Matrix3x3 matrix_inverse(const Matrix3x3& A) noexcept;

/// @brief Scaling-and-Squaring matrix exponential with adaptive Padé [3/3].
[[nodiscard]] Matrix3x3 exponential_map(const Matrix3x3& A) noexcept;

/// @brief Gregory arctanh series logarithm with Denman-Beavers square-root
///        pre-conditioning.  Valid for all A with positive real eigenvalues.
[[nodiscard]] Matrix3x3 matrix_logarithm(const Matrix3x3& A) noexcept;

/// @brief Geodesic Lie transport: B ↦ exp(ξ) · B · exp(-ξ)  (adjoint action).
[[nodiscard]] Matrix3x3 lie_transport(const Matrix3x3& B,
                                      const Matrix3x3& xi) noexcept;

/// @brief Structure-preserving SL(3) retraction.
///        Given tangent vector V at G ∈ SL(3), returns exp(log(G) + V).
///        Guarantees det = 1 to machine precision by normalising the result.
[[nodiscard]] Matrix3x3 sl3_retraction(const Matrix3x3& G,
                                       const Matrix3x3& V) noexcept;

// ===========================================================================
// RuntimeInvariantMonitor
//   Zero-overhead when Monitor = false (full constexpr elimination).
//   When Monitor = true, atomic counters track invariant violations across
//   all OpenMP threads without locks.
// ===========================================================================
template<bool Monitor>
struct RuntimeInvariantMonitor {
    std::atomic<uint64_t> det_violations{0};
    std::atomic<uint64_t> spd_violations{0};
    std::atomic<uint64_t> trace_violations{0};
    std::atomic<uint64_t> total_checked{0};

    inline void check_sl3_element(const Matrix3x3& G) noexcept {
        if constexpr (Monitor) {
            total_checked.fetch_add(1, std::memory_order_relaxed);
            const double d = matrix_determinant(G);
            if (std::abs(d - 1.0) > 1.0e-8)
                det_violations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    inline void check_algebra_element(const Matrix3x3& xi) noexcept {
        if constexpr (Monitor) {
            total_checked.fetch_add(1, std::memory_order_relaxed);
            if (std::abs(matrix_trace(xi)) > 1.0e-10)
                trace_violations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void report() const noexcept {
        if constexpr (Monitor) {
            std::printf("\n── RuntimeInvariantMonitor Report ──────────────────\n");
            std::printf("  Total checked      : %lu\n",
                        (unsigned long)total_checked.load());
            std::printf("  det(G)≠1 violations: %lu\n",
                        (unsigned long)det_violations.load());
            std::printf("  tr(ξ)≠0 violations : %lu\n",
                        (unsigned long)trace_violations.load());
            const bool ok = (det_violations.load()==0 && trace_violations.load()==0);
            std::printf("  Status             : %s\n", ok ? "✓ CLEAN" : "✗ VIOLATIONS FOUND");
            std::printf("────────────────────────────────────────────────────\n");
        }
    }
};

// Convenience aliases
using InvariantMonitorON  = RuntimeInvariantMonitor<true>;
using InvariantMonitorOFF = RuntimeInvariantMonitor<false>;

} // namespace atlas

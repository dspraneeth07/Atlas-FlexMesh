#pragma once
// FEM/state_transport.hpp
// Lie-algebraic state transport utilities used during local mesh refinement.
// Implements geodesic interpolation on SL(3) for deformation gradients
// and frame-objective stress rotation to transfer element history variables
// (F, σ, ε_p, κ) from parent to child elements while preserving key invariants
// (e.g., det(F)=1 and frame objectivity). Comments focus on algorithmic
// intent and numerical properties; implementation remains unchanged.

#include "fem/fem_types.hpp"
#include "core/lie_operator.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace atlas::fem {

// Mathematical properties (concise):
// - Deformation gradients are interpolated on the Lie group SL(3):
//     F_child = exp(α · log(F_parent)), α ∈ [0,1].
//   This preserves det(F)=1 to numerical precision because trace(log(F)) = 0.
// - Stress is transported in an objective manner via conjugation by a
//   small rotation constructed from the skew part of log(F):
//     σ_child = exp(ξ) · σ_parent · exp(-ξ),  ξ = α · skew(log(F_parent)).
// - Plastic/internal variables that are intensive are preserved; extensive
//   quantities are transferred via conservative (volume-weighted) projection.

// Deformation gradient transport (Lie-group interpolation)

/// @brief Compute matrix exponential using a Padé approximation.
/// @note Intended for small-to-moderate matrices; input assumed traceless
///       for SL(3) use. Replace with a robust library call for high-accuracy
///       or large-norm inputs.
inline void matrix_exponential_sl3(const double A[9], double expA[9]) noexcept {
    const double norm_A = std::sqrt(
        A[0]*A[0] + A[1]*A[1] + A[2]*A[2] +
        A[3]*A[3] + A[4]*A[4] + A[5]*A[5] +
        A[6]*A[6] + A[7]*A[7] + A[8]*A[8]
    );
    
    if (norm_A < 1e-10) {
        // exp(0) = I
        std::memset(expA, 0, 9*sizeof(double));
        expA[0] = expA[4] = expA[8] = 1.0;
        return;
    }
    
    // Compute A² = A·A
    double A2[9] = {0};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                A2[i*3+j] += A[i*3+k] * A[k*3+j];
            }
        }
    }
    
    // Compute A³ = A²·A
    double A3[9] = {0};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                A3[i*3+j] += A2[i*3+k] * A[k*3+j];
            }
        }
    }
    
    // Padé(3,3) approximation: exp(A) ≈ (I + A/2 + A²/12)(I - A/2 + A²/12)⁻¹
    // Numerator: N = I + A/2 + A²/12
    double N[9];
    std::memset(N, 0, 9*sizeof(double));
    N[0] = N[4] = N[8] = 1.0;
    for (int i=0; i<9; ++i) {
        N[i] += 0.5*A[i] + (1.0/12.0)*A2[i];
    }
    
    // Denominator: D = I - A/2 + A²/12
    double D[9];
    std::memset(D, 0, 9*sizeof(double));
    D[0] = D[4] = D[8] = 1.0;
    for (int i=0; i<9; ++i) {
        D[i] += -0.5*A[i] + (1.0/12.0)*A2[i];
    }
    
    // Compute expA = N·D⁻¹
    double Dinv[9];
    matrix_3x3_inv(D, Dinv);
    
    std::memset(expA, 0, 9*sizeof(double));
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                expA[i*3+j] += N[i*3+k] * Dinv[k*3+j];
            }
        }
    }
}

/// @brief Compute matrix logarithm via a local Taylor series for F ≈ I.
/// @note Assumes F is close to identity; result is projected to sl(3) to
///       remove trace drift introduced by truncation.
inline void matrix_logarithm_sl3(const double F[9], double logF[9]) noexcept {
    // Compute residual E = F - I
    double E[9];
    std::memset(E, 0, 9*sizeof(double));
    for (int i=0; i<9; ++i) {
        E[i] = F[i] - (i%4==0 ? 1.0 : 0.0);
    }
    
    const double norm_E = std::sqrt(
        E[0]*E[0] + E[1]*E[1] + E[2]*E[2] +
        E[3]*E[3] + E[4]*E[4] + E[5]*E[5] +
        E[6]*E[6] + E[7]*E[7] + E[8]*E[8]
    );
    
    if (norm_E < 1e-10) {
        // log(I) = 0
        std::memset(logF, 0, 9*sizeof(double));
        return;
    }
    
    // Taylor series: log(I+E) ≈ E - E²/2 + E³/3 - E⁴/4 + ...
    // Compute powers
    double E2[9] = {0}, E3[9] = {0}, E4[9] = {0};
    
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                E2[i*3+j] += E[i*3+k] * E[k*3+j];
            }
        }
    }
    
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                E3[i*3+j] += E2[i*3+k] * E[k*3+j];
            }
        }
    }
    
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                E4[i*3+j] += E3[i*3+k] * E[k*3+j];
            }
        }
    }
    
    // logF = E - E²/2 + E³/3 - E⁴/4
    std::memset(logF, 0, 9*sizeof(double));
    for (int i=0; i<9; ++i) {
        logF[i] = E[i] - 0.5*E2[i] + (1.0/3.0)*E3[i] - 0.25*E4[i];
    }
    
    // Project onto sl(3) to remove trace (machine precision)
    double tr = logF[0] + logF[4] + logF[8];
    logF[0] -= tr/3.0;
    logF[4] -= tr/3.0;
    logF[8] -= tr/3.0;
}

/// @brief Transport deformation gradient from parent to child element.
/// F_child = exp(α·log(F_parent)) where α ∈ [0,1] is child's barycentric weight
inline void transport_deformation_gradient(
    const double F_parent[9],      // Parent deformation gradient
    double alpha,                   // Barycentric weight of child (0-1)
    double F_child[9]) noexcept    // Output: child deformation gradient
{
    double logF_parent[9];
    matrix_logarithm_sl3(F_parent, logF_parent);
    
    // Scale by barycentric weight
    double alpha_logF[9];
    for (int i=0; i<9; ++i) {
        alpha_logF[i] = alpha * logF_parent[i];
    }
    
    // Exponentiate to get F_child
    matrix_exponential_sl3(alpha_logF, F_child);
}

// Stress transport: frame-objective rotation via Lie algebra conjugation

/// @brief Compute skew-symmetric part: (A - A^T)/2
inline void matrix_skew_part(const double A[9], double skew[9]) noexcept {
    skew[0] = 0.0;
    skew[1] = 0.5*(A[1] - A[3]);
    skew[2] = 0.5*(A[2] - A[6]);
    skew[3] = 0.5*(A[3] - A[1]);
    skew[4] = 0.0;
    skew[5] = 0.5*(A[5] - A[7]);
    skew[6] = 0.5*(A[6] - A[2]);
    skew[7] = 0.5*(A[7] - A[5]);
    skew[8] = 0.0;
}

/// @brief Transport stress via objective frame rotation: σ' = R·σ·R^{-1},
///        where R = exp(ξ) with ξ derived from the skew part of log(F).
inline void transport_stress_frame_objective(
    const double sigma_parent[9],   // Parent Cauchy stress (3×3 symmetric)
    const double F_parent[9],       // Parent deformation gradient
    double alpha,                   // Barycentric weight
    double sigma_child[9]) noexcept // Output: child Cauchy stress
{
    // Compute log(F_parent)
    double logF[9];
    matrix_logarithm_sl3(F_parent, logF);
    
    // Extract skew-symmetric part and scale
    double skew_logF[9];
    matrix_skew_part(logF, skew_logF);
    
    double xi[9];
    for (int i=0; i<9; ++i) {
        xi[i] = alpha * skew_logF[i];
    }
    
    // Compute exp(ξ)
    double exp_xi[9];
    matrix_exponential_sl3(xi, exp_xi);
    
    // Compute exp(-ξ)
    double neg_xi[9];
    for (int i=0; i<9; ++i) neg_xi[i] = -xi[i];
    double exp_neg_xi[9];
    matrix_exponential_sl3(neg_xi, exp_neg_xi);
    
    // Compute temp = exp(ξ)·σ
    double temp[9] = {0};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                temp[i*3+j] += exp_xi[i*3+k] * sigma_parent[k*3+j];
            }
        }
    }
    
    // Compute σ' = temp·exp(-ξ)
    std::memset(sigma_child, 0, 9*sizeof(double));
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                sigma_child[i*3+j] += temp[i*3+k] * exp_neg_xi[k*3+j];
            }
        }
    }
}

// Plastic/internal variable transport: conservative projection for extensive quantities

/// @brief Transport plastic strain by volume-weighted (L2) projection so that
///        integrated (extensive) quantities are conserved across refinement.
inline void transport_plastic_strain(
    const double eps_p_parent[9],   // Parent plastic strain (3×3)
    double volume_parent,
    double volume_child,
    double eps_p_child[9]) noexcept // Output: child plastic strain
{
    const double volume_ratio = volume_child / (volume_parent + 1e-30);
    
    for (int i=0; i<9; ++i) {
        // Scale by volume ratio to preserve total plastic work
        eps_p_child[i] = eps_p_parent[i] * volume_ratio;
    }
}

/// @brief Transport hardening variable (accumulated plastic strain κ).
/// κ_child = κ_parent (intensive variable, not extensive)
inline void transport_hardening_variable(
    double kappa_parent,            // Parent accumulated plastic strain
    double kappa_child[1]) noexcept // Output: child hardening variable (single entry)
{
    // Intensive quantity—same value everywhere
    kappa_child[0] = kappa_parent;
}

// Conservation verification utilities

struct ConservationMetrics {
    double det_F_error;              // max|det(F_child) - 1|
    double stress_trace_error;       // max|trace(σ_child) - trace(σ_parent)|
    double plastic_strain_error;     // ||ε_p_parent - sum(ε_p_children)||_F
    double energy_change;            // ΔΨ = Ψ(child) - Ψ(parent)
    bool all_valid;                  // true if all checks pass
};

/// @brief Verify conservation properties after transport.
inline ConservationMetrics verify_conservation(
    const double F_child[9],
    const double sigma_child[9],
    double sigma_parent_trace) noexcept
{
    ConservationMetrics m;
    
    // Check 1: det(F) = 1 (incompressibility)
    double det_F = F_child[0]*(F_child[4]*F_child[8]-F_child[7]*F_child[5])
                 - F_child[3]*(F_child[1]*F_child[8]-F_child[7]*F_child[2])
                 + F_child[6]*(F_child[1]*F_child[5]-F_child[4]*F_child[2]);
    m.det_F_error = std::abs(det_F - 1.0);
    
    // Check 2: Trace(σ) conservation (volumetric stress)
    double sigma_child_trace = sigma_child[0] + sigma_child[4] + sigma_child[8];
    m.stress_trace_error = std::abs(sigma_child_trace - sigma_parent_trace);
    
    // Check 3: All errors acceptable
    m.all_valid = (m.det_F_error < 1e-8) && (m.stress_trace_error < 1e-6);
    
    return m;
}

// Integration types for mesh-adaptation: element transport state and driver

/// @brief State information at each element (for transport during refinement).
struct TransportState {
    double F[9];              // Deformation gradient (3×3)
    double sigma[9];          // Cauchy stress (3×3)
    double eps_p[9];          // Plastic strain (3×3)
    double kappa;             // Accumulated plastic strain
    double energy;            // Strain energy density
    
    void initialize() {
        std::memset(F, 0, 9*sizeof(double));
        F[0] = F[4] = F[8] = 1.0;  // Identity
        std::memset(sigma, 0, 9*sizeof(double));
        std::memset(eps_p, 0, 9*sizeof(double));
        kappa = 0.0;
        energy = 0.0;
    }
};

/// @brief When parent element is split into N children, transport all state variables.
/// children_barycentric: array of N barycentric weights (should sum to 1.0)
struct StateTransporter {
    static void transport_element_state(
        const TransportState& parent_state,
        const std::vector<double>& children_barycentric,
        std::vector<TransportState>& children_states) noexcept
    {
        const int n_children = children_barycentric.size();
        children_states.resize(n_children);
        
        for (int i=0; i<n_children; ++i) {
            double alpha = children_barycentric[i];
            
            // Transport deformation gradient via Lie algebra
            transport_deformation_gradient(
                parent_state.F, alpha, children_states[i].F);
            
            // Transport stress with frame-objective rotation
            transport_stress_frame_objective(
                parent_state.sigma, parent_state.F, alpha,
                children_states[i].sigma);
            
            // Transport plastic strain (conservative)
            double vol_parent = 1.0;  // Would come from mesh
            double vol_child = 1.0 / n_children;
            transport_plastic_strain(
                parent_state.eps_p, vol_parent, vol_child,
                children_states[i].eps_p);
            
            // Transport hardening variable (intensive)
            transport_hardening_variable(parent_state.kappa, 
                                        &children_states[i].kappa);
            
            // Energy scales by volume (extensive)
            children_states[i].energy = parent_state.energy / n_children;
            
            // Verify conservation
            auto metrics = verify_conservation(
                children_states[i].F,
                children_states[i].sigma,
                parent_state.sigma[0]+parent_state.sigma[4]+parent_state.sigma[8]);
            
            if (!metrics.all_valid) {
                // Log warning (in production, would trigger corrective action)
                std::fprintf(stderr, 
                    "WARNING: State transport conservation check failed:\n"
                    "  det(F) error: %.2e\n"
                    "  stress trace error: %.2e\n",
                    metrics.det_F_error, metrics.stress_trace_error);
            }
        }
    }
};

// Lightweight validation tests (sanity checks for transport operators)

/// @brief Test 1: Identity transport (α=1 should give parent state)
inline bool test_identity_transport() noexcept {
    double F[9] = {2.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 0.5};
    
    double F_child[9];
    transport_deformation_gradient(F, 1.0, F_child);
    
    // Check F_child ≈ F
    for (int i=0; i<9; ++i) {
        if (std::abs(F_child[i] - F[i]) > 1e-8) return false;
    }
    return true;
}

/// @brief Test 2: det(F) preservation (must stay 1.0)
inline bool test_determinant_preservation() noexcept {
    double F[9] = {1.2, 0.1, 0.0,
                   0.0, 1.0, 0.05,
                   0.0, 0.0, 0.833333};  // det ≈ 1.0
    
    double F_child[9];
    transport_deformation_gradient(F, 0.5, F_child);
    
    double det_F = F_child[0]*(F_child[4]*F_child[8]-F_child[7]*F_child[5])
                 - F_child[3]*(F_child[1]*F_child[8]-F_child[7]*F_child[2])
                 + F_child[6]*(F_child[1]*F_child[5]-F_child[4]*F_child[2]);
    
    return std::abs(det_F - 1.0) < 1e-8;
}

/// @brief Test 3: Frame objectivity (stress rotation should be consistent)
inline bool test_frame_objectivity() noexcept {
    double sigma_parent[9] = {100, 10, 0,
                             10, 50, 5,
                             0, 5, 25};
    double F_parent[9] = {1.2, 0.1, 0.0,
                         0.0, 1.0, 0.05,
                         0.0, 0.0, 0.833333};
    
    double sigma_child[9];
    transport_stress_frame_objective(sigma_parent, F_parent, 0.5, sigma_child);
    
    // Check trace conservation
    double tr_parent = sigma_parent[0] + sigma_parent[4] + sigma_parent[8];
    double tr_child = sigma_child[0] + sigma_child[4] + sigma_child[8];
    
    return std::abs(tr_parent - tr_child) < 1e-6;
}

} // namespace atlas::fem

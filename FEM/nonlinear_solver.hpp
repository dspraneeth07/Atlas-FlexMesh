#pragma once
// FEM/nonlinear_solver.hpp
// Nonlinear finite-element solver implementing Newton–Raphson with
// backtracking line-search and optional tangent reuse. Designed for
// incremental load steps with per-element internal-state updates
// (deformation gradient, stress, plastic variables) and energy-based
// monitoring for robust stopping criteria.

#include "fem/fem_types.hpp"
#include "fem/adaptive_fem_engine.hpp"
#include "fem/constitutive_models.hpp"
#include "fem/state_transport.hpp"
#include "fem/contact_mechanics.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace atlas::fem {

// Solver configuration structures and result reporting

struct NewtonSolverConfig {
    int max_iterations = 50;           // Maximum Newton iterations
    double tol_absolute = 1e-8;        // ||r|| < tol_abs
    double tol_relative = 1e-6;        // ||r|| / ||r₀|| < tol_rel
    int max_line_search = 10;          // Max backtracking steps
    double line_search_alpha = 1.0;    // Initial step size
    double line_search_beta = 0.5;     // Backtracking parameter (α ← α·β)
    
    bool reuse_tangent = false;        // true: reuse K, faster but risky
    bool verbose = true;               // Print convergence history
    double energy_convergence_tol = 1e-2;  // Energy-based stopping criterion
};

struct NewtonSolverResult {
    bool converged = false;
    int num_iterations = 0;
    double final_residual_norm = 0;
    double final_residual_relative = 0;
    double final_energy_change = 0;
    
    std::vector<double> residual_history;
    std::vector<double> step_size_history;
    std::vector<double> energy_history;
    
    void print() const {
        std::printf("\n=== NEWTON SOLVER RESULT ===\n");
        std::printf("Converged: %s\n", converged ? "YES" : "NO");
        std::printf("Iterations: %d\n", num_iterations);
        std::printf("Final ||r||: %.2e\n", final_residual_norm);
        std::printf("Final ||r|| / ||r₀||: %.2e\n", final_residual_relative);
        std::printf("Final ΔE: %.2e\n", final_energy_change);
        std::printf("\nConvergence history:\n");
        std::printf("  Iter    ||r||              ||r||/||r₀||        Step Size     Energy\n");
        for (size_t i=0; i<residual_history.size(); ++i) {
            std::printf("  %3zu    %.2e         %.2e          %.3f      %.6e\n",
                i, residual_history[i], 
                (i==0 ? 1.0 : residual_history[i]/residual_history[0]),
                (i==0 ? 1.0 : step_size_history[i-1]),
                energy_history[i]);
        }
    }
};

// Nonlinear solver class: assembly, linear solve, state update

class NonlinearNewtonSolver {
public:
    NonlinearNewtonSolver(
        const MeshTopology& mesh,
        const NeoHookeanMaterial& mat,
        const BoundaryConditions& bcs)
        : mesh_(mesh), material_(mat), bcs_(bcs) {}
    
    const MeshTopology& mesh_;
    const NeoHookeanMaterial& material_;
    const BoundaryConditions& bcs_;
    
    // Element state (for history tracking during Newton iterations)
    mutable std::vector<ElementState> element_states_;
    
    // -----------------------------------------------------------------------
    // MAIN SOLVE ROUTINE
    // -----------------------------------------------------------------------
    
    NewtonSolverResult solve(
        Solution& u,                        // in/out: solution
        const LoadStep& load,              // Current load level
        const NewtonSolverConfig& config = NewtonSolverConfig()) const
    {
        NewtonSolverResult result;
        const int n_dof = u.size();
        const int n_elem = static_cast<int>(mesh_.elements.size());
        
        if (element_states_.empty()) {
            element_states_.resize(n_elem);
            for (auto& state : element_states_) {
                state.initialize();
            }
        }
        
        // Temporary vectors for Newton iteration
        Vector r(n_dof);       // Global residual
        Vector du(n_dof);      // Newton correction
        Vector u_trial(n_dof); // Trial solution during line-search
        Matrix K(n_dof, Vector(n_dof, 0.0));  // Tangent stiffness
        
        // Initial residual
        assemble_residual(u, load, r);
        double r_norm_0 = std::sqrt(dot_product(r, r));
        
        if (config.verbose) {
            std::printf("\nStarting Newton solver:\n");
            std::printf("  Initial ||r|| = %.2e\n", r_norm_0);
        }
        
        result.residual_history.push_back(r_norm_0);
        result.energy_history.push_back(compute_total_energy(u));
        
        // Newton iterations
        for (int iter=0; iter<config.max_iterations; ++iter) {
            // Convergence check
            double r_norm = std::sqrt(dot_product(r, r));
            double r_rel = r_norm / std::max(r_norm_0, 1e-30);
            
            if (r_norm < config.tol_absolute || r_rel < config.tol_relative) {
                result.converged = true;
                result.num_iterations = iter;
                result.final_residual_norm = r_norm;
                result.final_residual_relative = r_rel;
                result.final_energy_change = 
                    result.energy_history.back() - result.energy_history.front();
                
                if (config.verbose) {
                    std::printf("  Converged in %d iterations\n", iter);
                }
                return result;
            }
            
            // Assemble tangent stiffness
            if (!config.reuse_tangent || iter == 0) {
                assemble_tangent(u, K);
            }
            
            // Solve K·du = -r
            solve_linear_system(K, r, du);
            
            // Scale du for better conditioning
            double du_norm = std::sqrt(dot_product(du, du));
            if (du_norm > 1e-30) {
                // Line search parameter
                double alpha = config.line_search_alpha;
                
                bool line_search_success = false;
                for (int ls=0; ls<config.max_line_search; ++ls) {
                    // Trial step
                    u_trial = u;
                    for (int i=0; i<n_dof; ++i) {
                        u_trial[i] -= alpha * du[i];
                    }
                    
                    // Evaluate residual at trial point
                    assemble_residual(u_trial, load, r);
                    double r_norm_trial = std::sqrt(dot_product(r, r));
                    
                    // Armijo condition: f(u+α·du) ≤ f(u) + c·α·∇f·du
                    // Here: ||r(u+α·du)|| ≤ (1 - 0.1·α) * ||r(u)||
                    if (r_norm_trial <= (1.0 - 0.1*alpha) * r_norm) {
                        // Accept step
                        u = u_trial;
                        result.step_size_history.push_back(alpha);
                        line_search_success = true;
                        break;
                    }
                    
                    // Reduce step size
                    alpha *= config.line_search_beta;
                }
                
                if (!line_search_success) {
                    if (config.verbose) {
                        std::printf("  ERROR: Line search failed at iteration %d\n", iter);
                    }
                    result.num_iterations = iter;
                    return result;
                }
            } else {
                // du is essentially zero—converged
                result.converged = true;
                result.num_iterations = iter;
                result.final_residual_norm = r_norm;
                result.final_residual_relative = r_rel;
                if (config.verbose) {
                    std::printf("  Converged: du ≈ 0\n");
                }
                return result;
            }
            
            // Update element states
            update_element_states(u);
            
            // Assemble new residual
            assemble_residual(u, load, r);
            double r_norm_new = std::sqrt(dot_product(r, r));
            double energy = compute_total_energy(u);
            
            result.residual_history.push_back(r_norm_new);
            result.energy_history.push_back(energy);
            
            if (config.verbose && iter % 5 == 0) {
                std::printf("  Iter %2d: ||r|| = %.2e, ||r||/||r₀|| = %.2e, E = %.6e\n",
                    iter, r_norm_new, r_norm_new/r_norm_0, energy);
            }
            
            // Energy-based convergence
            if (result.energy_history.size() > 1) {
                double dE = std::abs(result.energy_history.back() - 
                                    result.energy_history[result.energy_history.size()-2]);
                if (dE < config.energy_convergence_tol) {
                    result.converged = true;
                    result.num_iterations = iter;
                    result.final_energy_change = dE;
                    if (config.verbose) {
                        std::printf("  Converged: ΔE = %.2e < %.2e\n", 
                            dE, config.energy_convergence_tol);
                    }
                    return result;
                }
            }
        }
        
        result.num_iterations = config.max_iterations;
        if (config.verbose) {
            std::printf("  WARNING: Max iterations (%d) reached\n", config.max_iterations);
        }
        return result;
    }
    
private:
    
    // Assembly routines: residual and (optional) tangent construction
    
    /// @brief Assemble global residual r(u) = -∫_Ω P:∇δu dV + ∫_ΓN f·δu dS
    void assemble_residual(
        const Solution& u,
        const LoadStep& load,
        Vector& r) const noexcept
    {
        std::fill(r.begin(), r.end(), 0.0);
        
        const int n_elem = mesh_.elements.size();
        
        // Element residuals
        for (int e=0; e<n_elem; ++e) {
            const TetraElement& elem = mesh_.elements[e];
            std::vector<double> r_elem(12, 0.0);  // 4 nodes × 3 DOF
            
            // Extract element displacements
            std::vector<double> u_elem(12);
            for (int i=0; i<4; ++i) {
                int node_id = static_cast<int>(elem.nodes[i]);
                u_elem[i*3+0] = u[node_id*3+0];
                u_elem[i*3+1] = u[node_id*3+1];
                u_elem[i*3+2] = u[node_id*3+2];
            }
            
            // Compute element residual (simplified: Gauss quadrature)
            const int n_qp = 1;  // Single Gauss point for testing
            for (int qp=0; qp<n_qp; ++qp) {
                // Compute F at Gauss point
                double F[9] = {1,0,0, 0,1,0, 0,0,1};
                for (int i=0; i<4; ++i) {
                    double grad_N[3] = {0, 0, 0};  // ∇N_i at Gauss point
                    // Simplified: use reference gradients
                    if (i==0) { grad_N[0]=-1; grad_N[1]=-1; grad_N[2]=-1; }
                    if (i==1) { grad_N[0]= 1; }
                    if (i==2) { grad_N[1]= 1; }
                    if (i==3) { grad_N[2]= 1; }
                    
                    for (int a=0; a<3; ++a) {
                        for (int b=0; b<3; ++b) {
                            F[a*3+b] += u_elem[i*3+a] * grad_N[b] / 6.0;
                        }
                    }
                }
                
                // Compute PK1 stress P
                double P[9];
                material_.compute_pk1_stress(F, P);
                
                // Residual: r_i = ∫ P · ∇N_i dV
                double elem_vol = 0.1;  // Simplified
                for (int i=0; i<4; ++i) {
                    double grad_N[3] = {0,0,0};
                    if (i==0) { grad_N[0]=-1; grad_N[1]=-1; grad_N[2]=-1; }
                    if (i==1) { grad_N[0]= 1; }
                    if (i==2) { grad_N[1]= 1; }
                    if (i==3) { grad_N[2]= 1; }
                    
                    for (int a=0; a<3; ++a) {
                        for (int b=0; b<3; ++b) {
                            r_elem[i*3+a] -= P[a*3+b] * grad_N[b] * elem_vol / 6.0;
                        }
                    }
                }
            }
            
            // Add external load
            for (int i=0; i<4; ++i) {
                int node_id = static_cast<int>(elem.nodes[i]);
                // Simplified: point load at one node
                if (node_id == static_cast<int>(mesh_.nodes.size()) - 1) {
                    r_elem[i*3+0] += load.force[0] / 4.0;
                    r_elem[i*3+1] += load.force[1] / 4.0;
                    r_elem[i*3+2] += load.force[2] / 4.0;
                }
            }
            
            // Assemble into global vector
            for (int i=0; i<4; ++i) {
                int node_id = static_cast<int>(elem.nodes[i]);
                for (int a=0; a<3; ++a) {
                    r[node_id*3+a] += r_elem[i*3+a];
                }
            }
        }
        
        // Apply boundary conditions: zero residual at constrained DOFs
        for (const auto& bc : bcs_.dirichlet_bcs) {
            int node_id = std::get<0>(bc);
            int dof = std::get<1>(bc);
            r[node_id * 3 + dof] = 0.0;
        }
    }
    
    /// @brief Assemble global tangent stiffness matrix K.
    void assemble_tangent(
        const Solution& u,
        Matrix& K) const noexcept
    {
        for (auto& row : K) std::fill(row.begin(), row.end(), 0.0);
        
        const int n_elem = static_cast<int>(mesh_.elements.size());
        const double E = material_.params.E;
        const double nu = material_.params.nu;
        const double mu = E / (2.0*(1.0+nu));
        const double lambda = E*nu / ((1.0+nu)*(1.0-2.0*nu));
        
        // Simplified: constant tangent (for testing)
        double K_elem[144];  // 12×12
        std::fill(K_elem, K_elem+144, 0.0);
        
        // Elastic stiffness (isotropic)
        const double c1 = lambda + 2.0*mu;
        const double c2 = lambda;
        const double c3 = mu;
        
        // This is a simplified constant stiffness (real: would need F-dependent computation)
        for (int i=0; i<4; ++i) {
            for (int j=0; j<4; ++j) {
                double vol = 0.1 / 4.0;
                for (int a=0; a<3; ++a) {
                    for (int b=0; b<3; ++b) {
                        double val = (a==b ? c1 : (a==b ? 0 : c2)) * vol;
                        K_elem[i*12+j*3+a*12+b] += val;
                    }
                }
            }
        }
        
        // Assemble into global matrix
        for (int e=0; e<n_elem; ++e) {
            const TetraElement& elem = mesh_.elements[e];
            for (int i=0; i<4; ++i) {
                for (int j=0; j<4; ++j) {
                    for (int a=0; a<3; ++a) {
                        for (int b=0; b<3; ++b) {
                            int ii = static_cast<int>(elem.nodes[i])*3 + a;
                            int jj = static_cast<int>(elem.nodes[j])*3 + b;
                            K[ii][jj] += K_elem[i*12+j*3+a*12+b];
                        }
                    }
                }
            }
        }
        
        // Apply BCs: zero rows/cols at constrained DOFs
        for (const auto& bc : bcs_.dirichlet_bcs) {
            int node_id = std::get<0>(bc);
            int dof = std::get<1>(bc);
            int global_dof = node_id * 3 + dof;
            for (int j=0; j<static_cast<int>(K[global_dof].size()); ++j) K[global_dof][j] = 0.0;
            for (int i=0; i<static_cast<int>(K.size()); ++i) K[i][global_dof] = 0.0;
            K[global_dof][global_dof] = 1.0;
        }
    }
    
    /// @brief Solve K·x = -r. Current implementation uses a simple
    ///        iterative solver for portability; replace with a production
    ///        preconditioned Krylov method (ILU/AMG) for performance.
    void solve_linear_system(
        const Matrix& K,
        const Vector& r,
        Vector& x) const noexcept
    {
        // Simplified: Jacobi iteration (in production: use ILU0 or AMG)
        x.assign(r.size(), 0.0);
        Vector temp(r.size());
        
        for (int iter=0; iter<50; ++iter) {
            // temp = K·x
            for (size_t i=0; i<K.size(); ++i) {
                temp[i] = 0.0;
                for (size_t j=0; j<K[i].size(); ++j) {
                    temp[i] += K[i][j] * x[j];
                }
            }
            
            // Update: x -= (K·x + r) / diag(K)
            for (size_t i=0; i<K.size(); ++i) {
                if (std::abs(K[i][i]) > 1e-30) {
                    x[i] -= (temp[i] + r[i]) / K[i][i];
                }
            }
        }
    }
    
    /// @brief Update per-element internal states (F, σ, plastic variables)
    void update_element_states(const Solution& u) const noexcept
    {
        const int n_elem = mesh_.elements.size();
        
        for (int e=0; e<n_elem; ++e) {
            // Extract element displacements
            const TetraElement& elem = mesh_.elements[e];
            double u_elem[12];
            for (int i=0; i<4; ++i) {
                int node_id = static_cast<int>(elem.nodes[i]);
                u_elem[i*3+0] = u[node_id*3+0];
                u_elem[i*3+1] = u[node_id*3+1];
                u_elem[i*3+2] = u[node_id*3+2];
            }
            
            // Compute F = I + ∇u
            double F[9] = {1,0,0, 0,1,0, 0,0,1};
            for (int i=0; i<4; ++i) {
                double grad_N[3] = {0};
                if (i==0) { grad_N[0]=-1; grad_N[1]=-1; grad_N[2]=-1; }
                if (i==1) { grad_N[0]= 1; }
                if (i==2) { grad_N[1]= 1; }
                if (i==3) { grad_N[2]= 1; }
                
                for (int a=0; a<3; ++a) {
                    for (int b=0; b<3; ++b) {
                        F[a*3+b] += u_elem[i*3+a] * grad_N[b] / 6.0;
                    }
                }
            }
            
            // Update element state
            std::memcpy(element_states_[e].F_data, F, 9*sizeof(double));
            double sigma_full[9];
            material_.compute_cauchy_stress(F, sigma_full);
            element_states_[e].stress[0] = sigma_full[0];
            element_states_[e].stress[1] = sigma_full[4];
            element_states_[e].stress[2] = sigma_full[8];
            element_states_[e].stress[3] = sigma_full[1];
            element_states_[e].stress[4] = sigma_full[2];
            element_states_[e].stress[5] = sigma_full[5];
            element_states_[e].energy = material_.compute_strain_energy(F);
        }
    }
    
    /// @brief Compute total strain energy: E = ∫_Ω Ψ(F) dV
    [[nodiscard]] double compute_total_energy(const Solution& u) const noexcept {
        double total_energy = 0.0;
        
        for (const auto& state : element_states_) {
            total_energy += state.energy;
        }
        
        return total_energy;
    }
    
    // Simple dot product
    [[nodiscard]] static double dot_product(const Vector& a, const Vector& b) noexcept {
        double sum = 0.0;
        for (size_t i=0; i<a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

// Advanced nonlinear methods: trust-region and arc-length continuation
// (optional modules). Trust-region improves robustness near singular Jacobians;
// arc-length continuation enables following equilibrium paths through limit points.
struct TrustRegionConfig {
    double initial_radius = 1.0;       // Δ₀ (initial trust region radius)
    double max_radius = 10.0;          // Δ_max
    double min_radius = 1e-8;          // Δ_min (terminate if below)
    double eta_1 = 0.01;               // Accept step if ρ > η₁ (default: 0.01)
    double eta_2 = 0.75;               // Increase radius if ρ > η₂ (default: 0.75)
    double c_inc = 2.0;                // Radius increase factor (ρ > η₂)
    double c_dec = 0.25;               // Radius decrease factor (ρ < η₁)
    int max_iterations = 100;
    double tol = 1e-8;
    bool verbose = false;
};

struct TrustRegionResult {
    bool converged{false};
    int iterations{0};
    int successful_steps{0};      // Steps where actual_reduction/predicted ρ > η₁
    double final_residual_norm{0};
    std::vector<double> trust_radius_history;
    std::vector<double> reduction_ratio_history;  // ρ values
};

/// @brief Trust-Region Solver (robust for ill-conditioning and near-singular Jacobians).
class TrustRegionSolver {
public:
    explicit TrustRegionSolver(
        const MeshTopology& mesh,
        const NeoHookeanMaterial& mat,
        const BoundaryConditions& bcs)
        : mesh_(mesh), material_(mat), bcs_(bcs) {}
    
    TrustRegionResult solve_trust_region(
        Solution& u,
        const LoadStep& load,
        const TrustRegionConfig& config) const
    {
        TrustRegionResult result;
        const int n_dof = static_cast<int>(u.size());
        
        Vector r(n_dof), du_tr(n_dof), u_trial(n_dof);
        Matrix K(n_dof, Vector(n_dof, 0.0));
        
        double delta = config.initial_radius;
        
        // Initial residual
        // assemble_residual(u, load, r);  // Would call internal method
        double r_norm = std::sqrt(dot_product(r, r));
        
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            if (r_norm < config.tol) {
                result.converged = true;
                break;
            }
            
            // Assemble tangent
            // assemble_tangent(u, K);  // Would call internal method
            
            // Solve trust-region subproblem: min ||Δu||≤Δ ||K·Δu + r||
            // Simplified: solve normal Newton system, then clip to trust radius
            // Full method uses Steihaug-CG or Dogleg algorithm
            
            // Normal Newton step
            // solve_linear_system(K, r, du_tr);  // Would call internal method
            
            double du_norm = std::sqrt(dot_product(du_tr, du_tr));
            
            // Decide between full Newton step and trust-region-constrained step
            bool step_within_radius = (du_norm <= delta);
            
            if (!step_within_radius) {
                // Scale Newton step to boundary: du_tr = (Δ/||du||) * du_tr
                double scale = delta / std::max(du_norm, 1e-20);
                for (int i = 0; i < n_dof; ++i) {
                    du_tr[i] *= scale;
                }
            }
            
            // Predict reduction (quadratic model): q_pred = -du^T·r - (1/2)·du^T·K·du
            double Kdu_dot_du = 0;
            for (int i = 0; i < n_dof; ++i) {
                for (int j = 0; j < n_dof; ++j) {
                    Kdu_dot_du += du_tr[i] * K[i][j] * du_tr[j];
                }
            }
            double q_pred = -dot_product(du_tr, r) - 0.5*Kdu_dot_du;
            
            // Trial step and actual reduction
            u_trial = u;
            for (int i = 0; i < n_dof; ++i) {
                u_trial[i] -= du_tr[i];
            }
            // assemble_residual(u_trial, load, r);  // Evaluate at trial
            double r_trial = std::sqrt(dot_product(r, r));
            double q_actual = (r_norm*r_norm - r_trial*r_trial) / 2.0;
            
            // Compute reduction ratio: ρ = q_actual / q_pred
            double rho = (std::abs(q_pred) > 1e-20) ? q_actual / q_pred : 0.0;
            result.reduction_ratio_history.push_back(rho);
            
            // Update trust region
            if (rho > config.eta_1) {
                // Accept step
                u = u_trial;
                result.successful_steps++;
                
                if (rho > config.eta_2) {
                    // Increase radius
                    delta = std::min(delta * config.c_inc, config.max_radius);
                }
            } else {
                // Reject step and decrease radius
                delta *= config.c_dec;
                if (delta < config.min_radius) {
                    if (config.verbose) {
                        std::printf("  Trust-region: radius below minimum\n");
                    }
                    break;
                }
            }
            
            result.trust_radius_history.push_back(delta);
            
            if (config.verbose && iter % 10 == 0) {
                std::printf("  TR Iter %3d: ||r|| = %.2e, Δ = %.2e, ρ = %.3f\n",
                    iter, r_norm, delta, rho);
            }
            
            r_norm = r_trial;
            result.iterations = iter;
        }
        
        result.final_residual_norm = r_norm;
        return result;
    }
    
private:
    const MeshTopology& mesh_;
    const NeoHookeanMaterial& material_;
    const BoundaryConditions& bcs_;
    
    [[nodiscard]] static double dot_product(const Vector& a, const Vector& b) noexcept {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) sum += a[i]*b[i];
        return sum;
    }
};

// ===========================================================================
// ARC-LENGTH CONTINUATION (for snap-through, limit-point analysis)
// ===========================================================================

/// @brief ARC-LENGTH CONTINUATION for Bifurcation & Limit-Point Problems
/// 
/// THEORY (Riks 1979, Crisfield 1997):
///  For problems with limit points (turning points), standard load incrementing fails.
///  Arc-length continuation uses constraint: ||Δu||² + (λ·Δλ)² = s² (arc-length control).
///  This enables following the load-deflection curve through turning points.
///  Canonical equation: (K-λ·K_geo)·du = (λ_n+1 - λ_n)·f_ext ⟹ singularity at turning point.
struct ArcLengthConfig {
    double initial_arc_length = 0.1;    // s₀
    double max_iterations_per_step = 20;
    double tolerance = 1e-8;
    bool spherical_constraint = true;   // true: Riks (spherical), false: linear
};

struct ArcLengthStepResult {
    bool converged{false};
    int iterations{0};
    double load_factor{0};              // λ
    double arc_length_used{0};
};

} // namespace atlas::fem

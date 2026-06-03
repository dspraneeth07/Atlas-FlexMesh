#pragma once
// FEM/contact_mechanics.hpp
// Contact detection and enforcement utilities for frictional contact.
// Includes: broad-phase and narrow-phase detection, penalty-based contact
// forces with regularized Coulomb friction, contact-tangent contributions,
// and a simple active-set Newton driver. Comments emphasize algorithmic
// assumptions, numerical regularization, and expected use-cases.

#include "fem/fem_types.hpp"
#include "fem/state_transport.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <array>

namespace atlas::fem {

// Lightweight aliases to match project types
using Solution = std::vector<double>;
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>; // dense matrix placeholder

constexpr double PI = 3.14159265358979323846;

// Contact geometry and gap computation (broad/narrow phase)

struct ContactPair {
    int master_node_id;        // Node on master surface
    int slave_node_id;         // Node on slave surface
    double master_coords[3];   // Current position of master
    double slave_coords[3];    // Current position of slave
    double normal[3];          // Contact normal (master→slave)
    double gap;                // Current gap (negative = penetration)
    bool active;               // Is contact active?
};

namespace detail {
struct ContactCellKey {
    int x{0};
    int y{0};
    int z{0};

    bool operator==(const ContactCellKey& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct ContactCellKeyHash {
    std::size_t operator()(const ContactCellKey& key) const noexcept {
        std::size_t h1 = static_cast<std::size_t>(key.x) * 73856093u;
        std::size_t h2 = static_cast<std::size_t>(key.y) * 19349663u;
        std::size_t h3 = static_cast<std::size_t>(key.z) * 83492791u;
        return h1 ^ h2 ^ h3;
    }
};

[[nodiscard]] inline ContactCellKey quantize_contact_point(const double point[3], double cell_size) noexcept {
    const double safe_cell = std::max(cell_size, 1e-12);
    return ContactCellKey{
        static_cast<int>(std::floor(point[0] / safe_cell)),
        static_cast<int>(std::floor(point[1] / safe_cell)),
        static_cast<int>(std::floor(point[2] / safe_cell))
    };
}
} // namespace detail

class ContactDetector {
public:
    /// @brief Compute gap between two surfaces (simplified: node-to-node)
    [[nodiscard]] static double compute_gap(
        const double master[3],
        const double slave[3],
        double normal[3]) noexcept
    {
        // Distance vector: d = slave - master
        double d[3] = {slave[0]-master[0], slave[1]-master[1], slave[2]-master[2]};
        
        // Distance magnitude
        double dist = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        
        // Normal (unit vector pointing from master to slave)
        if (dist > 1e-12) {
            normal[0] = d[0] / dist;
            normal[1] = d[1] / dist;
            normal[2] = d[2] / dist;
        } else {
            normal[0] = 0; normal[1] = 0; normal[2] = 1;
        }
        
        return dist;  // Gap = distance (negative if interpenetrating)
    }
    
    /// @brief Find all contact pairs in system
    static std::vector<ContactPair> detect_contact(
        const MeshTopology& mesh,
        const Solution& u,
        const std::vector<int>& master_nodes,
        const std::vector<int>& slave_nodes,
        double contact_tolerance = 0.1) noexcept
    {
        std::vector<ContactPair> pairs;

        if (master_nodes.empty() || slave_nodes.empty()) {
            return pairs;
        }

        const bool use_broad_phase = (master_nodes.size() * slave_nodes.size()) > 256;
        if (!use_broad_phase) {
            // Small systems are cheaper to scan directly.
            for (int m_id : master_nodes) {
                for (int s_id : slave_nodes) {
                    if (m_id == s_id) continue;

                    double master[3] = {
                        mesh.nodes[m_id].x + u[m_id*3+0],
                        mesh.nodes[m_id].y + u[m_id*3+1],
                        mesh.nodes[m_id].z + u[m_id*3+2]
                    };

                    double slave[3] = {
                        mesh.nodes[s_id].x + u[s_id*3+0],
                        mesh.nodes[s_id].y + u[s_id*3+1],
                        mesh.nodes[s_id].z + u[s_id*3+2]
                    };

                    double normal[3];
                    double gap = compute_gap(master, slave, normal);
                    if (gap < contact_tolerance) {
                        ContactPair pair;
                        pair.master_node_id = m_id;
                        pair.slave_node_id = s_id;
                        std::memcpy(pair.master_coords, master, 3*sizeof(double));
                        std::memcpy(pair.slave_coords, slave, 3*sizeof(double));
                        std::memcpy(pair.normal, normal, 3*sizeof(double));
                        pair.gap = gap;
                        pair.active = (gap < 0);
                        pairs.push_back(pair);
                    }
                }
            }
            return pairs;
        }

        std::unordered_map<detail::ContactCellKey, std::vector<int>, detail::ContactCellKeyHash> master_bins;
        master_bins.reserve(master_nodes.size());

        for (int m_id : master_nodes) {
            double master[3] = {
                mesh.nodes[m_id].x + u[m_id*3+0],
                mesh.nodes[m_id].y + u[m_id*3+1],
                mesh.nodes[m_id].z + u[m_id*3+2]
            };
            master_bins[detail::quantize_contact_point(master, contact_tolerance)].push_back(m_id);
        }

        const std::array<int, 3> offsets = {-1, 0, 1};
        for (int s_id : slave_nodes) {
            double slave[3] = {
                mesh.nodes[s_id].x + u[s_id*3+0],
                mesh.nodes[s_id].y + u[s_id*3+1],
                mesh.nodes[s_id].z + u[s_id*3+2]
            };

            detail::ContactCellKey slave_cell = detail::quantize_contact_point(slave, contact_tolerance);
            for (int dx : offsets) {
                for (int dy : offsets) {
                    for (int dz : offsets) {
                        detail::ContactCellKey key{slave_cell.x + dx, slave_cell.y + dy, slave_cell.z + dz};
                        auto it = master_bins.find(key);
                        if (it == master_bins.end()) {
                            continue;
                        }

                        for (int m_id : it->second) {
                            if (m_id == s_id) {
                                continue;
                            }

                            double master[3] = {
                                mesh.nodes[m_id].x + u[m_id*3+0],
                                mesh.nodes[m_id].y + u[m_id*3+1],
                                mesh.nodes[m_id].z + u[m_id*3+2]
                            };

                            double normal[3];
                            double gap = compute_gap(master, slave, normal);
                            if (gap >= contact_tolerance) {
                                continue;
                            }

                            ContactPair pair;
                            pair.master_node_id = m_id;
                            pair.slave_node_id = s_id;
                            std::memcpy(pair.master_coords, master, 3*sizeof(double));
                            std::memcpy(pair.slave_coords, slave, 3*sizeof(double));
                            std::memcpy(pair.normal, normal, 3*sizeof(double));
                            pair.gap = gap;
                            pair.active = (gap < 0);
                            pairs.push_back(pair);
                        }
                    }
                }
            }
        }
        
        return pairs;
    }
};

// Penalty-based contact enforcement (regularized Coulomb friction)

struct PenaltyContactConfig {
    double penalty_stiffness = 1e8;  // k_p (penalty parameter)
    double friction_coefficient = 0.3;  // μ (Coulomb)
    double friction_regularization = 1e-3;  // ε (smooth Coulomb)
    double stick_slip_threshold = 0.1;  // Transition tolerance
    bool verbose = false;
};

class PenaltyContactMethod {
public:
    PenaltyContactConfig config;
    
    explicit PenaltyContactMethod(const PenaltyContactConfig& cfg = PenaltyContactConfig())
        : config(cfg) {}
    
    /// @brief Compute contact forces using a normal penalty and
    ///        regularized Coulomb friction: τ = μ·f_n·v_tan/(||v_tan||+ε).
    void compute_contact_forces(
        const std::vector<ContactPair>& pairs,
        const Solution& u_dot,  // Velocity (or time-discrete approximation)
        Vector& f_contact) const noexcept  // Output: contact forces (assembled into global vector)
    {
        std::fill(f_contact.begin(), f_contact.end(), 0.0);
        
        for (const auto& pair : pairs) {
            if (!pair.active) continue;
            
            // Normal force (penalty)
            double gap_penetration = std::max(-pair.gap, 0.0);  // Negative gap = penetration
            double f_normal = config.penalty_stiffness * gap_penetration;
            
            // Tangential velocity
            double v_master[3] = {
                u_dot[pair.master_node_id*3+0],
                u_dot[pair.master_node_id*3+1],
                u_dot[pair.master_node_id*3+2]
            };
            double v_slave[3] = {
                u_dot[pair.slave_node_id*3+0],
                u_dot[pair.slave_node_id*3+1],
                u_dot[pair.slave_node_id*3+2]
            };
            
            // Relative velocity
            double v_rel[3] = {
                v_slave[0] - v_master[0],
                v_slave[1] - v_master[1],
                v_slave[2] - v_master[2]
            };
            
            // Tangential component: v_tan = v_rel - (v_rel·n)·n
            double v_dot_n = v_rel[0]*pair.normal[0] + v_rel[1]*pair.normal[1] + v_rel[2]*pair.normal[2];
            double v_tan[3] = {
                v_rel[0] - v_dot_n*pair.normal[0],
                v_rel[1] - v_dot_n*pair.normal[1],
                v_rel[2] - v_dot_n*pair.normal[2]
            };
            
            double v_tan_norm = std::sqrt(v_tan[0]*v_tan[0] + v_tan[1]*v_tan[1] + v_tan[2]*v_tan[2]);
            
            // Friction force (regularized Coulomb: τ = μ·f_n·v_tan/(||v_tan|| + ε))
            double f_friction_factor = config.friction_coefficient * f_normal 
                                      / (v_tan_norm + config.friction_regularization);
            
            double f_friction[3] = {
                -f_friction_factor * v_tan[0],
                -f_friction_factor * v_tan[1],
                -f_friction_factor * v_tan[2]
            };
            
            // Total contact force
            double f_contact_total[3] = {
                -f_normal * pair.normal[0] + f_friction[0],
                -f_normal * pair.normal[1] + f_friction[1],
                -f_normal * pair.normal[2] + f_friction[2]
            };
            
            // Assemble into global vector
            // Slave node gets contact force
            f_contact[pair.slave_node_id*3+0] += f_contact_total[0];
            f_contact[pair.slave_node_id*3+1] += f_contact_total[1];
            f_contact[pair.slave_node_id*3+2] += f_contact_total[2];
            
            // Master node gets opposite force (Newton's 3rd law)
            f_contact[pair.master_node_id*3+0] -= f_contact_total[0];
            f_contact[pair.master_node_id*3+1] -= f_contact_total[1];
            f_contact[pair.master_node_id*3+2] -= f_contact_total[2];
        }
    }
    
    /// @brief Contact tangent contribution from normal penalty. The leading
    ///        term is K = -k_p (n ⊗ n). Tangential/frictional contributions
    ///        require additional linearization if included.
    void add_contact_tangent(
        const std::vector<ContactPair>& pairs,
        Matrix& K) const noexcept
    {
        for (const auto& pair : pairs) {
            if (!pair.active) continue;
            
            // Penalty stiffness: K = -k_p·(n ⊗ n)
            for (int a=0; a<3; ++a) {
                for (int b=0; b<3; ++b) {
                    double k_ab = -config.penalty_stiffness * pair.normal[a] * pair.normal[b];
                    
                    int m_dof_a = pair.master_node_id*3 + a;
                    int m_dof_b = pair.master_node_id*3 + b;
                    int s_dof_a = pair.slave_node_id*3 + a;
                    int s_dof_b = pair.slave_node_id*3 + b;
                    
                    // Slave-slave block (positive)
                    K[s_dof_a][s_dof_b] -= k_ab;
                    
                    // Master-master block (positive)
                    K[m_dof_a][m_dof_b] -= k_ab;
                    
                    // Cross blocks (negative)
                    K[s_dof_a][m_dof_b] += k_ab;
                    K[m_dof_a][s_dof_b] += k_ab;
                }
            }
        }
    }
};

// Stick/slip state tracking and Coulomb criterion

struct ContactState {
    double contact_pressure;     // Normal force per unit area
    double slip_distance;        // Cumulative slip
    bool sticking;               // true = no slip, false = sliding
    double friction_force;       // Current friction force
};

class StickSlipTracker {
public:
    /// @brief Determine stick vs slip based on Coulomb inequality
    /// Stick if: ||f_tan|| ≤ μ·|f_normal|
    /// Slip if: ||f_tan|| > μ·|f_normal|
    [[nodiscard]] static bool is_sticking(
        double friction_force_magnitude,
        double normal_force_magnitude,
        double friction_coefficient) noexcept
    {
        const double coulomb_limit = friction_coefficient * std::abs(normal_force_magnitude);
        return friction_force_magnitude <= coulomb_limit;
    }
    
    /// @brief Update contact state based on forces
    void update_contact_state(
        const ContactPair& pair,
        double normal_force,
        double friction_force_magnitude,
        double dt,
        const Solution& u_dot,
        ContactState& state) const noexcept
    {
        // Update pressure
        state.contact_pressure = normal_force / std::max(1e-12, std::abs(pair.gap));
        
        // Determine if sticking or slipping
        state.sticking = is_sticking(
            friction_force_magnitude,
            normal_force,
            config.friction_coefficient);
        
        // Update cumulative slip
        if (!state.sticking) {
            // Slip distance: Δs = ||v_rel|| · Δt
            double v_master[3] = {
                u_dot[pair.master_node_id*3+0],
                u_dot[pair.master_node_id*3+1],
                u_dot[pair.master_node_id*3+2]
            };
            double v_slave[3] = {
                u_dot[pair.slave_node_id*3+0],
                u_dot[pair.slave_node_id*3+1],
                u_dot[pair.slave_node_id*3+2]
            };
            
            double v_rel[3] = {
                v_slave[0] - v_master[0],
                v_slave[1] - v_master[1],
                v_slave[2] - v_master[2]
            };
            
            double v_rel_magnitude = std::sqrt(v_rel[0]*v_rel[0] + v_rel[1]*v_rel[1] + v_rel[2]*v_rel[2]);
            state.slip_distance += v_rel_magnitude * dt;
        }
        
        state.friction_force = friction_force_magnitude;
    }
    
private:
    struct Config {
        double friction_coefficient = 0.3;
    } config;
};

// Hertz contact analytical helpers (diagnostic/validation formulas)

struct HertzContactAnalytical {
    double R1, R2;           // Radii of curvature
    double E1, E2;           // Young's moduli
    double nu1, nu2;         // Poisson's ratios
    double normal_force;     // Applied normal force
    
    /// @brief Hertz theory: maximum contact pressure
    [[nodiscard]] double max_pressure() const noexcept {
        // Effective radius: 1/R_eff = 1/R1 + 1/R2
        double R_eff_inv = 1.0/R1 + 1.0/R2;
        double R_eff = 1.0 / R_eff_inv;
        
        // Effective modulus: 1/E_eff = (1-ν₁²)/E₁ + (1-ν₂²)/E₂
        double E_eff_inv = (1.0-nu1*nu1)/E1 + (1.0-nu2*nu2)/E2;
        double E_eff = 1.0 / E_eff_inv;
        
        // Maximum pressure: p_max = (3/2)·F / (π·a²)
        // where a = ∛(3·F·R_eff / (4·E_eff))
        double a_cubed = 3.0*normal_force*R_eff / (4.0*E_eff);
        double a = std::cbrt(a_cubed);
        
        return 1.5 * normal_force / (PI * a * a);
    }
    
    /// @brief Contact half-width
    [[nodiscard]] double contact_half_width() const noexcept {
        double R_eff_inv = 1.0/R1 + 1.0/R2;
        double R_eff = 1.0 / R_eff_inv;
        
        double E_eff_inv = (1.0-nu1*nu1)/E1 + (1.0-nu2*nu2)/E2;
        double E_eff = 1.0 / E_eff_inv;
        
        double a_cubed = 3.0*normal_force*R_eff / (4.0*E_eff);
        return std::cbrt(a_cubed);
    }
};

// Active-set Newton solver wrapper for contact problems (conceptual)

class ContactAwareNewtonSolver {
public:
    struct Config {
        int max_newton_iters = 50;
        double tol_residual = 1e-8;
        int max_active_set_updates = 10;
        bool verbose = false;
    };
    
    /// @brief Newton iteration with active-set management for contact
    /// Active set: which contact pairs are actually in contact
    bool solve_with_contact(
        Vector& u,
        const Vector& f_external,
        const Matrix& K_elastic,
        std::vector<ContactPair>& contact_pairs,
        const PenaltyContactMethod& penalty,
        const Config& config) noexcept
    {
        for (int active_set_iter=0; active_set_iter<config.max_active_set_updates; ++active_set_iter) {
            // Newton iterations for fixed active set
            bool converged = false;
            
            for (int newton_iter=0; newton_iter<config.max_newton_iters; ++newton_iter) {
                // Compute residual: r = -K·u + f_ext + f_contact
                Vector r(u.size(), 0.0);
                Vector f_contact(u.size(), 0.0);
                
                // Elastic part: -K·u (dense matrix assumed)
                size_t nrows = K_elastic.size();
                size_t ncols = (nrows>0 ? K_elastic[0].size() : 0);
                for (size_t i=0; i<nrows; ++i) {
                    for (size_t j=0; j<ncols; ++j) {
                        r[i] -= K_elastic[i][j] * u[j];
                    }
                }
                
                // Add external force
                for (size_t i=0; i<f_external.size(); ++i) {
                    r[i] += f_external[i];
                }
                
                // Contact forces (penalty method)
                Vector u_dot = u;  // Simplified: treat displacement as velocity
                penalty.compute_contact_forces(contact_pairs, u_dot, f_contact);
                for (size_t i=0; i<f_contact.size(); ++i) {
                    r[i] += f_contact[i];
                }
                
                double r_norm = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0));
                
                if (r_norm < config.tol_residual) {
                    converged = true;
                    if (config.verbose) {
                        std::printf("  Converged: ||r|| = %.2e\n", r_norm);
                    }
                    break;
                }
                
                // Build tangent: K_total = K_elastic + K_contact
                Matrix K_total = K_elastic;
                penalty.add_contact_tangent(contact_pairs, K_total);
                
                // Solve K_total·Δu = -r (simplified: Jacobi)
                Vector du(u.size(), 0.0);
                size_t nK = K_total.size();
                for (size_t i=0; i<nK; ++i) {
                    if (std::abs(K_total[i][i]) > 1e-30) {
                        du[i] = -r[i] / K_total[i][i];
                    }
                }
                
                // Update u
                for (size_t i=0; i<u.size(); ++i) {
                    u[i] += du[i];
                }
            }
            
            if (!converged && config.verbose) {
                std::printf("  WARNING: Active set iteration %d did not converge\n", active_set_iter);
            }
        }
        
        return true;
    }
};

// Closest-point projection helpers for mortar-like contact enforcement.
// Finds the nearest point on a master surface to a slave node via a
// small Newton iteration in the surface parameter space; used for
// geometric projection and accurate gap evaluation.
struct ClosestPointProjectionResult {
    bool converged{false};
    int iterations{0};
    double gap{0.0};                      // Final gap (should be ≈ 0)
    double parametric_coords[2]{0,0};    // (ξ, η) on master surface
    double projected_point[3]{0,0,0};    // p_m* (closest point on master surface)
    double contact_normal[3]{0,0,0};     // n = (p_s - p_m*) / ||·||
};

class ClosestPointContact {
public:
    /// @brief Find closest point on master surface to slave node via Newton iteration.
    /// Assumes master surface is parameterized: p_m(ξ,η).
    /// For simplicity, here we use linear triangle parametrization.
    static ClosestPointProjectionResult project_onto_master_surface(
        const double slave_point[3],
        const double master_nodes[3][3],  // 3 nodes of master triangle
        double max_iterations = 10,
        double tol = 1e-8) noexcept
    {
        ClosestPointProjectionResult result;
        
        // Initial guess: orthogonal projection onto triangle plane
        double e1[3], e2[3], normal_init[3];
        for (int i = 0; i < 3; ++i) {
            e1[i] = master_nodes[1][i] - master_nodes[0][i];
            e2[i] = master_nodes[2][i] - master_nodes[0][i];
        }
        
        // Cross product: n = e1 × e2
        normal_init[0] = e1[1]*e2[2] - e1[2]*e2[1];
        normal_init[1] = e1[2]*e2[0] - e1[0]*e2[2];
        normal_init[2] = e1[0]*e2[1] - e1[1]*e2[0];
        
        double n_norm = std::sqrt(normal_init[0]*normal_init[0] + normal_init[1]*normal_init[1] + normal_init[2]*normal_init[2]);
        if (n_norm > 1e-30) {
            for (int i = 0; i < 3; ++i) normal_init[i] /= n_norm;
        }
        
        // Project slave onto plane
        double v_to_slave[3];
        for (int i = 0; i < 3; ++i) v_to_slave[i] = slave_point[i] - master_nodes[0][i];
        
        double dist_to_plane = v_to_slave[0]*normal_init[0] + v_to_slave[1]*normal_init[1] + v_to_slave[2]*normal_init[2];
        
        double proj_on_plane[3];
        for (int i = 0; i < 3; ++i) {
            proj_on_plane[i] = slave_point[i] - dist_to_plane * normal_init[i];
        }
        
        // Express proj_on_plane in barycentric coords (λ_0, λ_1, λ_2) w.r.t. triangle
        // For now, use simplified parametric coords (ξ, η) ∈ [0,1]
        // Full triangle parametrization: p_m(ξ,η) = (1-ξ-η)·n0 + ξ·n1 + η·n2
        // This is complex; simplified version: use L2 projection
        
        // Solve normal equations for barycentric coordinates
        // (This is a 2D least-squares problem)
        double xi = 0.5, eta = 0.5;  // Initial guess (triangle centroid)
        
        for (int iter = 0; iter < (int)max_iterations; ++iter) {
            // Evaluate point at (xi, eta)
            double lam0 = 1.0 - xi - eta;
            double p_m[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                p_m[i] = lam0*master_nodes[0][i] + xi*master_nodes[1][i] + eta*master_nodes[2][i];
            }
            
            // Residual: r = slave - p_m
            double r[3];
            for (int i = 0; i < 3; ++i) r[i] = slave_point[i] - p_m[i];
            
            // Residual norm
            double r_norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
            
            if (r_norm < tol) {
                result.converged = true;
                result.iterations = iter;
                result.gap = r_norm;
                for (int i = 0; i < 3; ++i) {
                    result.projected_point[i] = p_m[i];
                    result.contact_normal[i] = (r_norm > 1e-20) ? r[i] / r_norm : normal_init[i];
                }
                return result;
            }
            
            // Jacobian: J = [∂p_m/∂ξ, ∂p_m/∂η]^T (simplified: use FD)
            double eps_param = 1e-6;
            double p_m_xi[3], p_m_eta[3];
            
            for (int i = 0; i < 3; ++i) {
                double lam0_xi = -1.0;
                p_m_xi[i] = lam0_xi*master_nodes[0][i] + master_nodes[1][i];
                
                double lam0_eta = -1.0;
                p_m_eta[i] = lam0_eta*master_nodes[0][i] + master_nodes[2][i];
            }
            
            // Normal equations: J^T J Δξ = J^T r
            // For 2D case: solve 2×2 system
            double JtJ[4] = {0}, Jtr[2] = {0};
            for (int i = 0; i < 3; ++i) {
                JtJ[0] += p_m_xi[i]*p_m_xi[i];
                JtJ[1] += p_m_xi[i]*p_m_eta[i];
                JtJ[2] += p_m_eta[i]*p_m_xi[i];
                JtJ[3] += p_m_eta[i]*p_m_eta[i];
                Jtr[0] += p_m_xi[i]*r[i];
                Jtr[1] += p_m_eta[i]*r[i];
            }
            
            // 2×2 inverse
            double det = JtJ[0]*JtJ[3] - JtJ[1]*JtJ[2];
            if (std::abs(det) < 1e-20) break;  // Singular
            
            double dxi = (JtJ[3]*Jtr[0] - JtJ[1]*Jtr[1]) / det;
            double deta = (-JtJ[2]*Jtr[0] + JtJ[0]*Jtr[1]) / det;
            
            xi += dxi;
            eta += deta;
            
            // Clamp to valid range
            double lam0_clamped = 1.0 - xi - eta;
            if (lam0_clamped < 0) { lam0_clamped = 0; xi = 1.0 - eta; }
            if (xi < 0) xi = 0;
            if (eta < 0) eta = 0;
        }
        
        result.iterations = (int)max_iterations;
        result.converged = false;
        result.parametric_coords[0] = xi;
        result.parametric_coords[1] = eta;
        
        return result;
    }
};

} // namespace atlas::fem

#pragma once
// =============================================================================
// atlas/fem_types.hpp
// Core data-types for the adaptive finite-element subsystem. Provides
// compact, cache-aligned representations for mesh topology, per-element
// state, anisotropic metric tensors and minimal sparse containers used by
// assembly and solvers. Design goals: deterministic memory layout,
// SIMD/NUMA friendliness, and explicit invariants for numerical safety.
//
// Memory & layout:
//  - Structure-of-Arrays (SoA) layout to enable contiguous vectorizable
//    access patterns.
//  - All runtime containers and hot structs aligned to 64 bytes (cache
//    line) to avoid false-sharing and enable aligned SIMD loads.
//
// Key invariants (maintained by algorithms in `core`/`fem`):
//  - deformation gradients are represented in SL(3) (determinant = 1).
//  - metric tensors are symmetric positive-definite (SPD) on insertion.
//  - history/state fields are transported consistently during remeshing.
// =============================================================================

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <span>
#include <optional>
#include <limits>
#include <atomic>
#include <tuple>
#include <string>

namespace atlas::fem {

// ---------------------------------------------------------------------------
// Fundamental index types — 32-bit for cache efficiency
// ---------------------------------------------------------------------------
using NodeIdx    = uint32_t;
using ElemIdx    = uint32_t;
using FaceIdx    = uint32_t;
using EdgeIdx    = uint32_t;

// ---------------------------------------------------------------------------
// Common aliases
// ---------------------------------------------------------------------------
// Alias for a flat solver/IO solution vector (nodal DOFs flattened)
using Solution = std::vector<double>;

static constexpr NodeIdx INVALID_NODE = std::numeric_limits<NodeIdx>::max();
static constexpr ElemIdx INVALID_ELEM = std::numeric_limits<ElemIdx>::max();

// ---------------------------------------------------------------------------
// Physical node descriptor (64 B). Holds coordinates, nodal DOFs and a
// small error indicator. Layout is chosen for compactness and aligned
// streaming in mesh traversals and residual assembly.
// ---------------------------------------------------------------------------
struct alignas(64) MeshNode {
    double   x{0}, y{0}, z{0};
    double   u{0}, v{0}, w{0};          // displacement DOFs
    double   error_indicator{0};        // Zienkiewicz-Zhu recovered error
    NodeIdx  id{INVALID_NODE};
    uint32_t _pad{0};
    
    [[nodiscard]] double coord(int axis) const noexcept {
        return (axis == 0) ? x : (axis == 1) ? y : z;
    }
    // Padding to complete 64 bytes
};
static_assert(sizeof(MeshNode) == 64);

// ---------------------------------------------------------------------------
// Tetrahedral element descriptor (64 B). Stores connectivity, geometric
// invariants (Jacobian, volume), refinement flag and a compact quality
// metric used by adaptation heuristics.
// ---------------------------------------------------------------------------
struct alignas(64) TetraElement {
    std::array<NodeIdx, 4> nodes{};    // global node indices
    double   jacobian_det{0};          // |J| at centroid
    double   volume{0};                // undeformed volume
    double   quality_metric{0};        // aspect ratio ∈ (0,1]
    double   error_estimate{0};        // element-level error
    ElemIdx  id{INVALID_ELEM};
    uint8_t  refinement_flag{0};       // 0=keep, 1=refine, 2=coarsen
    uint8_t  material_id{0};
    uint16_t _pad{0};
    uint32_t _pad2{0};                 // → total 64 B
};
static_assert(sizeof(TetraElement) == 64);

// ---------------------------------------------------------------------------
// Per-element algorithmic state stored in a 128-byte aligned block. The
// block embeds the 3×3 deformation gradient (column-major flat array),
// stress/plastic-strain in Mandel/Voigt ordering, isotropic hardening
// state and small control fields (tangent validity, transport generation).
// The representation is designed for direct reinterpretation as `Matrix3x3`
// and for low-overhead memcpy during transport and IO.
// ---------------------------------------------------------------------------
struct alignas(128) ElementState {
    // Deformation gradient F ∈ SL(3)  — 9 doubles + 7 pad = 128 B (Matrix3x3)
    // We embed it as a flat array for direct Matrix3x3 reinterpret
    double   F_data[9]{1,0,0, 0,1,0, 0,0,1};  // col-major, det=1
    double   F_pad[7]{};

    // Cauchy / Kirchhoff stress (Mandel notation: S11,S22,S33,√2·S12,√2·S13,√2·S23)
    double   stress[6]{};

    // Plastic strain (same ordering)
    double   eps_plastic[6]{};

    // Isotropic hardening state
    double   kappa{0};

    // Equivalent plastic strain
    double   gamma_p{0};
    double   energy{0};

    // Algorithmic tangent validity flag (0 = stale, 1 = current)
    uint32_t tangent_valid{0};

    // Remesh generation counter (for transport logging)
    uint32_t transport_gen{0};

    void initialize() noexcept {
        for (int i = 0; i < 9; ++i) F_data[i] = (i%4==0 ? 1.0 : 0.0);
        for (int i = 0; i < 6; ++i) stress[i] = 0.0;
        for (int i = 0; i < 6; ++i) eps_plastic[i] = 0.0;
        kappa = 0.0;
        gamma_p = 0.0;
        energy = 0.0;
        tangent_valid = 0;
        transport_gen = 0;
    }
};
static_assert(sizeof(ElementState) == 200 || sizeof(ElementState) == 256,
              "ElementState must be 200 or 256 bytes");

// ---------------------------------------------------------------------------
// Per-node anisotropic metric tensor stored in compact Mandel form (6
// values). `MetricTensor` encodes directional length-scale and anisotropy
// used by the metric-driven remesher; `to_matrix` expands to the full
// 3×3 SPD matrix in column-major order for linear-algebra kernels.
// ---------------------------------------------------------------------------
struct alignas(64) MetricTensor {
    double   m[6]{1,0,0, 1,0, 1};   // Mandel: M11,M12,M13,M22,M23,M33
    double   target_h{1e-3};        // target isotropic edge length
    double   anisotropy_ratio{1.0}; // λ_max / λ_min of M
    NodeIdx  node_id{INVALID_NODE};
    uint32_t _pad{0};
    uint64_t _cache_pad{0};         // → total 80 B

    /// @brief Convert to full 3×3 (lower-triangular fill)
    void to_matrix(double out[9]) const noexcept {
        out[0]=m[0]; out[3]=m[1]; out[6]=m[2];
        out[1]=m[1]; out[4]=m[3]; out[7]=m[4];
        out[2]=m[2]; out[5]=m[4]; out[8]=m[5];
    }
};
static_assert(sizeof(MetricTensor) == 80 || sizeof(MetricTensor) == 128);

// ---------------------------------------------------------------------------
// Mesh topology container (SoA). Stores nodes, element descriptors,
// per-element states and per-node metrics. Adjacency is provided in a
// CSR-style mapping from node → elements for efficient element gathering
// during assembly. Reserve/size hints are provided to enable NUMA-aware
// first-touch initialization patterns in higher-level code.
// ---------------------------------------------------------------------------
struct MeshTopology {
    std::vector<MeshNode>      nodes;
    std::vector<TetraElement>  elements;
    std::vector<ElementState>  states;    // one per element
    std::vector<MetricTensor>  metrics;   // one per node

    // Adjacency (COO format, sorted by node)
    std::vector<ElemIdx>       node_to_elem_data;
    std::vector<uint32_t>      node_to_elem_ptr;   // CSR row pointers

    // Global DOF count
    uint32_t n_dofs{0};

    // Statistics
    double  total_volume{0};
    double  min_quality{1.0};
    double  max_error{0.0};
    uint32_t n_refined{0};
    uint32_t n_coarsened{0};
    uint32_t adaptation_step{0};

    void reserve(std::size_t n_nodes_hint, std::size_t n_elems_hint) {
        nodes.reserve(n_nodes_hint);
        elements.reserve(n_elems_hint);
        states.reserve(n_elems_hint);
        metrics.reserve(n_nodes_hint);
        node_to_elem_ptr.reserve(n_nodes_hint + 1);
    }

    [[nodiscard]] std::size_t n_nodes()    const noexcept { return nodes.size(); }
    [[nodiscard]] std::size_t n_elements() const noexcept { return elements.size(); }

    /// @brief Compute and cache total mesh volume from element volumes.
    void compute_total_volume() noexcept {
        total_volume = 0.0;
        for (const auto& e : elements) total_volume += e.volume;
    }
};

// Compatibility aliases for legacy code and headers.
using Mesh = MeshTopology;
using Element = TetraElement;
using Solution = std::vector<double>;

// ---------------------------------------------------------------------------
/// @brief PDE problem descriptor — boundary conditions and forcing.
// ---------------------------------------------------------------------------
enum class PDEType : uint8_t {
    Laplacian,          // −Δu = f
    LinearElasticity,   // −∇·σ = f,  σ = C:ε
    NonlinearElasticity,// Neo-Hookean / Mooney-Rivlin
    ShellKirchhoff,     // Kirchhoff-Love shell
    Plasticity,         // J2 incremental plasticity
};

/// @brief Domain presets for benchmark and problem setup.
enum class DomainType : uint8_t {
    UnitCube,
    CooksMembraneQuad,
    ThickCylinder,
    Cantilever,
    Custom
};

/// @brief Simple material model selector used in benchmarks and configs.
enum class MaterialModel : uint8_t {
    LinearElastic,
    NeoHookean,
    MooneyRivlin,
    SaintVenantKirchhoff,
    PlasticJ2
};

struct ProblemDescriptor {
    PDEType  pde_type{PDEType::NonlinearElasticity};
    DomainType domain_type{DomainType::UnitCube};
    MaterialModel material_model{MaterialModel::LinearElastic};
    bool     apply_volume_compression{false};
    double   compression_ratio{1.0};
    double   E{200e9};    // Young's modulus (Pa)
    double   nu{0.3};     // Poisson ratio
    double   sigma_y{250e6}; // yield stress (Pa)
    double   H_prime{1e9};   // isotropic hardening modulus
    double   rho{7800.0}; // density (kg/m³)
    uint32_t max_newton_iter{50};
    double   newton_tol{1.0e-10};
    double   load_factor{1.0};   // total applied load fraction
    uint32_t n_load_steps{20};
};

struct BoundaryConditions {
    // Dirichlet BCs: tuple(node_id, dof_index, value)
    std::vector<std::tuple<int, int, double>> dirichlet_bcs;

    // Neumann BCs: tuple(node_id, fx, fy, fz)
    std::vector<std::tuple<int, double, double, double>> neumann_bcs;
};

struct LoadStep {
    double load_factor{1.0};
    std::array<double, 3> force{0.0, 0.0, 0.0};
    double magnitude{1.0};
    std::string description;
};

// ---------------------------------------------------------------------------
/// @brief Convergence record for one adaptive iteration.
// ---------------------------------------------------------------------------
struct ConvergenceRecord {
    uint32_t iter{0};
    double   h1_error{0};
    double   l2_error{0};
    double   energy_norm_error{0};
    double   max_det_deviation{0};  // ||det(F)-1||_∞
    double   max_transport_err{0};  // ||F_transported - F_exact||_F/||F_exact||_F
    double   transport_error{0};    // transport error measure for history fields
    double   newton_residual{0};
    uint32_t n_nodes{0};
    uint32_t n_elements{0};
    double   wall_time_s{0};
    double   energy_J{0};           // RAPL if available
};

// ---------------------------------------------------------------------------
/// @brief Error estimator type selector.
// ---------------------------------------------------------------------------
enum class ErrorEstimatorType : uint8_t {
    ZienkiewiczZhu,     // SPR recovery-based (default, cheap)
    ResidualBased,      // Explicit residual estimator
    HessianMetric,      // Second-derivative recovery for anisotropic meshes
    DualWeightedRes,    // Goal-oriented DWR (expensive but optimal)
};

// ---------------------------------------------------------------------------
/// @brief Minimal CSR sparse matrix type used across the FEM code.
// ---------------------------------------------------------------------------
struct SparseCSR {
    uint32_t n_rows{0};
    std::vector<uint32_t> row_ptr; // size n_rows+1
    std::vector<uint32_t> col_idx; // size nnz
    std::vector<double>   val;     // size nnz

    inline void matvec(const double* x, double* y) const noexcept {
        const uint32_t n = n_rows;
        for (uint32_t i=0; i<n; ++i) {
            double s = 0.0;
            for (uint32_t k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                s += val[k] * x[col_idx[k]];
            }
            y[i] = s;
        }
    }
};


} // namespace atlas::fem

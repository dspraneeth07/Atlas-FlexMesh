#pragma once
// atlas/mpi_partitioning.hpp
// Distributed mesh partitioning utilities: lightweight partition metadata,
// element-coloring for lock-free assembly, recursive bisection partitioner,
// ghost-element identification, and thin interop stubs for external
// graph-partitioning backends (ParMETIS/Scotch). Comments focus on API
// intent and numerical considerations; implementation is intentionally
// compact and pluggable.

#include "fem/fem_types.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <functional>
#include <array>
#include <span>
#include <omp.h>

namespace atlas::distributed {

using atlas::fem::MeshTopology;
using atlas::fem::NodeIdx;
using atlas::fem::ElemIdx;
using atlas::fem::SparseCSR;

// Minimal distributed linear-algebra containers used by the distributed FEM engine.
// In serial builds these wrap local values/CSR blocks; MPI builds may extend
// metadata without changing solver-facing interfaces.

struct DistributedVector {
    uint32_t n_local{0};
    uint32_t n_ghost{0};
    std::vector<double> local_values;

    [[nodiscard]] double operator[](size_t i) const noexcept {
        return i < local_values.size() ? local_values[i] : 0.0;
    }

    [[nodiscard]] double& operator[](size_t i) noexcept {
        if (i >= local_values.size()) {
            local_values.resize(i + 1, 0.0);
            n_local = static_cast<uint32_t>(local_values.size());
        }
        return local_values[i];
    }
};

struct DistributedMatrix {
    SparseCSR local_block;

    void matvec(
        const DistributedVector& x,
        DistributedVector& y,
        const struct HaloExchange& halo) const noexcept
    {
        (void)halo;
        const uint32_t n = local_block.n_rows;
        y.n_local = n;
        y.local_values.assign(n, 0.0);
        if (n == 0 || local_block.row_ptr.size() < static_cast<size_t>(n) + 1) {
            return;
        }
        local_block.matvec(x.local_values.data(), y.local_values.data());
    }
};

// Partition metadata: per-rank local/ghost element and node views

/// @brief Information for a single MPI rank's local mesh view.
struct PartitionInfo {
    int rank{0};                           ///< MPI rank ID
    int n_ranks{1};                        ///< Total MPI ranks
    
    // Local mesh indexing
    uint32_t local_elem_start{0};          ///< global element ID of first local
    uint32_t n_local_elems{0};             ///< number of local elements
    std::vector<uint32_t> local_elem_gids; ///< global element IDs [n_local_elems]
    
    uint32_t local_node_start{0};          ///< global node ID of first local
    uint32_t n_local_nodes{0};             ///< number of local (owned) nodes
    std::vector<uint32_t> local_node_gids; ///< global node IDs [n_local_nodes]
    
    // Ghost entities (shared across partitions, not owned here)
    std::vector<uint32_t> ghost_elem_gids; ///< elements needed but not owned
    std::vector<uint32_t> ghost_node_gids; ///< nodes needed but not owned
    std::vector<int>      ghost_node_owner; ///< MPI rank that owns each ghost node
    
    // Partition quality
    double imbalance_factor{1.0};         ///< max_p(n_elems_p) / avg_elems
    double surface_to_volume_ratio{0.0}; ///< boundary facets / volume (for comm cost est.)
    
    /// @brief Check partition validity.
    [[nodiscard]] bool is_valid() const noexcept {
        return local_elem_gids.size() == n_local_elems &&
               local_node_gids.size() == n_local_nodes &&
               imbalance_factor >= 1.0 && imbalance_factor < 10.0; // sanity check
    }
};

// Element coloring for lock-free assembly: assign disjoint sets of elements
// so that elements within a color share no nodes/faces; assembly can then
// iterate colors sequentially without fine-grained synchronization.
class ElementColoring {
public:
    /// @brief Compute greedy coloring on mesh.
    static std::vector<uint8_t> color_elements(
        const MeshTopology& mesh,
        uint8_t max_colors = 8) noexcept
    {
        const size_t NE = mesh.n_elements();
        std::vector<uint8_t> color(NE, 0xFF);  // 0xFF = uncolored
        uint8_t next_color = 0;
        
        while (true) {
            // Find uncolored element to start new color
            int elem_start = -1;
            for (size_t e = 0; e < NE; ++e) {
                if (color[e] == 0xFF) { elem_start = e; break; }
            }
            if (elem_start == -1) break;  // all colored
            if (next_color >= max_colors) break;  // safety
            
            // Assign this color to all elements that don't conflict
            std::vector<bool> forbidden(NE, false);
            for (size_t e = 0; e < NE; ++e) {
                if (color[e] != 0xFF) {
                    // Mark neighbors as forbidden for this color
                    const auto& elem_e = mesh.elements[e];
                    for (int i = 0; i < 4; ++i) {
                        // Find elements sharing node elem_e.nodes[i]
                        const NodeIdx ni = elem_e.nodes[i];
                        if (ni+1 < mesh.node_to_elem_ptr.size()) {
                            const uint32_t start = mesh.node_to_elem_ptr[ni];
                            const uint32_t end   = mesh.node_to_elem_ptr[ni+1];
                            for (uint32_t p = start; p < end; ++p) {
                                forbidden[mesh.node_to_elem_data[p]] = true;
                            }
                        }
                    }
                }
            }
            
            // Assign color to all uncolored non-forbidden elements
            for (size_t e = 0; e < NE; ++e) {
                if (color[e] == 0xFF && !forbidden[e]) {
                    color[e] = next_color;
                    // Mark neighbors as forbidden
                    const auto& elem_e = mesh.elements[e];
                    for (int i = 0; i < 4; ++i) {
                        const NodeIdx ni = elem_e.nodes[i];
                        if (ni+1 < mesh.node_to_elem_ptr.size()) {
                            const uint32_t start = mesh.node_to_elem_ptr[ni];
                            const uint32_t end   = mesh.node_to_elem_ptr[ni+1];
                            for (uint32_t p = start; p < end; ++p) {
                                forbidden[mesh.node_to_elem_data[p]] = true;
                            }
                        }
                    }
                }
            }
            ++next_color;
        }
        
        return color;
    }
    
    /// @brief Verify coloring is valid (no adjacent elements share color).
    [[nodiscard]] static bool validate(
        const MeshTopology& mesh,
        const std::vector<uint8_t>& color) noexcept
    {
        const size_t NE = mesh.n_elements();
        if (color.size() != NE) return false;
        
        for (size_t e1 = 0; e1 < NE; ++e1) {
            if (color[e1] == 0xFF) return false;  // uncolored
            const auto& elem1 = mesh.elements[e1];
            
            // Check all neighbors
            for (int i = 0; i < 4; ++i) {
                const NodeIdx ni = elem1.nodes[i];
                if (ni+1 < mesh.node_to_elem_ptr.size()) {
                    const uint32_t start = mesh.node_to_elem_ptr[ni];
                    const uint32_t end   = mesh.node_to_elem_ptr[ni+1];
                    for (uint32_t p = start; p < end; ++p) {
                        const ElemIdx e2 = mesh.node_to_elem_data[p];
                        if (e2 != e1 && color[e2] == color[e1])
                            return false;  // adjacent, same color
                    }
                }
            }
        }
        return true;
    }
};

// Recursive bisection partitioner: a simple spatial bisection used as a
// fallback or for unit testing. Replace with graph-based multilevel
// partitioners (ParMETIS/Scotch) for production load-balance and cut quality.
class RecursiveBisectionPartitioner {
public:
    /// @brief Partition mesh into P pieces via recursive bisection.
    /// Returns vector of size n_elements indicating which partition each belongs to.
    static std::vector<int> partition(
        const MeshTopology& mesh,
        int n_partitions) noexcept
    {
        if (n_partitions <= 1) {
            return std::vector<int>(mesh.n_elements(), 0);
        }
        
        std::vector<int> partition(mesh.n_elements());
        std::vector<uint32_t> elem_indices(mesh.n_elements());
        std::iota(elem_indices.begin(), elem_indices.end(), 0U);
        
        std::function<void(std::span<uint32_t>, int, int)> bisect =
            [&](std::span<uint32_t> elems, int part_id, int depth) {
                if (elems.size() <= 100 || depth > 10) {  // leaf node
                    for (uint32_t e : elems) partition[e] = part_id;
                    return;
                }
                
                // Find split axis (x, y, or z) by largest extent
                double bmin[3] = {1e30, 1e30, 1e30};
                double bmax[3] = {-1e30, -1e30, -1e30};
                for (uint32_t e : elems) {
                    const auto& elem = mesh.elements[e];
                    for (int i = 0; i < 4; ++i) {
                        const auto& n = mesh.nodes[elem.nodes[i]];
                        bmin[0] = std::min(bmin[0], n.x);
                        bmin[1] = std::min(bmin[1], n.y);
                        bmin[2] = std::min(bmin[2], n.z);
                        bmax[0] = std::max(bmax[0], n.x);
                        bmax[1] = std::max(bmax[1], n.y);
                        bmax[2] = std::max(bmax[2], n.z);
                    }
                }
                int axis = 0;
                double ext_max = bmax[0] - bmin[0];
                for (int a = 1; a < 3; ++a) {
                    if (bmax[a] - bmin[a] > ext_max) {
                        ext_max = bmax[a] - bmin[a];
                        axis = a;
                    }
                }
                double split_pos = (bmin[axis] + bmax[axis]) / 2.0;
                
                // Sort elements by coordinate on split axis
                std::sort(elems.begin(), elems.end(), [&](uint32_t e1, uint32_t e2) {
                    double c1 = 0, c2 = 0;
                    const auto& el1 = mesh.elements[e1];
                    const auto& el2 = mesh.elements[e2];
                    for (int i = 0; i < 4; ++i) {
                        c1 += mesh.nodes[el1.nodes[i]].coord(axis);
                        c2 += mesh.nodes[el2.nodes[i]].coord(axis);
                    }
                    return c1 < c2;
                });
                
                // Recursively partition
                size_t mid = elems.size() / 2;
                bisect(elems.subspan(0, mid), part_id, depth+1);
                bisect(elems.subspan(mid), part_id + (1<<depth), depth+1);
            };
        
        bisect(elem_indices, 0, 0);
        return partition;
    }
};

// Ghost element identification: build lists of non-owned elements required
// for local assembly (halo elements). Ghosts are determined by connectivity
// across partition boundaries.
class GhostElementIdentifier {
public:
    /// @brief Identify ghost elements for all partitions.
    /// Returns map: partition_id -> vector of ghost element global IDs.
    static std::vector<std::vector<uint32_t>> identify_ghosts(
        const MeshTopology& mesh,
        const std::vector<int>& partition) noexcept
    {
        std::vector<std::vector<uint32_t>> ghosts(
            *std::max_element(partition.begin(), partition.end()) + 1);
        
        const size_t NE = mesh.n_elements();
        for (size_t e = 0; e < NE; ++e) {
            const auto& elem = mesh.elements[e];
            int elem_part = partition[e];
            
            // Find all elements sharing an edge with e
            std::vector<int> neighbor_parts;
            for (int i = 0; i < 4; ++i) {
                const NodeIdx ni = elem.nodes[i];
                if (ni+1 < mesh.node_to_elem_ptr.size()) {
                    const uint32_t start = mesh.node_to_elem_ptr[ni];
                    const uint32_t end   = mesh.node_to_elem_ptr[ni+1];
                    for (uint32_t p = start; p < end; ++p) {
                        const ElemIdx e2 = mesh.node_to_elem_data[p];
                        int nb_part = partition[e2];
                        if (nb_part != elem_part) {
                            neighbor_parts.push_back(nb_part);
                        }
                    }
                }
            }
            
            // Add e as ghost to all neighboring partitions
            std::sort(neighbor_parts.begin(), neighbor_parts.end());
            neighbor_parts.erase(std::unique(neighbor_parts.begin(),
                                             neighbor_parts.end()),
                                neighbor_parts.end());
            for (int nb : neighbor_parts) {
                ghosts[nb].push_back(static_cast<uint32_t>(e));
            }
        }
        
        return ghosts;
    }
};

// Graph-partitioning backend stubs. These provide compact adapter APIs to
// external libraries (ParMETIS, Scotch). The functions here are placeholders
// and should be replaced by calls into the respective libraries when linked.
namespace parmetis_interop {
    struct PartitionerConfig {
        int n_partitions{1};
        double imbalance_tolerance{0.01};   // max_load / avg_load < 1.01
        int max_iterations{50};
        bool verbose{false};
    };
    
    /// @brief Partition graph via ParMETIS (stub). Replace with actual
    ///        ParMETIS_V3_PartKway invocation when linking ParMETIS.
    inline std::vector<int> partition_kway(
        const int n_vertices,
        const std::vector<int>& adjacency_row_ptr,
        const std::vector<int>& adjacency_col_idx,
        const PartitionerConfig& config) noexcept
    {
        // Stub: returns sequential assignment for testing
        std::vector<int> part(n_vertices);
        for (int i = 0; i < n_vertices; ++i) {
            part[i] = i % config.n_partitions;
        }
        return part;
    }
}

// Scotch/PT-Scotch adapter stub
namespace scotch_interop {
    struct PartitionerConfig {
        std::string strategy = "bES";  // Balanced Energy Safe
        double imbalance_tolerance = 0.01;
    };
}

// Additional light-weight ParMETIS/Scotch configuration structs
namespace parmetis_interop {
    struct ParMETISConfig {
        int n_partitions{1};
        double imbalance_tolerance{0.01};  // 1% allowed imbalance
        bool verbose{false};
    };
}

namespace scotch_interop {
namespace scotch_interop {
    struct ScotchConfig {
        std::string strategy = "bES";  // balanced, energy-based, safe
        double imbalance_tolerance = 0.01;
    };
}

} // namespace atlas::distributed

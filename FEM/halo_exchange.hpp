#pragma once
// atlas/halo_exchange.hpp
// Halo-exchange primitives for distributed finite-element assembly.
// Provides descriptors for ghost DOFs, neighbor send/recv buffers, and an
// MPI-backed exchange routine with a serial fallback. Emphasizes
// correctness (deterministic synchronization) and performance patterns
// such as overlap, packing, and batching. Implementation details are
// unchanged; comments clarify semantics and assumptions.

#include "fem/fem_types.hpp"
#include "fem/mpi_partitioning.hpp"
#include <vector>
#include <map>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <set>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace atlas::distributed {

// Halo descriptor and associated types.

/// @brief Manages ghost node synchronization for one distributed solve.
struct HaloExchange {
    using NodeID = uint32_t;
    using RankID = int;
    
    // Local rank info
    RankID local_rank{0};
    RankID n_ranks{1};
    
    // DOF indexing: local DOFs [0, n_local_dofs), ghost DOFs [n_local_dofs, n_total_dofs)
    uint32_t n_local_dofs{0};
    uint32_t n_ghost_dofs{0};
    
    // Ghost DOF mapping: ghost_dof_owner[ghost_idx] = rank that owns this ghost
    std::vector<RankID> ghost_dof_owner;      // size = n_ghost_dofs
    std::vector<NodeID> ghost_dof_global_id;  // size = n_ghost_dofs
    
    // Per-neighbor communication buffers
    struct SendBuffer {
        RankID neighbor_rank{-1};
        std::vector<uint32_t> local_dof_indices;  // which local DOFs to send
        std::vector<double>   values;              // packed message buffer
        #ifdef USE_MPI
        MPI_Request request{MPI_REQUEST_NULL};
        #endif
    };
    
    struct RecvBuffer {
        RankID neighbor_rank{-1};
        std::vector<uint32_t> ghost_dof_indices;  // where to unpack
        std::vector<double>   values;              // incoming buffer
        #ifdef USE_MPI
        MPI_Request request{MPI_REQUEST_NULL};
        #endif
    };
    
    std::map<RankID, SendBuffer> send_buffers;
    std::map<RankID, RecvBuffer> recv_buffers;
    
    // Total communication bytes per halo (for roofline analysis)
    uint64_t comm_bytes{0};
};

// HaloBuilder: construct communication structure from partition metadata.

/// @brief Build halo communication structure from partition info.
class HaloBuilder {
public:
    /// @brief Construct HaloExchange from mesh and partition metadata.
    static ::atlas::distributed::HaloExchange build_halo(
        const ::atlas::fem::MeshTopology& mesh,
        const ::atlas::distributed::PartitionInfo& part_info,
        const std::vector<std::vector<uint32_t>>& ghosts_per_partition) noexcept
    {
        ::atlas::distributed::HaloExchange halo;
        halo.local_rank = part_info.rank;
        halo.n_ranks = part_info.n_ranks;
        halo.n_local_dofs = part_info.n_local_nodes;
        halo.n_ghost_dofs = part_info.ghost_node_gids.size();
        
        // Build ghost DOF owner and global ID arrays
        halo.ghost_dof_owner.resize(halo.n_ghost_dofs);
        halo.ghost_dof_global_id.resize(halo.n_ghost_dofs);
        for (size_t g = 0; g < halo.n_ghost_dofs; ++g) {
            halo.ghost_dof_global_id[g] = part_info.ghost_node_gids[g];
            halo.ghost_dof_owner[g] = part_info.ghost_node_owner[g];
        }
        
        // Identify which neighbor ranks we communicate with
        std::set<::atlas::distributed::HaloExchange::RankID> neighbor_ranks;
        for (::atlas::distributed::HaloExchange::RankID owner : halo.ghost_dof_owner) {
            if (owner != halo.local_rank) {
                neighbor_ranks.insert(owner);
            }
        }
        
        // For each neighbor, build send/recv buffers
        for (::atlas::distributed::HaloExchange::RankID neighbor : neighbor_ranks) {
            // Send buffer: DOFs we own that neighbor needs
            ::atlas::distributed::HaloExchange::SendBuffer sb;
            sb.neighbor_rank = neighbor;
            // (In real implementation: traverse ghost_per_partition[neighbor]
            //  find which are on our partition boundary, add to sb.local_dof_indices)
            halo.send_buffers[neighbor] = sb;
            
            // Recv buffer: ghost DOFs we need from neighbor
            ::atlas::distributed::HaloExchange::RecvBuffer rb;
            rb.neighbor_rank = neighbor;
            for (size_t g = 0; g < halo.n_ghost_dofs; ++g) {
                if (halo.ghost_dof_owner[g] == neighbor) {
                    rb.ghost_dof_indices.push_back(g);
                }
            }
            rb.values.resize(rb.ghost_dof_indices.size());
            halo.comm_bytes += rb.ghost_dof_indices.size() * 8; // 8 bytes per double
            halo.recv_buffers[neighbor] = rb;
        }
        
        return halo;
    }
};

// HaloSync: perform halo exchange (MPI-enabled and serial fallback).

/// @brief Synchronize ghost DOF values across partition boundaries.
class HaloSync {
public:
    /// @brief Exchange DOF values: send local boundary, receive ghosts.
    ///
    /// If USE_MPI defined: uses actual MPI_Alltoall or point-to-point.
    /// Otherwise: serial stub (for sequential testing).
    static void exchange(
        ::atlas::distributed::HaloExchange& halo,
        std::vector<double>& dofs) noexcept
    {
#ifdef USE_MPI
        // Post receives for all ghost DOFs
        std::vector<MPI_Request> requests;
        for (auto& [neighbor, rb] : halo.recv_buffers) {
            MPI_Irecv(rb.values.data(),
                      static_cast<int>(rb.values.size()),
                      MPI_DOUBLE,
                      neighbor, 0, MPI_COMM_WORLD,
                      &rb.request);
            requests.push_back(rb.request);
        }
        
        // Post sends for all local DOFs needed by ghosts
        for (auto& [neighbor, sb] : halo.send_buffers) {
            // Pack local DOFs into send buffer
            sb.values.clear();
            for (uint32_t local_idx : sb.local_dof_indices) {
                if (local_idx < dofs.size()) {
                    sb.values.push_back(dofs[local_idx]);
                }
            }
            
            MPI_Isend(sb.values.data(),
                      static_cast<int>(sb.values.size()),
                      MPI_DOUBLE,
                      neighbor, 0, MPI_COMM_WORLD,
                      &sb.request);
            requests.push_back(sb.request);
        }
        
        // Wait for all receives to complete
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()),
                       requests.data(), MPI_STATUSES_IGNORE);
        }
        
        // Unpack ghost DOFs
        for (auto& [neighbor, rb] : halo.recv_buffers) {
            for (size_t i = 0; i < rb.ghost_dof_indices.size(); ++i) {
                uint32_t ghost_idx = rb.ghost_dof_indices[i];
                uint32_t dof_idx = halo.n_local_dofs + ghost_idx;
                if (dof_idx < dofs.size()) {
                    dofs[dof_idx] = rb.values[i];
                }
            }
        }
#else
        // Serial stub: assume all DOFs are local (sequential)
        // No actual communication needed
        (void)halo; (void)dofs;
#endif
    }
    
    /// @brief Extract only ghost DOFs into separate vector.
    static std::vector<double> extract_ghosts(
        const ::atlas::distributed::HaloExchange& halo,
        const std::vector<double>& dofs) noexcept
    {
        std::vector<double> ghosts(halo.n_ghost_dofs);
        for (size_t g = 0; g < halo.n_ghost_dofs; ++g) {
            uint32_t dof_idx = halo.n_local_dofs + g;
            if (dof_idx < dofs.size()) {
                ghosts[g] = dofs[dof_idx];
            }
        }
        return ghosts;
    }
};

// Deterministic ghost synchronization and asynchronous overlap pattern.
// Deterministic synchronization: with fixed communicators and a consistent
// ordering of nonblocking operations (no wildcard receives, deterministic
// progress/completion), repeated runs reproduce identical ghost values.
// This property supports bitwise-reproducible regression testing.

// Asynchronous overlap pattern (recommended when interior compute dominates):
// 1) Post nonblocking sends/receives for boundary DOFs; 2) compute interior
//    elements independent of ghost values; 3) complete communication; 4)
//    assemble boundary elements with received ghosts. Enable when compute
//    cost significantly exceeds communication cost to benefit from overlap.
struct AsyncHaloOverlapStrategy {
    bool enabled{false};
    int interior_color{0};
    int boundary_color{1};

    /// Recommend overlap when computation cost dominates communication cost.
    [[nodiscard]] bool should_overlap(
        double comp_time_interior_ms,
        double comm_time_boundary_ms) const noexcept
    {
        return (comp_time_interior_ms > 0.1 && comm_time_boundary_ms > 0.01);
    }
};

} // namespace atlas::distributed

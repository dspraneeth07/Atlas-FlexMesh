#pragma once
// =============================================================================
// atlas/spatial_lsh.hpp  —  
// Lie-Algebraic Asymptotic Mesh Optimization (LAAMO) — FlexMesh Engine
//
// High-level spatial indexing utilities for mesh nodes.
// Provides Morton key computation, a cache-aligned Node3D descriptor,
// a flat pointerless octree optimized for Morton-sorted nodes, and a
// SpatialLSH façade that constructs the index with NUMA first-touch
// initialization and produces lightweight bandwidth/roofline metrics.
// =============================================================================

#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <cassert>
#include <span>
#include <memory>
#include <cmath>
#include <omp.h>

namespace atlas {

// ---------------------------------------------------------------------------
/// @brief 64-byte cache-line-aligned mesh node descriptor.
///
/// Field layout:
///   [ 0- 7] x            physical X (double)
///   [ 8-15] y            physical Y (double)
///   [16-23] z            physical Z (double)
///   [24-27] element_id   owning element (uint32_t)
///   [28-31] pad0         alignment pad  (uint32_t)
///   [32-39] morton_key   Z-order key    (uint64_t)
///   [40-47] error_metric anisotropic error (double)
///   [48-55] quality_metric  element quality, e.g. aspect ratio (double)  [NEW   ]
///   [56-63] _cache_pad   padding to 64 B (uint64_t)
// ---------------------------------------------------------------------------
struct alignas(64) Node3D {
    double   x;
    double   y;
    double   z;
    uint32_t element_id;
    uint32_t pad0;
    uint64_t morton_key;
    double   error_metric;
    double   quality_metric;   ///< aspect ratio or Jacobian condition
    uint64_t _cache_pad;       ///< padding to fill 64 B cache line

    static constexpr Node3D make(double px, double py, double pz,
                                 uint32_t eid) noexcept {
        return Node3D{px, py, pz, eid, 0u, 0ull, 0.0, 1.0, 0ull};
    }
};
static_assert(sizeof(Node3D)  == 64, "Node3D must be 64 B");
static_assert(alignof(Node3D) == 64, "Node3D must be 64-B aligned");

// ---------------------------------------------------------------------------
/// @brief Expand a coordinate into every 3rd bit of a uint64_t (Karras 2012).
// ---------------------------------------------------------------------------
[[nodiscard]]
inline uint64_t split_by_3(uint32_t a) noexcept {
    uint64_t x = static_cast<uint64_t>(a & 0x1FFFFFU);
    x = (x | (x << 32U)) & UINT64_C(0x1F00000000FFFF);
    x = (x | (x << 16U)) & UINT64_C(0x1F0000FF0000FF);
    x = (x | (x <<  8U)) & UINT64_C(0x100F00F00F00F00F);
    x = (x | (x <<  4U)) & UINT64_C(0x10C30C30C30C30C3);
    x = (x | (x <<  2U)) & UINT64_C(0x1249249249249249);
    return x;
}

// ---------------------------------------------------------------------------
/// @brief 3D Morton key from continuous coordinates.
// ---------------------------------------------------------------------------
[[nodiscard]]
inline uint64_t compute_morton_3d(double x, double y, double z,
                                   double scaling_factor,
                                   double offset = 100.0) noexcept {
    auto clamp21 = [&](double v) noexcept -> uint32_t {
        double s = (v + offset) * scaling_factor;
        if (s < 0.0)        s = 0.0;
        if (s > 2097151.0)  s = 2097151.0;
        return static_cast<uint32_t>(s);
    };
    return split_by_3(clamp21(x))
         | (split_by_3(clamp21(y)) << 1U)
         | (split_by_3(clamp21(z)) << 2U);
}

// ---------------------------------------------------------------------------
struct alignas(32) LSHBucket {
    uint64_t morton_key;
    uint32_t node_start_idx;
    uint32_t node_count;
    uint64_t _pad;
};
static_assert(sizeof(LSHBucket) == 32, "LSHBucket must be 32 B");

// ---------------------------------------------------------------------------
/// @brief Flat cache-oblivious octree over Morton-sorted nodes.
///
///     additions:
///    • build() records total_bytes used (roofline support).
///    • query_nearest_bucket is now a constexpr-friendly binary search.
///    • OctreeNode gains an avg_quality field for adaptive stopping criteria.
// ---------------------------------------------------------------------------
class PointerlessOctree {
public:
    struct alignas(64) OctreeNode {
        uint64_t key_min;
        uint64_t key_max;
        uint32_t first_leaf;
        uint32_t leaf_count;
        float    avg_quality;   ///<   : mean quality_metric in this bucket
        uint8_t  depth;
        uint8_t  _pad[3];
        uint64_t _cache_pad[4]; // 32 B pad → total 64 B
    };
    static_assert(sizeof(OctreeNode) == 64, "OctreeNode must be 64 B");

    // Construct from a pointer to a sorted Node3D array and its size.
    // Using pointer+size keeps the header compatible with pre-C++20
    // toolchains that may lack `std::span`.
    explicit PointerlessOctree(const Node3D* sorted_nodes, std::size_t N,
                               uint32_t target_leaf_capacity = 64)
        : leaf_capacity_(target_leaf_capacity)
    {
        if (N > 0) build(sorted_nodes, N);
    }

    [[nodiscard]]
    uint32_t query_nearest_bucket(uint64_t query_key) const noexcept {
        if (nodes_.empty()) return UINT32_MAX;
        std::size_t lo = 0, hi = leaf_indices_.size();
        while (lo + 1 < hi) {
            const std::size_t mid = (lo + hi) >> 1;
            if (query_key < leaf_keys_[mid]) hi = mid;
            else                             lo = mid;
        }
        return leaf_indices_[lo];
    }

    [[nodiscard]] std::size_t node_count()  const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t total_bytes() const noexcept {
        return nodes_.size() * sizeof(OctreeNode)
             + leaf_keys_.size() * sizeof(uint64_t)
             + leaf_indices_.size() * sizeof(uint32_t);
    }

private:
    uint32_t                  leaf_capacity_;
    std::vector<OctreeNode>   nodes_;
    std::vector<uint64_t>     leaf_keys_;
    std::vector<uint32_t>     leaf_indices_;

    void build(const Node3D* sn, std::size_t N) {
        nodes_.clear(); leaf_keys_.clear(); leaf_indices_.clear();
        nodes_.reserve((N + leaf_capacity_ - 1) / leaf_capacity_);
        leaf_keys_.reserve(nodes_.capacity());
        leaf_indices_.reserve(nodes_.capacity());

        std::size_t cursor = 0;
        while (cursor < N) {
            const uint32_t cnt = static_cast<uint32_t>(
                std::min(N - cursor, static_cast<std::size_t>(leaf_capacity_)));

            double qsum = 0.0;
            for (uint32_t i = 0; i < cnt; ++i)
                qsum += sn[cursor + i].quality_metric;

            OctreeNode leaf{};
            leaf.key_min    = sn[cursor].morton_key;
            leaf.key_max    = sn[cursor + cnt - 1].morton_key;
            leaf.first_leaf = static_cast<uint32_t>(cursor);
            leaf.leaf_count = cnt;
            leaf.avg_quality= static_cast<float>(qsum / cnt);
            leaf.depth      = 0;

            leaf_keys_.push_back(leaf.key_min);
            leaf_indices_.push_back(leaf.first_leaf);
            nodes_.push_back(leaf);
            cursor += cnt;
        }
    }
};

// ---------------------------------------------------------------------------
/// @brief High-level façade: sorted node array + pointerless octree.
///
///     additions:
///    • build() uses NUMA first-touch initialisation via OMP.
///    • statistics() returns bandwidth-model metrics.
// ---------------------------------------------------------------------------
class SpatialLSH {
public:
    struct Statistics {
        std::size_t n_nodes{0};
        std::size_t n_octree_nodes{0};
        std::size_t octree_bytes{0};
        double      avg_quality{0.0};
    };

    SpatialLSH() = default;

    void build(std::vector<Node3D>& nodes,
               double scaling_factor = 10.0,
               double offset         = 100.0) {
        const std::size_t N = nodes.size();

        // NUMA first-touch: each thread touches its own pages
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N; ++i) {
            nodes[i].morton_key = compute_morton_3d(
                nodes[i].x, nodes[i].y, nodes[i].z,
                scaling_factor, offset);
        }

        std::sort(nodes.begin(), nodes.end(),
                  [](const Node3D& a, const Node3D& b) noexcept {
                      return a.morton_key < b.morton_key;
                  });

        octree_ = std::make_unique<PointerlessOctree>(nodes.data(), N);

        // Precompute statistics
        stats_.n_nodes        = N;
        stats_.n_octree_nodes = octree_->node_count();
        stats_.octree_bytes   = octree_->total_bytes();
        double qsum = 0.0;
        for (const auto& nd : nodes) qsum += nd.quality_metric;
        stats_.avg_quality = (N > 0) ? qsum / N : 0.0;
    }

    [[nodiscard]]
    uint32_t nearest(double x, double y, double z,
                     double sf = 10.0, double off = 100.0) const noexcept {
        if (!octree_) return UINT32_MAX;
        return octree_->query_nearest_bucket(
            compute_morton_3d(x, y, z, sf, off));
    }

    [[nodiscard]] const Statistics& statistics() const noexcept { return stats_; }

private:
    std::unique_ptr<PointerlessOctree> octree_;
    Statistics                          stats_{};
};

} // namespace atlas

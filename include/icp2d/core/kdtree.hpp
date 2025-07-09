

// While the following KdTree code is written from scratch, it is heavily
// inspired by the nanoflann library. Thus, the following original license of
// nanoflann is included to be sure.

// https://github.com/jlblancoc/nanoflann/blob/master/include/nanoflann.hpp
/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2024  Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/
#pragma once

#include "icp2d/core/knn_result.hpp"
#include "icp2d/core/projection.hpp"
#include "icp2d/core/traits.hpp"
#include <Eigen/Core>
#include <atomic>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

namespace icp2d {

// Forward declarations
template <typename PointCloud, typename Projection = AxisAlignedProjection>
struct UnsafeKdTree;

template <typename PointCloud, typename Projection = AxisAlignedProjection>
struct SafeKdTree;

template <typename PointCloud, typename Projection = AxisAlignedProjection>
struct KdTree;

/// @brief KdTree node.
template <typename Projection> struct KdTreeNode {
  // Node type
  bool is_leaf = false; ///< Whether this node is a leaf node.

  // Leaf node data
  size_t first =
      0; ///< First point index in the leaf node (valid when is_leaf = true)
  size_t last =
      0; ///< Last point index in the leaf node (valid when is_leaf = true)

  // Internal node data
  Projection proj;         ///< Projection function (valid when is_leaf = false)
  double     thresh = 0.0; ///< Threshold value (valid when is_leaf = false)

  size_t left = std::numeric_limits<size_t>::max(); ///< Left child node index.
  size_t right =
      std::numeric_limits<size_t>::max(); ///< Right child node index.

  // Default constructor
  KdTreeNode() = default;

  // Helper methods
  void set_as_leaf(size_t f, size_t l) {
    is_leaf = true;
    first   = f;
    last    = l;
  }

  void set_as_internal(const Projection &p, double t, size_t l, size_t r) {
    is_leaf = false;
    proj    = p;
    thresh  = t;
    left    = l;
    right   = r;
  }
};

/// @brief KdTree builder.
template <typename Projection> struct KdTreeBuilder {
  /// @brief Build KdTree.
  /// @param kdtree  KdTree to build
  /// @param points  Point cloud
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree &kdtree, const PointCloud &points) const {
    const size_t N = traits::size(points);
    if (N == 0) {
      return;
    }

    // Initialize nodes
    kdtree.nodes.resize(2 * N - 1);
    size_t node_count = 0;

    // Build tree recursively
    kdtree.root =
        create_node(kdtree, node_count, points, kdtree.indices.begin(),
                    kdtree.indices.begin(), kdtree.indices.end());

    // Shrink nodes
    kdtree.nodes.resize(node_count);
  }

protected:
  /// @brief Create a node recursively.
  /// @param kdtree        KdTree to build
  /// @param node_count    Number of nodes (will be updated)
  /// @param points        Point cloud
  /// @param global_first  First iterator of the global indices
  /// @param first        First iterator of the current node
  /// @param last         Last iterator of the current node
  /// @return             Index of the created node.
  template <typename PointCloud, typename KdTree, typename IndexConstIterator>
  size_t create_node(KdTree &kdtree, size_t &node_count,
                     const PointCloud &points, IndexConstIterator global_first,
                     IndexConstIterator first, IndexConstIterator last) const {
    const size_t N          = std::distance(first, last);
    const size_t node_index = node_count++;
    auto        &node       = kdtree.nodes[node_index];

    // Create a leaf node if the number of points is small enough
    if (N <= 1) {
      node.set_as_leaf(std::distance(global_first, first),
                       std::distance(global_first, last));
      return node_index;
    }

    // Find the best split axis and threshold
    ProjectionSetting projection_setting;
    auto proj = Projection::find_axis(points, first, last, projection_setting);

    // Find median point
    auto median_itr = first + N / 2;
    std::nth_element(first, median_itr, last, [&](size_t i, size_t j) {
      return proj(traits::point(points, i)) < proj(traits::point(points, j));
    });

    // Get median value for threshold
    double median_val = proj(traits::point(points, *median_itr));

    // Create child nodes
    const size_t left = create_node(kdtree, node_count, points, global_first,
                                    first, median_itr);
    const size_t right =
        create_node(kdtree, node_count, points, global_first, median_itr, last);

    // Set this node as an internal node
    node.set_as_internal(proj, median_val, left, right);

    return node_index;
  }
};

/// @brief Unsafe KdTree implementation for 2D points.
/// @note  This implementation is not thread-safe.
template <typename PointCloud, typename Projection> struct UnsafeKdTree {
public:
  /// @brief Constructor
  /// @param points Point cloud
  template <typename Builder = KdTreeBuilder<Projection>>
  explicit UnsafeKdTree(const PointCloud &points,
                        const Builder    &builder = Builder())
      : points(std::make_shared<PointCloud>(points)) {
    // Initialize indices
    indices.resize(traits::size(points));
    std::iota(indices.begin(), indices.end(), 0);

    // Build tree
    builder.build_tree(*this, points);
  }

  /// @brief Constructor
  /// @param points Point cloud
  template <typename Builder = KdTreeBuilder<Projection>>
  explicit UnsafeKdTree(std::shared_ptr<const PointCloud> points,
                        const Builder                    &builder = Builder())
      : points(points) {
    // Initialize indices
    indices.resize(traits::size(*points));
    std::iota(indices.begin(), indices.end(), 0);

    // Build tree
    builder.build_tree(*this, *points);
  }

  /// @brief Find k-nearest neighbors
  /// @param query Query point
  /// @param k Number of neighbors
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @return Number of found neighbors
  size_t knn_search(const Eigen::Vector2d &query, size_t k, size_t *k_indices,
                    double *k_sq_dists) const {
    KnnResult<-1> result(k_indices, k_sq_dists, k);
    knn_search(query, root, result);
    result.sort();
    return result.num_found();
  }

  /// @brief Find k-nearest neighbors with static k
  /// @param query Query point
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @return Number of found neighbors
  template <size_t k>
  size_t knn_search(const Eigen::Vector2d &query, size_t *k_indices,
                    double *k_sq_dists) const {
    KnnResult<k> result(k_indices, k_sq_dists);
    knn_search(query, root, result);
    result.sort();
    return result.num_found();
  }

  /// @brief Find the nearest neighbor
  /// @param query Query point
  /// @param k_index [out] Index of the nearest neighbor
  /// @param k_sq_dist [out] Squared distance to the nearest neighbor
  /// @return true if a neighbor is found
  bool nearest(const Eigen::Vector2d &query, size_t *k_index,
               double *k_sq_dist) const {
    return knn_search<1>(query, k_index, k_sq_dist) > 0;
  }

private:
  /// @brief Find k-nearest neighbors for 2D point.
  template <typename Result>
  bool knn_search(const Eigen::Vector2d &query, size_t node_index,
                  Result &result) const {
    const auto &node = nodes[node_index];

    // Check if it's a leaf node.
    if (node.is_leaf) {
      // Compare the query point with all points in the leaf node.
      for (size_t i = node.first; i < node.last; i++) {
        const double sq_dist =
            (traits::point(*points, indices[i]) - query).squaredNorm();
        result.push(indices[i], sq_dist);
      }
      return !result.is_full();
    }

    const double val         = node.proj(query);
    const double diff        = val - node.thresh;
    const double cut_sq_dist = diff * diff;

    size_t best_child;
    size_t other_child;

    if (diff < 0.0) {
      best_child  = node.left;
      other_child = node.right;
    } else {
      best_child  = node.right;
      other_child = node.left;
    }

    // Check the best child node first.
    knn_search(query, best_child, result);

    // Check if the other child node needs to be tested.
    // We need to check the other child if:
    // 1. The result is not full yet, or
    // 2. The distance to the splitting plane is less than the worst distance in
    // the result
    if (!result.is_full() || cut_sq_dist < result.worst_distance()) {
      knn_search(query, other_child, result);
    }

    return !result.is_full();
  }

public:
  std::shared_ptr<const PointCloud>   points;  ///< Point cloud
  std::vector<size_t>                 indices; ///< Point indices
  std::vector<KdTreeNode<Projection>> nodes;   ///< KdTree nodes
  size_t                              root;    ///< Root node index
};

/// @brief Safe KdTree implementation for 2D points.
/// @note  This implementation is thread-safe.
template <typename PointCloud, typename Projection> struct SafeKdTree {
public:
  /// @brief Constructor
  /// @param points Point cloud
  template <typename Builder = KdTreeBuilder<Projection>>
  explicit SafeKdTree(const PointCloud &points,
                      const Builder    &builder = Builder())
      : kdtree(points, builder) {}

  /// @brief Constructor
  /// @param points Point cloud
  template <typename Builder = KdTreeBuilder<Projection>>
  explicit SafeKdTree(std::shared_ptr<const PointCloud> points,
                      const Builder                    &builder = Builder())
      : kdtree(points, builder) {}

  /// @brief Find k-nearest neighbors
  /// @param query Query point
  /// @param k Number of neighbors
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @return Number of found neighbors
  size_t knn_search(const Eigen::Vector2d &query, size_t k, size_t *k_indices,
                    double *k_sq_dists) const {
    std::lock_guard<std::mutex> lock(mutex);
    return kdtree.knn_search(query, k, k_indices, k_sq_dists);
  }

  /// @brief Find k-nearest neighbors with static k
  /// @param query Query point
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @return Number of found neighbors
  template <size_t k>
  size_t knn_search(const Eigen::Vector2d &query, size_t *k_indices,
                    double *k_sq_dists) const {
    std::lock_guard<std::mutex> lock(mutex);
    return kdtree.template knn_search<k>(query, k_indices, k_sq_dists);
  }

  /// @brief Find the nearest neighbor
  /// @param query Query point
  /// @param k_index [out] Index of the nearest neighbor
  /// @param k_sq_dist [out] Squared distance to the nearest neighbor
  /// @return true if a neighbor is found
  bool nearest(const Eigen::Vector2d &query, size_t *k_index,
               double *k_sq_dist) const {
    std::lock_guard<std::mutex> lock(mutex);
    return knn_search<1>(query, k_index, k_sq_dist) > 0;
  }

public:
  UnsafeKdTree<PointCloud, Projection> kdtree; ///< KdTree implementation

private:
  mutable std::mutex mutex; ///< Mutex for thread-safety
};

/// @brief KdTree implementation for 2D points.
/// @note  This is a wrapper around UnsafeKdTree and SafeKdTree.
template <typename PointCloud, typename Projection> struct KdTree {
public:
  /// @brief Constructor
  /// @param points Point cloud
  template <typename Builder = KdTreeBuilder<Projection>>
  explicit KdTree(const PointCloud &points, const Builder &builder = Builder())
      : kdtree(points, builder) {}

  /// @brief Constructor
  /// @param points Point cloud
  template <typename Builder = KdTreeBuilder<Projection>>
  explicit KdTree(std::shared_ptr<const PointCloud> points,
                  const Builder                    &builder = Builder())
      : kdtree(points, builder) {}

  /// @brief Find k-nearest neighbors
  /// @param query Query point
  /// @param k Number of neighbors
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @return Number of found neighbors
  size_t knn_search(const Eigen::Vector2d &query, size_t k, size_t *k_indices,
                    double *k_sq_dists) const {
    return kdtree.knn_search(query, k, k_indices, k_sq_dists);
  }

  /// @brief Find k-nearest neighbors with static k
  /// @param query Query point
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @return Number of found neighbors
  template <size_t k>
  size_t knn_search(const Eigen::Vector2d &query, size_t *k_indices,
                    double *k_sq_dists) const {
    return kdtree.template knn_search<k>(query, k_indices, k_sq_dists);
  }

  /// @brief Find the nearest neighbor
  /// @param query Query point
  /// @param k_index [out] Index of the nearest neighbor
  /// @param k_sq_dist [out] Squared distance to the nearest neighbor
  /// @return true if a neighbor is found
  bool nearest(const Eigen::Vector2d &query, size_t *k_index,
               double *k_sq_dist) const {
    return knn_search<1>(query, k_index, k_sq_dist) > 0;
  }

public:
  UnsafeKdTree<PointCloud, Projection> kdtree; ///< KdTree implementation
};

namespace traits {

template <typename PointCloud, typename Projection>
struct Traits<UnsafeKdTree<PointCloud, Projection>> {
  static size_t
  nearest_neighbor_search(const UnsafeKdTree<PointCloud, Projection> &tree,
                          const Eigen::Vector2d &point, size_t *k_indices,
                          double *k_sq_dists) {
    return tree.nearest_neighbor_search(point, k_indices, k_sq_dists);
  }

  static size_t knn_search(const UnsafeKdTree<PointCloud, Projection> &tree,
                           const Eigen::Vector2d &point, size_t k,
                           size_t *k_indices, double *k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

template <typename PointCloud, typename Projection>
struct Traits<KdTree<PointCloud, Projection>> {
  static size_t
  nearest_neighbor_search(const KdTree<PointCloud, Projection> &tree,
                          const Eigen::Vector2d &point, size_t *k_indices,
                          double *k_sq_dists) {
    return tree.nearest_neighbor_search(point, k_indices, k_sq_dists);
  }

  static size_t knn_search(const KdTree<PointCloud, Projection> &tree,
                           const Eigen::Vector2d &point, size_t k,
                           size_t *k_indices, double *k_sq_dists) {
    return tree.knn_search(point, k, k_indices, k_sq_dists);
  }
};

} // namespace traits

} // namespace icp2d

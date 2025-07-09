// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <icp2d/core/kdtree.hpp>

namespace icp2d {

using NodeIndexType = size_t; // Define NodeIndexType as size_t

/// @brief Kd-tree builder with OpenMP for 2D points.
template <typename Projection = AxisAlignedProjection> struct KdTreeBuilderOMP {
public:
  /// @brief Constructor
  /// @param num_threads  Number of threads
  KdTreeBuilderOMP(int num_threads = 4)
      : num_threads(num_threads), max_leaf_size(20) {}

  /// @brief Build KdTree for 2D points
  template <typename KdTree, typename PointCloud>
  void build_tree(KdTree &kdtree, const PointCloud &points) const {
    kdtree.indices.resize(traits::size(points));
    std::iota(kdtree.indices.begin(), kdtree.indices.end(), 0);

    std::atomic<size_t> node_count{0}; // Fix atomic initialization
    kdtree.nodes.resize(traits::size(points));

#pragma omp parallel num_threads(num_threads)
    {
#pragma omp single nowait
      {
        kdtree.root =
            create_node(kdtree, node_count, points, kdtree.indices.begin(),
                        kdtree.indices.begin(), kdtree.indices.end());
      }
    }

    kdtree.nodes.resize(node_count.load());
  }

  /// @brief Create a Kd-tree node from the given point indices for 2D.
  /// @param global_first     Global first point index iterator (i.e.,
  /// this->indices.begin()).
  /// @param first            First point index iterator to be scanned.
  /// @param last             Last point index iterator to be scanned.
  /// @return                 Index of the created node.
  template <typename PointCloud, typename KdTree, typename IndexConstIterator>
  NodeIndexType
  create_node(KdTree &kdtree, std::atomic<size_t> &node_count,
              const PointCloud &points, IndexConstIterator global_first,
              IndexConstIterator first, IndexConstIterator last) const {
    const size_t        N          = std::distance(first, last);
    const NodeIndexType node_index = node_count.fetch_add(1);
    auto               &node       = kdtree.nodes[node_index];

    // Create a leaf node.
    if (N <= static_cast<size_t>(max_leaf_size)) {
      node.set_as_leaf(std::distance(global_first, first),
                       std::distance(global_first, last));
      return node_index;
    }

    // Find the best axis to split the input points.
    const auto proj =
        Projection::find_axis(points, first, last, projection_setting);
    const auto median_itr = first + N / 2;
    std::nth_element(first, median_itr, last, [&](size_t i, size_t j) {
      return proj(traits::point(points, i)) < proj(traits::point(points, j));
    });

    // Create a non-leaf node.
    node.set_as_internal(proj, proj(traits::point(points, *median_itr)), 0, 0);

    // Create left and right child nodes.
#pragma omp task default(shared) if (N > 512)
    node.left = create_node(kdtree, node_count, points, global_first, first,
                            median_itr);
#pragma omp task default(shared) if (N > 512)
    node.right =
        create_node(kdtree, node_count, points, global_first, median_itr, last);
#pragma omp taskwait

    return node_index;
  }

public:
  int               num_threads;   ///< Number of threads
  int               max_leaf_size; ///< Maximum number of points in a leaf node.
  ProjectionSetting projection_setting; ///< Projection setting.
};

} // namespace icp2d

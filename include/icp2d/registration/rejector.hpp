// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace icp2d {

/// @brief Null correspondence rejector. This class accepts all input
/// correspondences.
struct NullRejector {
  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_index, size_t source_index,
                  double sq_dist) const {
    (void)target;
    (void)source;
    (void)T;
    (void)target_index;
    (void)source_index;
    (void)sq_dist;
    return false;
  }
};

/// @brief Rejecting correspondences with large distances.
struct DistanceRejector {
  DistanceRejector(double max_distance = 1.0)
      : max_dist_sq(max_distance * max_distance) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_index, size_t source_index,
                  double sq_dist) const {
    (void)target;
    (void)source;
    (void)T;
    (void)target_index;
    (void)source_index;
    return sq_dist > max_dist_sq;
  }

  double max_dist_sq; ///< Maximum squared distance between corresponding points
};

} // namespace icp2d

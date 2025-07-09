// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <icp2d/core/traits.hpp>

namespace icp2d {

/// @brief Null rejector that accepts all correspondences
struct NullRejector {
  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_idx, size_t source_idx,
                  double sq_distance) const {
    return false;
  }
};

/// @brief Rejects correspondences based on distance threshold
struct DistanceRejector {
  /// @brief Constructor
  /// @param max_distance Maximum allowed distance between corresponding points
  explicit DistanceRejector(double max_distance = 1.0)
      : max_sq_distance(max_distance * max_distance) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_idx, size_t source_idx,
                  double sq_distance) const {
    return sq_distance > max_sq_distance;
  }

  double max_sq_distance; ///< Maximum squared distance threshold
};

/// @brief Rejects correspondences based on normal consistency
struct NormalRejector {
  /// @brief Constructor
  /// @param min_cos_angle Minimum cosine of angle between normals (default:
  /// cos(45°))
  explicit NormalRejector(double min_cos_angle = 0.7071)
      : min_cos_angle(min_cos_angle) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_idx, size_t source_idx,
                  double sq_distance) const {
    // Skip check if either point cloud doesn't have normals
    if (!traits::has_normals(target) || !traits::has_normals(source)) {
      return false;
    }

    // Get and transform source normal
    const Eigen::Vector2d source_normal = traits::normal(source, source_idx);
    const Eigen::Vector2d transformed_normal = T.linear() * source_normal;

    // Get target normal
    const Eigen::Vector2d target_normal = traits::normal(target, target_idx);

    // Check angle between normals
    const double cos_angle = transformed_normal.dot(target_normal);
    return cos_angle < min_cos_angle;
  }

  double min_cos_angle; ///< Minimum cosine of angle between normals
};

/// @brief Combined rejector that uses both distance and normal consistency
struct CombinedRejector {
  /// @brief Constructor
  /// @param max_distance Maximum allowed distance between points
  /// @param min_cos_angle Minimum cosine of angle between normals
  CombinedRejector(double max_distance = 1.0, double min_cos_angle = 0.7071)
      : distance_rejector(max_distance), normal_rejector(min_cos_angle) {}

  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_idx, size_t source_idx,
                  double sq_distance) const {
    // Reject if either condition fails
    return distance_rejector(target, source, T, target_idx, source_idx,
                             sq_distance) ||
           normal_rejector(target, source, T, target_idx, source_idx,
                           sq_distance);
  }

  DistanceRejector distance_rejector;
  NormalRejector   normal_rejector;
};

} // namespace icp2d

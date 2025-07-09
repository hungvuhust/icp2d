// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <icp2d/core/traits.hpp>
#include <icp2d/util/lie.hpp>

namespace icp2d {

/// @brief Point-to-line per-point error factor for 2D ICP.
/// @note  Computes the perpendicular distance from a point to a line in 2D
struct PointToLineICPFactor {
  struct Setting {
    double max_correspondence_distance; ///< Maximum distance for valid
                                        ///< correspondence
    Setting(double max_correspondence_distance = 1.0)
        : max_correspondence_distance(max_correspondence_distance) {}
  };

  PointToLineICPFactor(const Setting &setting = Setting())
      : setting(setting), target_index(std::numeric_limits<size_t>::max()),
        source_index(std::numeric_limits<size_t>::max()) {}

  /// @brief Linearize the point-to-line error for 2D ICP
  /// @tparam TargetPointCloud Target point cloud type
  /// @tparam SourcePointCloud Source point cloud type
  /// @tparam TargetTree Target tree type (KDTree)
  /// @tparam CorrespondenceRejector Correspondence rejector type
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param target_tree Target KDTree for nearest neighbor search
  /// @param T Current 2D transformation estimate (SE(2))
  /// @param source_index Index of source point
  /// @param rejector Correspondence rejector
  /// @param H Output Hessian matrix (3x3 for SE(2))
  /// @param b Output gradient vector (3x1 for SE(2))
  /// @param e Output error value
  /// @return true if linearization successful
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector>
  bool linearize(const TargetPointCloud &target, const SourcePointCloud &source,
                 const TargetTree &target_tree, const Eigen::Isometry2d &T,
                 size_t source_index, const CorrespondenceRejector &rejector,
                 Eigen::Matrix<double, 3, 3> *H, Eigen::Matrix<double, 3, 1> *b,
                 double *e) {

    this->source_index = source_index;
    this->target_index = std::numeric_limits<size_t>::max();

    // Transform source point to target frame
    const Eigen::Vector2d source_pt = traits::point(source, source_index);
    const Eigen::Vector2d transed_source_pt = T * source_pt;

    // Find nearest neighbor in target
    size_t k_index;
    double k_sq_dist;
    if (!target_tree.nearest(transed_source_pt, &k_index, &k_sq_dist) ||
        rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false;
    }

    // Check distance threshold
    if (std::sqrt(k_sq_dist) > setting.max_correspondence_distance) {
      return false;
    }

    target_index = k_index;

    // Get target point and normal (line direction)
    const Eigen::Vector2d target_pt     = traits::point(target, target_index);
    const Eigen::Vector2d target_normal = traits::normal(target, target_index);

    // Compute point-to-line distance
    // For 2D: distance = dot(point - line_point, line_normal)
    const Eigen::Vector2d residual  = transed_source_pt - target_pt;
    const double          error_val = target_normal.dot(residual);

    // Jacobian computation for SE(2) transformation
    // SE(2) parameterization: [x, y, theta]
    // T = [R(theta) t; 0 1] where t = [x, y]

    // Source point in homogeneous coordinates
    const Eigen::Vector2d &src   = source_pt;
    const double           theta = atan2(T.linear()(1, 0), T.linear()(0, 0));
    const double           cos_theta = std::cos(theta);
    const double           sin_theta = std::sin(theta);

    // Jacobian of transformed point w.r.t. SE(2) parameters
    Eigen::Matrix<double, 2, 3> J_transform;
    J_transform << 1, 0, -sin_theta * src.x() - cos_theta * src.y(), 0, 1,
        cos_theta * src.x() - sin_theta * src.y();

    // Jacobian of error w.r.t. transformed point
    Eigen::Matrix<double, 1, 2> J_error;
    J_error = target_normal.transpose();

    // Chain rule: J = J_error * J_transform
    Eigen::Matrix<double, 1, 3> J = J_error * J_transform;

    // Compute Hessian and gradient
    *H = J.transpose() * J;
    *b = J.transpose() * error_val;
    *e = 0.5 * error_val * error_val;

    return true;
  }

  /// @brief Evaluate error without linearization
  /// @tparam TargetPointCloud Target point cloud type
  /// @tparam SourcePointCloud Source point cloud type
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param T Current 2D transformation
  /// @return Point-to-line error value
  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud &target, const SourcePointCloud &source,
               const Eigen::Isometry2d &T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    // Transform source point
    const Eigen::Vector2d source_pt = traits::point(source, source_index);
    const Eigen::Vector2d transed_source_pt = T * source_pt;

    // Get target point and normal
    const Eigen::Vector2d target_pt     = traits::point(target, target_index);
    const Eigen::Vector2d target_normal = traits::normal(target, target_index);

    // Compute point-to-line distance
    const Eigen::Vector2d residual  = transed_source_pt - target_pt;
    const double          error_val = target_normal.dot(residual);

    return 0.5 * error_val * error_val;
  }

  /// @brief Check if this factor represents a valid correspondence
  /// @return true if correspondence is valid (inlier)
  bool inlier() const {
    return target_index != std::numeric_limits<size_t>::max();
  }

  /// @brief Get the point-to-line distance (signed)
  /// @tparam TargetPointCloud Target point cloud type
  /// @tparam SourcePointCloud Source point cloud type
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param T Current 2D transformation
  /// @return Signed distance from point to line
  template <typename TargetPointCloud, typename SourcePointCloud>
  double signed_distance(const TargetPointCloud  &target,
                         const SourcePointCloud  &source,
                         const Eigen::Isometry2d &T) const {
    if (target_index == std::numeric_limits<size_t>::max()) {
      return 0.0;
    }

    const Eigen::Vector2d source_pt = traits::point(source, source_index);
    const Eigen::Vector2d transed_source_pt = T * source_pt;
    const Eigen::Vector2d target_pt     = traits::point(target, target_index);
    const Eigen::Vector2d target_normal = traits::normal(target, target_index);

    const Eigen::Vector2d residual = transed_source_pt - target_pt;
    return target_normal.dot(residual);
  }

public:
  Setting setting;      ///< Factor configuration
  size_t  target_index; ///< Index of corresponding target point
  size_t  source_index; ///< Index of source point
};

} // namespace icp2d

// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace icp2d {

/// @brief Null factor that gives no constraints for 2D ICP.
/// @note  Used as a placeholder when no additional factors are needed
struct NullFactor {
  NullFactor() = default;

  /// @brief Update linearized system consisting of linearized per-point factors
  /// for 2D
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Nearest neighbor search for the target point cloud
  /// @param T            Linearization point (SE(2) transformation)
  /// @param H            [in/out] Linearized information matrix (3x3)
  /// @param b            [in/out] Linearized information vector (3x1)
  /// @param e            [in/out] Error at the linearization point
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree>
  void update_linearized_system(const TargetPointCloud      &target,
                                const SourcePointCloud      &source,
                                const TargetTree            &target_tree,
                                const Eigen::Isometry2d     &T,
                                Eigen::Matrix<double, 3, 3> *H,
                                Eigen::Matrix<double, 3, 1> *b,
                                double                      *e) const {
    // Do nothing - null factor provides no constraints
    (void)target;
    (void)source;
    (void)target_tree;
    (void)T;
    (void)H;
    (void)b;
    (void)e;
  }

  /// @brief Update error consisting of per-point factors for 2D
  /// @param target   Target point cloud
  /// @param source   Source point cloud
  /// @param T        Evaluation point (SE(2) transformation)
  /// @param e        [in/out] Error at the evaluation point
  template <typename TargetPointCloud, typename SourcePointCloud>
  void update_error(const TargetPointCloud &target,
                    const SourcePointCloud &source, const Eigen::Isometry2d &T,
                    double *e) const {
    // Do nothing - null factor contributes no error
    (void)target;
    (void)source;
    (void)T;
    (void)e;
  }
};

/// @brief Factor to restrict the degrees of freedom of 2D optimization.
/// @note  Enables soft constraints for SE(2): [theta, tx, ty]
///        Can restrict rotation or translation independently
struct RestrictDoFFactor {
  /// @brief Constructor for 2D DoF restriction
  RestrictDoFFactor() {
    lambda = 1e9;   // High regularization for strong constraint
    mask.setOnes(); // All DoF active by default
  }

  /// @brief Set rotation mask (1.0 = active, 0.0 = inactive)
  /// @param rot_active If true, rotation is optimized; if false, rotation is
  /// fixed
  void set_rotation_mask(bool rot_active) { mask(0) = rot_active ? 1.0 : 0.0; }

  /// @brief Set translation mask (1.0 = active, 0.0 = inactive)
  /// @param trans_x_active If true, translation in X is optimized
  /// @param trans_y_active If true, translation in Y is optimized
  void set_translation_mask(bool trans_x_active, bool trans_y_active) {
    mask(1) = trans_x_active ? 1.0 : 0.0;
    mask(2) = trans_y_active ? 1.0 : 0.0;
  }

  /// @brief Set translation mask using Eigen::Array2d
  /// @param trans_mask Translation mask [x, y] (1.0 = active, 0.0 = inactive)
  void set_translation_mask(const Eigen::Array2d &trans_mask) {
    mask.tail<2>() = trans_mask;
  }

  /// @brief Set full DoF mask
  /// @param dof_mask Full mask [theta, tx, ty] (1.0 = active, 0.0 = inactive)
  void set_dof_mask(const Eigen::Array3d &dof_mask) { mask = dof_mask; }

  /// @brief Fix rotation only (translation remains free)
  void fix_rotation() {
    mask(0) = 0.0; // theta = 0
    mask(1) = 1.0; // tx free
    mask(2) = 1.0; // ty free
  }

  /// @brief Fix translation only (rotation remains free)
  void fix_translation() {
    mask(0) = 1.0; // theta free
    mask(1) = 0.0; // tx = 0
    mask(2) = 0.0; // ty = 0
  }

  /// @brief Fix all degrees of freedom (no optimization)
  void fix_all() { mask.setZero(); }

  /// @brief Set regularization strength
  /// @param lambda_val Regularization parameter (higher = stronger constraint)
  void set_lambda(double lambda_val) { lambda = lambda_val; }

  /// @brief Update linearized system with DoF constraints for 2D
  /// @param target       Target point cloud
  /// @param source       Source point cloud
  /// @param target_tree  Target tree for nearest neighbor search
  /// @param T            Current transformation estimate (SE(2))
  /// @param H            [in/out] Hessian matrix (3x3)
  /// @param b            [in/out] Gradient vector (3x1)
  /// @param e            [in/out] Error value
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree>
  void update_linearized_system(const TargetPointCloud      &target,
                                const SourcePointCloud      &source,
                                const TargetTree            &target_tree,
                                const Eigen::Isometry2d     &T,
                                Eigen::Matrix<double, 3, 3> *H,
                                Eigen::Matrix<double, 3, 1> *b,
                                double                      *e) const {
    // Add regularization to constrain inactive DoF
    // For inactive DoF (mask = 0), add large diagonal term to prevent movement
    const Eigen::Array3d constraint_weights = lambda * (1.0 - mask);
    H->diagonal() += constraint_weights.matrix();

    // No additional error contribution
    (void)target;
    (void)source;
    (void)target_tree;
    (void)T;
    (void)b;
    (void)e;
  }

  /// @brief Update error with DoF constraints for 2D
  /// @param target   Target point cloud
  /// @param source   Source point cloud
  /// @param T        Current transformation (SE(2))
  /// @param e        [in/out] Error value
  template <typename TargetPointCloud, typename SourcePointCloud>
  void update_error(const TargetPointCloud &target,
                    const SourcePointCloud &source, const Eigen::Isometry2d &T,
                    double *e) const {
    // For error evaluation, we can add penalty for violating constraints
    // Extract SE(2) parameters: [theta, tx, ty]
    const double theta = std::atan2(T.linear()(1, 0), T.linear()(0, 0));
    const Eigen::Vector2d translation = T.translation();

    // Penalty for constrained DoF
    double constraint_error = 0.0;
    if (mask(0) < 0.5) { // Rotation constrained
      constraint_error += 0.5 * lambda * theta * theta;
    }
    if (mask(1) < 0.5) { // X translation constrained
      constraint_error += 0.5 * lambda * translation.x() * translation.x();
    }
    if (mask(2) < 0.5) { // Y translation constrained
      constraint_error += 0.5 * lambda * translation.y() * translation.y();
    }

    *e += constraint_error;

    // Suppress unused parameter warnings
    (void)target;
    (void)source;
  }

public:
  double lambda; ///< Regularization parameter (higher = stronger constraint)
  Eigen::Array3d mask; ///< DoF mask for SE(2): [theta, tx, ty] (1.0 = active,
                       ///< 0.0 = inactive)
};

/// @brief Default correspondence rejector (accepts all)
struct DefaultRejector {
  template <typename TargetPointCloud, typename SourcePointCloud>
  bool operator()(const TargetPointCloud &target,
                  const SourcePointCloud &source, const Eigen::Isometry2d &T,
                  size_t target_index, size_t source_index,
                  double sq_distance) const {
    // Accept all correspondences
    (void)target;
    (void)source;
    (void)T;
    (void)target_index;
    (void)source_index;
    (void)sq_distance;
    return false;
  }
};

} // namespace icp2d

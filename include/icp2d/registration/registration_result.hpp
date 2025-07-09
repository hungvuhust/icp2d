#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace icp2d {

/// @brief Registration result for 2D ICP
struct RegistrationResult {
  RegistrationResult(const Eigen::Isometry2d &T = Eigen::Isometry2d::Identity())
      : T_target_source(T), converged(false), iterations(0), num_inliers(0),
        H(Eigen::Matrix<double, 3, 3>::Zero()),
        b(Eigen::Matrix<double, 3, 1>::Zero()), error(0.0) {}

  Eigen::Isometry2d T_target_source; ///<  Estimated transformation

  bool   converged;   ///< If the optimization converged
  size_t iterations;  ///< Number of optimization iterations
  size_t num_inliers; ///< Number of inliear points

  Eigen::Matrix<double, 3, 3> H; ///< Final information matrix (3x3 for SE(2))
  Eigen::Matrix<double, 3, 1> b; ///< Final information vector (3x1 for SE(2))
  double                      error; ///< Final error
};

} // namespace icp2d

#pragma once

#include <Eigen/Core>

namespace icp2d {

/// @brief Registration termination criteria
struct TerminationCriteria {
  /// @brief Constructor
  TerminationCriteria()
      : translation_eps(1e-3), rotation_eps(0.1 * M_PI / 180.0) {}

  /// @brief Check the convergence
  /// @param delta  Transformation update vector
  /// @return       True if converged
  bool converged(const Eigen::Matrix<double, 3, 1> &delta) const {
    return delta.template head<1>().norm() <= translation_eps &&
           delta.template tail<2>().norm() <= rotation_eps;
  }

  double translation_eps; ///< Translation tolerance [m]
  double rotation_eps;    ///< Rotation tolerance [rad]
};

} // namespace icp2d

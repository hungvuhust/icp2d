#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace icp2d {
/// @brief Create skew symmetric matrix for 2D rotation
/// @param theta  Rotation angle
/// @return       2x2 skew symmetric matrix
inline Eigen::Matrix2d skew(double theta) {
  Eigen::Matrix2d skew;
  skew << 0, -1, 1, 0;
  return theta * skew;
}

/*
 * SO2/SE2 expmap code adapted from Sophus SO3 implementation
 * https://github.com/strasdat/Sophus/blob/593db47500ea1a2de5f0e6579c86147991509c59/sophus/so3.hpp#L585
 *
 * Copyright 2011-2017 Hauke Strasdat
 *           2012-2017 Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights  to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// @brief SO2 expmap.
/// @param theta  Rotation angle
/// @return       2x2 rotation matrix
inline Eigen::Matrix2d so2_exp(double theta) {
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  Eigen::Matrix2d R;
  R << cos_theta, -sin_theta, sin_theta, cos_theta;

  return R;
}

/// @brief SE2 expmap.
/// @param xi  Twist vector [tx, ty, theta]
/// @return    SE2 matrix (3x3 homogeneous transformation)
inline Eigen::Matrix3d se2_exp(const Eigen::Vector3d &xi) {
  const Eigen::Vector2d rho   = xi.head<2>(); // translation part
  const double          theta = xi(2);        // rotation part

  Eigen::Matrix3d se2 = Eigen::Matrix3d::Identity();

  // Rotation part
  se2.block<2, 2>(0, 0) = so2_exp(theta);

  // Translation part
  if (std::abs(theta) < 1e-10) {
    // For small angles, V ≈ I
    se2.block<2, 1>(0, 2) = rho;
  } else {
    // V = (sin(theta)/theta) * I + ((1-cos(theta))/theta) * skew(1)
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);

    Eigen::Matrix2d V;
    V << sin_theta / theta, -(1 - cos_theta) / theta, (1 - cos_theta) / theta,
        sin_theta / theta;

    se2.block<2, 1>(0, 2) = V * rho;
  }

  return se2;
}

/// @brief Convert SE2 matrix to Isometry2d
/// @param se2_matrix  3x3 SE2 homogeneous matrix
/// @return            Eigen::Isometry2d transform
inline Eigen::Isometry2d se2_to_isometry(const Eigen::Matrix3d &se2_matrix) {
  Eigen::Isometry2d transform = Eigen::Isometry2d::Identity();
  transform.linear()          = se2_matrix.block<2, 2>(0, 0);
  transform.translation()     = se2_matrix.block<2, 1>(0, 2);
  return transform;
}

/// @brief SE2 expmap returning Isometry2d
/// @param xi  Twist vector [tx, ty, theta]
/// @return    Eigen::Isometry2d transform
inline Eigen::Isometry2d se2_exp_isometry(const Eigen::Vector3d &xi) {
  return se2_to_isometry(se2_exp(xi));
}

} // namespace icp2d
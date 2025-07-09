#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace icp2d {

/// @brief Kết quả của quá trình đăng ký
struct RegistrationResult {
  /// @brief Constructor
  /// @param T Initial transformation
  explicit RegistrationResult(const Eigen::Isometry2d &T)
      : T_target_source(T), converged(false), iterations(0), error(0.0) {}

  Eigen::Isometry2d T_target_source; ///< Phép biến đổi từ source sang target
  bool              converged;       ///< Đã hội tụ chưa
  int               iterations;      ///< Số lần lặp đã thực hiện
  double            error;           ///< Giá trị error cuối cùng
  Eigen::Matrix<double, 3, 3> H;           ///< Ma trận Hessian cuối cùng
  Eigen::Matrix<double, 3, 1> b;           ///< Vector gradient cuối cùng
  int                         num_inliers; ///< Số lượng điểm inlier
};

} // namespace icp2d

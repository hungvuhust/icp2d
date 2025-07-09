#pragma once

#include <Eigen/Core>

namespace icp2d {

/// @brief Tiêu chí dừng cho quá trình tối ưu hóa
struct TerminationCriteria {
  /// @brief Kiểm tra xem quá trình tối ưu hóa đã hội tụ chưa
  /// @param delta Sự thay đổi của tham số trong lần lặp cuối
  /// @param error Giá trị error hiện tại
  /// @return true nếu đã hội tụ
  bool converged(const Eigen::Matrix<double, 3, 1> &delta, double error) const {
    // Check absolute convergence criteria
    const bool rot_converged   = delta.head<1>().norm() < eps_rot;
    const bool trans_converged = delta.tail<2>().norm() < eps_trans;
    const bool error_converged = error < eps_error;

    // Check relative error improvement
    const bool error_improved =
        prev_error > 0.0 &&
        std::abs(error - prev_error) / prev_error < eps_rel_error;

    // Update previous error
    prev_error = error;

    return (rot_converged && trans_converged) || error_converged ||
           error_improved;
  }

  int    max_iterations{50}; ///< Số lần lặp tối đa
  double eps_rot{1e-4};      ///< Ngưỡng hội tụ cho góc quay
  double eps_trans{1e-4};    ///< Ngưỡng hội tụ cho dịch chuyển
  double eps_error{1e-3};    ///< Ngưỡng hội tụ cho error
  double eps_rel_error{
      1e-3}; ///< Ngưỡng hội tụ cho sự cải thiện error tương đối
  mutable double prev_error{-1.0}; ///< Error từ lần kiểm tra trước
};

} // namespace icp2d

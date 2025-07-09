
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <icp2d/core/traits.hpp>
#include <icp2d/util/lie.hpp>
#include <limits>
#include <memory>
#include <vector>

namespace icp2d {

/// @brief Point-to-line per-point error factor for 2D ICP.
/// @note  Computes the perpendicular distance from a point to a line in 2D
struct PointToLineICPFactor {
  struct Setting {
    double weight;    ///< Weight for this factor
    double step_size; ///< Step size adjustment coefficient

    Setting(double weight = 1.0, double step_size = 0.5)
        : weight(weight), step_size(step_size) {}
  };

  PointToLineICPFactor(const Setting &setting = Setting())
      : weight(setting.weight), step_size(setting.step_size),
        target_index(std::numeric_limits<size_t>::max()),
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
                 size_t source_idx, const CorrespondenceRejector &rejector,
                 Eigen::Matrix<double, 3, 3> *H, Eigen::Matrix<double, 3, 1> *b,
                 double *e) {
    // Lưu source_index
    source_index = source_idx;
    target_index = std::numeric_limits<size_t>::max();

    // Lấy điểm nguồn và biến đổi nó
    const Eigen::Vector2d source_pt      = traits::point(source, source_index);
    const Eigen::Vector2d transformed_pt = T * source_pt;

    // Tìm điểm gần nhất trong target
    size_t k_index;
    double k_sq_dist;
    if (!target_tree.nearest(transformed_pt, &k_index, &k_sq_dist)) {
      return false;
    }

    // Kiểm tra xem cặp điểm có bị loại bỏ không
    if (rejector(target, source, T, k_index, source_index, k_sq_dist)) {
      return false;
    }

    target_index = k_index;

    // Lấy điểm target và normal vector
    const Eigen::Vector2d target_pt     = traits::point(target, target_index);
    const Eigen::Vector2d target_normal = traits::normal(target, target_index);

    // Tính error vector (khoảng cách từ điểm đến đường)
    const Eigen::Vector2d residual  = transformed_pt - target_pt;
    const double          error_val = target_normal.dot(residual);

    // Scale error based on point distance from origin
    const double point_scale  = 1.0 / (1.0 + 0.1 * source_pt.norm());
    const double scaled_error = point_scale * error_val;

    // Apply Huber kernel
    const double huber_threshold = 0.3;
    double       huber_weight    = 1.0;
    if (std::abs(scaled_error) > huber_threshold) {
      huber_weight = huber_threshold / std::abs(scaled_error);
    }

    *e = 0.5 * weight * huber_weight * scaled_error * scaled_error;

    // Tính Jacobian
    // J = [dx/dθ dx/dtx dx/dty]
    // với θ là góc xoay, tx và ty là dịch chuyển
    Eigen::Matrix<double, 1, 3> J;

    // Đạo hàm theo θ: normal^T * [-y x] với góc quay hiện tại
    const double cos_theta = T.linear()(0, 0);
    const double sin_theta = T.linear()(1, 0);

    // Scale rotation Jacobian based on point distance
    const double rot_scale = point_scale * step_size * huber_weight;
    J(0) =
        rot_scale * target_normal.dot(Eigen::Vector2d(
                        -sin_theta * source_pt.x() - cos_theta * source_pt.y(),
                        cos_theta * source_pt.x() - sin_theta * source_pt.y()));

    // Scale translation Jacobian
    const double trans_scale = point_scale * step_size * huber_weight;
    J(1)                     = trans_scale * target_normal.x();
    J(2)                     = trans_scale * target_normal.y();

    // Cập nhật Hessian và gradient
    *H = weight * (J.transpose() * J);
    *b = -weight * (J.transpose() * scaled_error);

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
    const Eigen::Vector2d source_pt      = traits::point(source, source_index);
    const Eigen::Vector2d transformed_pt = T * source_pt;

    // Get target point and normal
    const Eigen::Vector2d target_pt     = traits::point(target, target_index);
    const Eigen::Vector2d target_normal = traits::normal(target, target_index);

    // Compute point-to-line distance
    const Eigen::Vector2d error_residual = transformed_pt - target_pt;
    const double          error_distance = target_normal.dot(error_residual);

    // Scale error based on point distance from origin
    const double point_scale  = 1.0 / (1.0 + 0.1 * source_pt.norm());
    const double scaled_error = point_scale * error_distance;

    // Apply Huber kernel
    const double huber_threshold = 0.3;
    double       huber_weight    = 1.0;
    if (std::abs(scaled_error) > huber_threshold) {
      huber_weight = huber_threshold / std::abs(scaled_error);
    }

    return 0.5 * weight * huber_weight * scaled_error * scaled_error;
  }

  /// @brief Check if this factor represents a valid correspondence
  /// @return true if correspondence is valid (inlier)
  bool inlier() const {
    return target_index != std::numeric_limits<size_t>::max();
  }

  /// @brief Get the step size
  /// @return Step size
  double get_step_size() const { return step_size; }

public:
  double weight;       ///< Weight for this factor
  double step_size;    ///< Step size adjustment coefficient
  size_t target_index; ///< Index of corresponding target point
  size_t source_index; ///< Index of source point
};

} // namespace icp2d

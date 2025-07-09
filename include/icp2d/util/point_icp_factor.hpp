
#pragma once

#include "icp2d/core/traits.hpp"
#include "icp2d/registration/rejector.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace icp2d {

/// @brief Point-to-point ICP factor cho 2D
/// @note Tính error dựa trên khoảng cách Euclidean giữa các điểm tương ứng
struct PointICPFactor {
  /// @brief Cài đặt cho point-to-point ICP factor
  struct Setting {
    double weight;    ///< Trọng số cho factor này
    double step_size; ///< Hệ số điều chỉnh bước

    Setting(double weight = 1.0, double step_size = 0.5)
        : weight(weight), step_size(step_size) {}
  };

  /// @brief Constructor
  /// @param setting Cài đặt cho factor
  PointICPFactor(const Setting &setting = Setting())
      : weight(setting.weight), step_size(setting.step_size) {}

  /// @brief Tuyến tính hóa factor cho một cặp điểm tương ứng
  /// @tparam TargetPointCloud Kiểu point cloud đích
  /// @tparam SourcePointCloud Kiểu point cloud nguồn
  /// @tparam TargetTree Kiểu cây tìm kiếm lân cận (KDTree)
  /// @tparam CorrespondenceRejector Kiểu bộ loại bỏ điểm tương ứng
  /// @param target Point cloud đích
  /// @param source Point cloud nguồn
  /// @param target_tree Cây tìm kiếm lân cận cho point cloud đích
  /// @param T Phép biến đổi hiện tại (SE(2))
  /// @param source_index Chỉ số của điểm nguồn
  /// @param rejector Bộ loại bỏ điểm tương ứng
  /// @param H [out] Ma trận Hessian (3x3)
  /// @param b [out] Vector gradient (3x1)
  /// @param e [out] Giá trị error
  /// @return true nếu tuyến tính hóa thành công
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector>
  bool linearize(const TargetPointCloud &target, const SourcePointCloud &source,
                 const TargetTree &target_tree, const Eigen::Isometry2d &T,
                 size_t source_idx, const CorrespondenceRejector &rejector,
                 Eigen::Matrix<double, 3, 3> *H, Eigen::Matrix<double, 3, 1> *b,
                 double *e) {
    // Lưu source_index
    source_index = source_idx;

    // Lấy điểm nguồn và biến đổi nó
    const Eigen::Vector2d source_pt      = traits::point(source, source_index);
    const Eigen::Vector2d transformed_pt = T * source_pt;

    // Tìm điểm gần nhất trong target
    size_t target_index;
    double sq_distance;
    if (!target_tree.nearest(transformed_pt, &target_index, &sq_distance)) {
      return false;
    }

    // Kiểm tra xem cặp điểm có bị loại bỏ không
    if (rejector(target, source, T, target_index, source_index, sq_distance)) {
      return false;
    }

    // Lấy điểm target tương ứng
    const Eigen::Vector2d target_pt = traits::point(target, target_index);

    // Tính error vector (khoảng cách giữa điểm transformed và target)
    const Eigen::Vector2d error = transformed_pt - target_pt;

    // Scale error based on point distance from origin to balance rotation and
    // translation
    const double point_scale =
        1.0 / (1.0 + 0.1 * source_pt.norm()); // Reduce the scaling effect
    const Eigen::Vector2d scaled_error = point_scale * error;

    // Apply Huber kernel with larger threshold
    const double huber_threshold =
        0.3; // Increased from 0.1 to be less aggressive
    const double error_norm   = scaled_error.norm();
    double       huber_weight = 1.0;
    if (error_norm > huber_threshold) {
      huber_weight = huber_threshold / error_norm;
    }

    *e = 0.5 * weight * huber_weight * scaled_error.squaredNorm();

    // Tính Jacobian
    // J = [dx/dθ dx/dtx dx/dty]
    // với θ là góc xoay, tx và ty là dịch chuyển
    Eigen::Matrix<double, 2, 3> J;

    // Đạo hàm theo θ: [-y x] với góc quay hiện tại
    const double cos_theta = T.linear()(0, 0);
    const double sin_theta = T.linear()(1, 0);

    // Scale rotation Jacobian based on point distance
    const double rot_scale = point_scale * step_size * huber_weight;
    J.col(0) << rot_scale *
                    (-sin_theta * source_pt.x() - cos_theta * source_pt.y()),
        rot_scale * (cos_theta * source_pt.x() - sin_theta * source_pt.y());

    // Scale translation Jacobian
    const double trans_scale = point_scale * step_size * huber_weight;
    J.col(1)                 = trans_scale * Eigen::Vector2d::UnitX();
    J.col(2)                 = trans_scale * Eigen::Vector2d::UnitY();

    // Cập nhật Hessian và gradient
    *H = weight * (J.transpose() * J);
    *b = -weight * (J.transpose() * scaled_error);

    return true;
  }

  /// @brief Tính error cho một cặp điểm tương ứng
  /// @tparam TargetPointCloud Kiểu point cloud đích
  /// @tparam SourcePointCloud Kiểu point cloud nguồn
  /// @tparam TargetTree Kiểu cây tìm kiếm lân cận
  /// @param target Point cloud đích
  /// @param source Point cloud nguồn
  /// @param target_tree Cây tìm kiếm lân cận cho point cloud đích
  /// @param T Phép biến đổi hiện tại (SE(2))
  /// @return Giá trị error (khoảng cách bình phương có trọng số)
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree>
  double error(const TargetPointCloud &target, const SourcePointCloud &source,
               const TargetTree        &target_tree,
               const Eigen::Isometry2d &T) const {
    // Lấy điểm nguồn và biến đổi nó
    const Eigen::Vector2d source_pt      = traits::point(source, source_index);
    const Eigen::Vector2d transformed_pt = T * source_pt;

    // Tìm điểm gần nhất trong target
    size_t target_index;
    double sq_distance;
    if (!target_tree.nearest(transformed_pt, &target_index, &sq_distance)) {
      return 0.0;
    }

    // Lấy điểm target tương ứng
    const Eigen::Vector2d target_pt = traits::point(target, target_index);

    // Tính error vector (khoảng cách giữa điểm transformed và target)
    const Eigen::Vector2d error = transformed_pt - target_pt;

    // Scale error based on point distance from origin to match linearize()
    const double point_scale =
        1.0 /
        (1.0 + 0.1 * source_pt.norm()); // Match the scaling in linearize()
    const Eigen::Vector2d scaled_error = point_scale * error;

    // Apply Huber kernel
    const double huber_threshold = 0.3; // Match threshold in linearize()
    const double error_norm      = scaled_error.norm();
    double       huber_weight    = 1.0;
    if (error_norm > huber_threshold) {
      huber_weight = huber_threshold / error_norm;
    }

    return 0.5 * weight * huber_weight * scaled_error.squaredNorm();
  }

  /// @brief Kiểm tra xem factor có phải là inlier không
  /// @return true nếu là inlier
  bool inlier() const { return true; }

  /// @brief Lấy hệ số điều chỉnh bước
  /// @return Hệ số điều chỉnh bước
  double get_step_size() const { return step_size; }

public:
  double weight;       ///< Trọng số cho factor này
  double step_size;    ///< Hệ số điều chỉnh bước
  size_t source_index; ///< Chỉ số của điểm nguồn
};

} // namespace icp2d

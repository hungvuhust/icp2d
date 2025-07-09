#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace icp2d {

/// @brief Reduction class để tính toán error và Jacobian cho tất cả các điểm
struct PointFactorReduction {
  /// @brief Tuyến tính hóa tất cả các factor
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param target_tree Target KDTree
  /// @param rejector Correspondence rejector
  /// @param T Current transformation
  /// @param factors Vector of point factors
  /// @return Tuple of (H, b, error)
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector,
            typename Factor>
  std::tuple<Eigen::Matrix<double, 3, 3>, Eigen::Matrix<double, 3, 1>, double>
  linearize(const TargetPointCloud &target, const SourcePointCloud &source,
            const TargetTree             &target_tree,
            const CorrespondenceRejector &rejector, const Eigen::Isometry2d &T,
            std::vector<Factor> &factors) const {
    Eigen::Matrix<double, 3, 3> H     = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 1> b     = Eigen::Matrix<double, 3, 1>::Zero();
    double                      error = 0.0;

    // Linearize each factor
    for (size_t i = 0; i < factors.size(); ++i) {
      Eigen::Matrix<double, 3, 3> Hi;
      Eigen::Matrix<double, 3, 1> bi;
      double                      ei;

      if (factors[i].linearize(target, source, target_tree, T, i, rejector, &Hi,
                               &bi, &ei)) {
        H += Hi;
        b += bi;
        error += ei;
      }
    }

    return {H, b, error};
  }

  /// @brief Tính tổng error cho tất cả các factor
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param target_tree Target KDTree
  /// @param T Current transformation
  /// @param factors Vector of point factors
  /// @return Total error
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename Factor>
  double error(const TargetPointCloud &target, const SourcePointCloud &source,
               const TargetTree &target_tree, const Eigen::Isometry2d &T,
               const std::vector<Factor> &factors) const {
    double total_error = 0.0;
    for (const auto &factor : factors) {
      total_error += factor.error(target, source, target_tree, T);
    }
    return total_error;
  }
};

} // namespace icp2d
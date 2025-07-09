#pragma once

#include "icp2d/core/traits.hpp"
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <array>

namespace icp2d {

/// @brief Parameters to control the projection axis search.
struct ProjectionSetting {
  size_t max_scan_count =
      128; ///< Maximum number of points to use for the axis search.
};

/// @brief Axis-aligned projection for 2D (selecting X or Y axis with the
/// largest variance).
/// @note  Up to max_scan_count samples are used to estimate the variance.
struct AxisAlignedProjection {
public:
  /// @brief Project the point to the selected axis.
  /// @param pt  Point to project
  /// @return    Projected value
  double operator()(const Eigen::Vector2d &pt) const { return pt[axis]; }

  /// @brief Find the axis with the largest variance.
  /// @param points     Point cloud
  /// @param first      First point index iterator
  /// @param last       Last point index iterator
  /// @param setting    Search setting
  /// @return           Projection with the largest variance axis
  template <typename PointCloud, typename IndexConstIterator>
  static AxisAlignedProjection
  find_axis(const PointCloud &points, IndexConstIterator first,
            IndexConstIterator last, const ProjectionSetting &setting) {
    const size_t    N      = std::distance(first, last);
    Eigen::Vector2d sum_pt = Eigen::Vector2d::Zero();
    Eigen::Vector2d sum_sq = Eigen::Vector2d::Zero();

    const size_t step =
        N < setting.max_scan_count ? 1 : N / setting.max_scan_count;
    const size_t num_steps = N / step;
    for (size_t i = 0; i < num_steps; i++) {
      const auto            itr = first + step * i;
      const Eigen::Vector2d pt  = traits::point(points, *itr);
      sum_pt += pt;
      sum_sq += pt.cwiseProduct(pt);
    }

    const Eigen::Vector2d mean = sum_pt / num_steps;
    const Eigen::Vector2d var  = (sum_sq / num_steps) - mean.cwiseProduct(mean);

    return AxisAlignedProjection{var[0] > var[1] ? 0 : 1};
  }

public:
  int axis; ///< Axis index (0: X, 1: Y)
};

/// @brief Normal projection for 2D (selecting the direction with the largest
/// variance).
/// @note  Up to max_scan_count samples are used to estimate the variance along
/// the axis.
struct NormalProjection {
public:
  /// @brief Project the point to the normal direction.
  /// @param pt   Point to project
  /// @return     Projected value
  double operator()(const Eigen::Vector2d &pt) const { return normal.dot(pt); }

  /// @brief  Find the direction with the largest variance.
  /// @param points   Point cloud
  /// @param first    First point index iterator
  /// @param last     Last point index iterator
  /// @param setting  Search setting
  /// @return         Projection with the largest variance direction
  template <typename PointCloud, typename IndexConstIterator>
  static NormalProjection
  find_axis(const PointCloud &points, IndexConstIterator first,
            IndexConstIterator last, const ProjectionSetting &setting) {
    const size_t    N      = std::distance(first, last);
    Eigen::Vector2d sum_pt = Eigen::Vector2d::Zero();
    Eigen::Matrix2d sum_sq = Eigen::Matrix2d::Zero();

    const size_t step =
        N < setting.max_scan_count ? 1 : N / setting.max_scan_count;
    const size_t num_steps = N / step;
    for (size_t i = 0; i < num_steps; i++) {
      const auto            itr = first + step * i;
      const Eigen::Vector2d pt  = traits::point(points, *itr);
      sum_pt += pt;
      sum_sq += pt * pt.transpose();
    }

    const Eigen::Vector2d mean = sum_pt / num_steps;
    const Eigen::Matrix2d cov  = (sum_sq / num_steps) - mean * mean.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig;
    eig.computeDirect(cov);

    NormalProjection p;
    // Lấy eigenvector tương ứng với eigenvalue lớn nhất (cột cuối cùng)
    p.normal = eig.eigenvectors().col(1);
    return p;
  }

public:
  Eigen::Vector2d normal; ///< Projection direction
};

} // namespace icp2d

#pragma once

#include <Eigen/Core>
#include <vector>

namespace icp2d {
namespace traits {

/// @brief Traits interface cho point cloud
template <typename T> struct Traits;

/// @brief Traits specialization cho vector<Eigen::Vector2d>
template <> struct Traits<std::vector<Eigen::Vector2d>> {
  using PointType = Eigen::Vector2d;

  static size_t size(const std::vector<Eigen::Vector2d> &points) {
    return points.size();
  }

  static const Eigen::Vector2d &
  point(const std::vector<Eigen::Vector2d> &points, size_t i) {
    return points[i];
  }

  static bool has_normals(const std::vector<Eigen::Vector2d> &points) {
    (void)points;
    return false;
  }

  static bool has_covs(const std::vector<Eigen::Vector2d> &points) {
    (void)points;
    return false;
  }

  static Eigen::Vector2d normal(const std::vector<Eigen::Vector2d> &points,
                                size_t                              i) {
    (void)points;
    (void)i;
    throw std::runtime_error("Vector<Vector2d> does not have normals");
  }

  static Eigen::Matrix2d cov(const std::vector<Eigen::Vector2d> &points,
                             size_t                              i) {
    (void)points;
    (void)i;
    throw std::runtime_error("Vector<Vector2d> does not have covariances");
  }
};

/// @brief Helper function để lấy kích thước của point cloud
template <typename T> size_t size(const T &points) {
  return Traits<T>::size(points);
}

/// @brief Helper function để lấy điểm thứ i từ point cloud
template <typename T> auto point(const T &points, size_t i) {
  return Traits<T>::point(points, i);
}

/// @brief Helper function để kiểm tra có normal hay không
template <typename T> bool has_normals(const T &points) {
  return Traits<T>::has_normals(points);
}

/// @brief Helper function để kiểm tra có covariance hay không
template <typename T> bool has_covs(const T &points) {
  return Traits<T>::has_covs(points);
}

/// @brief Helper function để lấy normal thứ i từ point cloud
template <typename T> auto normal(const T &points, size_t i) {
  return Traits<T>::normal(points, i);
}

/// @brief Helper function để lấy covariance thứ i từ point cloud
template <typename T> auto cov(const T &points, size_t i) {
  return Traits<T>::cov(points, i);
}

} // namespace traits
} // namespace icp2d

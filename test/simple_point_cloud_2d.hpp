#pragma once

#include <Eigen/Core>
#include <icp2d/core/traits.hpp>
#include <vector>

// Simple PointCloud implementation for testing
class SimplePointCloud2D {
public:
  std::vector<Eigen::Vector2d> points;

  SimplePointCloud2D() = default;

  void addPoint(double x, double y) { points.emplace_back(x, y); }

  void addPoint(const Eigen::Vector2d &pt) { points.push_back(pt); }

  size_t size() const { return points.size(); }
};

// Traits specialization for SimplePointCloud2D
namespace icp2d {
namespace traits {

template <> struct Traits<SimplePointCloud2D> {
  using PointType = Eigen::Vector2d;

  static size_t size(const SimplePointCloud2D &cloud) {
    return cloud.points.size();
  }

  static const Eigen::Vector2d &point(const SimplePointCloud2D &cloud,
                                      size_t                    i) {
    return cloud.points[i];
  }

  static bool has_normals(const SimplePointCloud2D &) { return false; }

  static bool has_covs(const SimplePointCloud2D &) { return false; }

  static Eigen::Vector2d normal(const SimplePointCloud2D &, size_t) {
    throw std::runtime_error("SimplePointCloud2D does not have normals");
  }

  static Eigen::Matrix2d cov(const SimplePointCloud2D &, size_t) {
    throw std::runtime_error("SimplePointCloud2D does not have covariances");
  }
};

} // namespace traits
} // namespace icp2d
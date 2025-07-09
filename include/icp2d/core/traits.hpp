
#pragma once

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <memory>
#include <vector>

// Forward declare KdTree to avoid circular dependency
namespace icp2d {
struct AxisAlignedProjection;
template <typename PointCloud, typename Projection> class KdTree;
} // namespace icp2d

namespace icp2d {
namespace traits {

/// @brief Point cloud traits for accessing points and normals
template <typename PointCloud> struct PointCloudTraits {
  static size_t size(const PointCloud &cloud) { return cloud.size(); }

  static bool has_points(const PointCloud &) { return true; }

  static bool has_normals(const PointCloud &) { return false; }

  static Eigen::Vector2d point(const PointCloud &cloud, size_t index) {
    return cloud[index];
  }

  static Eigen::Vector2d normal(const PointCloud &cloud, size_t index) {
    throw std::runtime_error("Normals not available for this point cloud type");
  }
};

// Template specialization for std::vector<Eigen::Vector2d>
template <> struct PointCloudTraits<std::vector<Eigen::Vector2d>> {
  static size_t size(const std::vector<Eigen::Vector2d> &cloud) {
    return cloud.size();
  }

  static bool has_points(const std::vector<Eigen::Vector2d> &) { return true; }

  static bool has_normals(const std::vector<Eigen::Vector2d> &) {
    return false;
  }

  static Eigen::Vector2d point(const std::vector<Eigen::Vector2d> &cloud,
                               size_t                              index) {
    return cloud[index];
  }

  static Eigen::Vector2d normal(const std::vector<Eigen::Vector2d> &cloud,
                                size_t                              index) {
    throw std::runtime_error(
        "Normals not available for std::vector<Eigen::Vector2d>");
  }
};

// Convenience functions that use the traits
template <typename PointCloud> size_t size(const PointCloud &cloud) {
  return PointCloudTraits<PointCloud>::size(cloud);
}

template <typename PointCloud> bool has_points(const PointCloud &cloud) {
  return PointCloudTraits<PointCloud>::has_points(cloud);
}

template <typename PointCloud> bool has_normals(const PointCloud &cloud) {
  return PointCloudTraits<PointCloud>::has_normals(cloud);
}

template <typename PointCloud>
Eigen::Vector2d point(const PointCloud &cloud, size_t index) {
  return PointCloudTraits<PointCloud>::point(cloud, index);
}

template <typename PointCloud>
Eigen::Vector2d normal(const PointCloud &cloud, size_t index) {
  return PointCloudTraits<PointCloud>::normal(cloud, index);
}

} // namespace traits
} // namespace icp2d

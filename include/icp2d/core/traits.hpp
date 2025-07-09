#pragma once

#include <Eigen/Core>
#include <type_traits>

namespace icp2d {

namespace traits {

template <typename T> struct Traits;

/// @brief Find k-nearest neighbors for 2D points.
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point (2D)
/// @param k          Number of neighbors
/// @param k_indices  [out] Indices of k-nearest neighbors
/// @param k_sq_dists [out] Squared distances to k-nearest neighbors
/// @return Number of found neighbors
template <typename T>
size_t knn_search(const T &tree, const Eigen::Vector2d &point, size_t k,
                  size_t *k_indices, double *k_sq_dists) {
  return Traits<T>::knn_search(tree, point, k, k_indices, k_sq_dists);
}

/// @brief Check if T has nearest_neighbor_search method.
template <typename T> struct has_nearest_neighbor_search {
  template <typename U, int = (&Traits<U>::nearest_neighbor_search, 0)>
  static std::true_type  test(U *);
  static std::false_type test(...);

  static constexpr bool value = decltype(test((T *)nullptr))::value;
};

/// @brief Find the nearest neighbor for 2D point. If
/// Traits<T>::nearest_neighbor_search is not defined, fallback to knn_search
/// with k=1.
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point (2D)
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @return 1 if a neighbor is found else 0
template <typename T,
          std::enable_if_t<has_nearest_neighbor_search<T>::value, bool> = true>
size_t nearest_neighbor_search(const T &tree, const Eigen::Vector2d &point,
                               size_t *k_index, double *k_sq_dist) {
  return Traits<T>::nearest_neighbor_search(tree, point, k_index, k_sq_dist);
}

/// @brief Find the nearest neighbor for 2D point. If
/// Traits<T>::nearest_neighbor_search is not defined, fallback to knn_search
/// with k=1.
/// @param tree       Nearest neighbor search (e.g., KdTree)
/// @param point      Query point (2D)
/// @param k_index    [out] Index of the nearest neighbor
/// @param k_sq_dist  [out] Squared distance to the nearest neighbor
/// @return 1 if a neighbor is found else 0
template <typename T,
          std::enable_if_t<!has_nearest_neighbor_search<T>::value, bool> = true>
size_t nearest_neighbor_search(const T &tree, const Eigen::Vector2d &point,
                               size_t *k_index, double *k_sq_dist) {
  return Traits<T>::knn_search(tree, point, 1, k_index, k_sq_dist);
}

template <typename T> struct Traits;

/// @brief  Get the number of points.
template <typename T> size_t size(const T &points) {
  return Traits<T>::size(points);
}

/// @brief Check if the point cloud has points.
template <typename T> bool has_points(const T &points) {
  return Traits<T>::has_points(points);
}

/// @brief Check if the point cloud has normals.
template <typename T> bool has_normals(const T &points) {
  return Traits<T>::has_normals(points);
}

/// @brief Check if the point cloud has covariances.
template <typename T> bool has_covs(const T &points) {
  return Traits<T>::has_covs(points);
}

/// @brief Get i-th point. 2D vector is used for 2D ICP problem.
template <typename T> auto point(const T &points, size_t i) {
  return Traits<T>::point(points, i);
}

/// @brief Get i-th normal. 2D vector is used for 2D ICP problem.
template <typename T> auto normal(const T &points, size_t i) {
  return Traits<T>::normal(points, i);
}

/// @brief Get i-th covariance. 2x2 matrix is used for 2D ICP problem.
template <typename T> auto cov(const T &points, size_t i) {
  return Traits<T>::cov(points, i);
}

/// @brief Resize the point cloud (this function should resize all attributes)
template <typename T> void resize(T &points, size_t size) {
  return Traits<T>::resize(points, size);
}

} // namespace traits

} // namespace icp2d

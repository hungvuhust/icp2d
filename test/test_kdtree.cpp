#include <cassert>
#include <cmath>
#include <icp2d/core/kdtree.hpp>
#include <iostream>
#include <random>
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
  static size_t size(const SimplePointCloud2D &cloud) {
    return cloud.points.size();
  }

  static bool has_points(const SimplePointCloud2D &) { return true; }

  static bool has_normals(const SimplePointCloud2D &) { return false; }

  static bool has_covs(const SimplePointCloud2D &) { return false; }

  static Eigen::Vector2d point(const SimplePointCloud2D &cloud, size_t i) {
    return cloud.points[i];
  }

  static void resize(SimplePointCloud2D &cloud, size_t new_size) {
    cloud.points.resize(new_size);
  }
};

} // namespace traits
} // namespace icp2d

// Utility function to calculate distance
double distance2D(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
  return (a - b).norm();
}

// Test 1: Basic functionality with simple points
void test_basic_functionality() {
  std::cout << "Test 1: Basic functionality" << std::endl;

  SimplePointCloud2D cloud;
  cloud.addPoint(0.0, 0.0);
  cloud.addPoint(1.0, 0.0);
  cloud.addPoint(0.0, 1.0);
  cloud.addPoint(1.0, 1.0);
  cloud.addPoint(0.5, 0.5);

  icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud);

  // Test nearest neighbor search
  Eigen::Vector2d query(0.1, 0.1);
  size_t          nearest_idx;
  double          nearest_dist;

  size_t found =
      kdtree.nearest_neighbor_search(query, &nearest_idx, &nearest_dist);

  assert(found == 1);
  assert(nearest_idx == 0); // Should find point (0,0)

  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;
  std::cout << "  Nearest: (" << cloud.points[nearest_idx].x() << ", "
            << cloud.points[nearest_idx].y() << ") at distance "
            << std::sqrt(nearest_dist) << std::endl;
  std::cout << "  ✓ Basic nearest neighbor search passed" << std::endl;
}

// Test 2: KNN search
void test_knn_search() {
  std::cout << "\nTest 2: KNN search" << std::endl;

  SimplePointCloud2D cloud;
  // Create a grid of points
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cloud.addPoint(i, j);
    }
  }

  icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud);

  // Test KNN search
  Eigen::Vector2d     query(2.1, 2.1);
  const int           k = 3;
  std::vector<size_t> k_indices(k);
  std::vector<double> k_distances(k);

  size_t found =
      kdtree.knn_search(query, k, k_indices.data(), k_distances.data());

  assert(found == k);

  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;
  std::cout << "  " << k << " nearest neighbors:" << std::endl;
  for (int i = 0; i < k; i++) {
    const auto &pt = cloud.points[k_indices[i]];
    std::cout << "    " << i + 1 << ": (" << pt.x() << ", " << pt.y()
              << ") at distance " << std::sqrt(k_distances[i]) << std::endl;
  }

  // Check that distances are sorted
  for (int i = 1; i < k; i++) {
    assert(k_distances[i - 1] <= k_distances[i]);
  }

  std::cout << "  ✓ KNN search passed (distances sorted)" << std::endl;
}

// Test 3: Static KNN search (template version)
void test_static_knn_search() {
  std::cout << "\nTest 3: Static KNN search" << std::endl;

  SimplePointCloud2D cloud;
  cloud.addPoint(0.0, 0.0);
  cloud.addPoint(1.0, 0.0);
  cloud.addPoint(0.0, 1.0);
  cloud.addPoint(2.0, 0.0);
  cloud.addPoint(0.0, 2.0);

  icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud);

  // Test static KNN search
  Eigen::Vector2d query(0.1, 0.1);
  constexpr int   k = 2;
  size_t          k_indices[k];
  double          k_distances[k];

  size_t found = kdtree.knn_search<k>(query, k_indices, k_distances);

  assert(found == k);

  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;
  std::cout << "  " << k << " nearest neighbors (static):" << std::endl;
  for (int i = 0; i < k; i++) {
    const auto &pt = cloud.points[k_indices[i]];
    std::cout << "    " << i + 1 << ": (" << pt.x() << ", " << pt.y()
              << ") at distance " << std::sqrt(k_distances[i]) << std::endl;
  }

  std::cout << "  ✓ Static KNN search passed" << std::endl;
}

// Test 4: Random points test
void test_random_points() {
  std::cout << "\nTest 4: Random points test" << std::endl;

  // Generate random points
  std::random_device rd;
  std::mt19937       gen(42); // Fixed seed for reproducible results
  std::uniform_real_distribution<double> dis(0.0, 10.0);

  SimplePointCloud2D cloud;
  const int          num_points = 1000;

  for (int i = 0; i < num_points; i++) {
    cloud.addPoint(dis(gen), dis(gen));
  }

  icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud);

  // Test multiple queries
  for (int test = 0; test < 10; test++) {
    Eigen::Vector2d query(dis(gen), dis(gen));

    // KDTree search
    size_t kd_nearest_idx;
    double kd_nearest_dist;
    kdtree.nearest_neighbor_search(query, &kd_nearest_idx, &kd_nearest_dist);

    // Brute force search for verification
    size_t bf_nearest_idx  = 0;
    double bf_nearest_dist = distance2D(query, cloud.points[0]);

    for (size_t i = 1; i < cloud.points.size(); i++) {
      double dist = distance2D(query, cloud.points[i]);
      if (dist < bf_nearest_dist) {
        bf_nearest_dist = dist;
        bf_nearest_idx  = i;
      }
    }

    // Compare results
    assert(kd_nearest_idx == bf_nearest_idx);
    assert(std::abs(std::sqrt(kd_nearest_dist) - bf_nearest_dist) < 1e-10);
  }

  std::cout << "  ✓ Random points test passed (KDTree matches brute force)"
            << std::endl;
}

// Test 5: Different projection types
void test_different_projections() {
  std::cout << "\nTest 5: Different projection types" << std::endl;

  SimplePointCloud2D cloud;
  for (int i = 0; i < 10; i++) {
    cloud.addPoint(i * 0.1, i * 0.1 + 0.05 * (i % 2)); // Slightly diagonal line
  }

  // Test with AxisAlignedProjection
  {
    icp2d::UnsafeKdTree<SimplePointCloud2D, icp2d::AxisAlignedProjection>
        kdtree_aa(cloud);

    Eigen::Vector2d query(0.25, 0.25);
    size_t          nearest_idx;
    double          nearest_dist;

    size_t found =
        kdtree_aa.nearest_neighbor_search(query, &nearest_idx, &nearest_dist);
    assert(found == 1);

    std::cout << "  AxisAlignedProjection: found point " << nearest_idx
              << " at distance " << std::sqrt(nearest_dist) << std::endl;
  }

  // Test with NormalProjection
  {
    icp2d::UnsafeKdTree<SimplePointCloud2D, icp2d::NormalProjection> kdtree_np(
        cloud);

    Eigen::Vector2d query(0.25, 0.25);
    size_t          nearest_idx;
    double          nearest_dist;

    size_t found =
        kdtree_np.nearest_neighbor_search(query, &nearest_idx, &nearest_dist);

    assert(found == 1);

    std::cout << "  NormalProjection: found point " << nearest_idx
              << " at distance " << std::sqrt(nearest_dist) << std::endl;
  }

  std::cout << "  ✓ Different projection types test passed" << std::endl;
}

// Test 6: Safe KdTree test
void test_safe_kdtree() {
  std::cout << "\nTest 6: Safe KdTree test" << std::endl;

  auto cloud = std::make_shared<SimplePointCloud2D>();
  cloud->addPoint(0.0, 0.0);
  cloud->addPoint(1.0, 1.0);
  cloud->addPoint(2.0, 2.0);

  icp2d::KdTree<SimplePointCloud2D> safe_kdtree(cloud);

  Eigen::Vector2d query(0.5, 0.5);
  size_t          nearest_idx;
  double          nearest_dist;

  size_t found =
      safe_kdtree.nearest_neighbor_search(query, &nearest_idx, &nearest_dist);
  assert(found == 1);

  std::cout << "  Safe KdTree found point " << nearest_idx << " at distance "
            << std::sqrt(nearest_dist) << std::endl;
  std::cout << "  ✓ Safe KdTree test passed" << std::endl;
}

int main() {
  std::cout << "Running KDTree 2D Tests..." << std::endl;
  std::cout << "=============================" << std::endl;

  try {
    test_basic_functionality();
    test_knn_search();
    test_static_knn_search();
    test_random_points();
    test_different_projections();
    test_safe_kdtree();

    std::cout << "\n=============================" << std::endl;
    std::cout << "✅ All tests passed successfully!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "❌ Test failed with unknown exception" << std::endl;
    return 1;
  }

  return 0;
}
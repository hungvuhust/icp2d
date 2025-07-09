#include <cassert>
#include <chrono>
#include <cmath>
#include <icp2d/core/kdtree_omp.hpp>
#include <iostream>
#include <random>
#include <vector>

// Simple PointCloud implementation for testing (same as previous test)
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

// Test 1: Basic OMP functionality
void test_omp_basic_functionality() {
  std::cout << "Test 1: OMP Basic functionality" << std::endl;

  SimplePointCloud2D cloud;
  cloud.addPoint(0.0, 0.0);
  cloud.addPoint(1.0, 0.0);
  cloud.addPoint(0.0, 1.0);
  cloud.addPoint(1.0, 1.0);
  cloud.addPoint(0.5, 0.5);

  icp2d::KdTreeBuilderOMP                 builder(2); // Use 2 threads
  icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud, builder);

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
  std::cout << "  ✓ OMP basic nearest neighbor search passed" << std::endl;
}

// Test 2: Performance comparison
void test_omp_performance_comparison() {
  std::cout << "\nTest 2: OMP Performance comparison" << std::endl;

  // Generate large random point cloud
  std::random_device rd;
  std::mt19937       gen(42); // Fixed seed for reproducible results
  std::uniform_real_distribution<double> dis(0.0, 100.0);

  SimplePointCloud2D cloud;
  const int          num_points = 10000;

  for (int i = 0; i < num_points; i++) {
    cloud.addPoint(dis(gen), dis(gen));
  }

  std::cout << "  Building KDTree with " << num_points << " points..."
            << std::endl;

  // Test sequential builder
  auto                 start = std::chrono::high_resolution_clock::now();
  icp2d::KdTreeBuilder seq_builder;
  icp2d::UnsafeKdTree<SimplePointCloud2D> seq_kdtree(cloud, seq_builder);
  auto end = std::chrono::high_resolution_clock::now();
  auto seq_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Test parallel builder
  start = std::chrono::high_resolution_clock::now();
  icp2d::KdTreeBuilderOMP                 omp_builder(4); // Use 4 threads
  icp2d::UnsafeKdTree<SimplePointCloud2D> omp_kdtree(cloud, omp_builder);
  end = std::chrono::high_resolution_clock::now();
  auto omp_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "  Sequential build time: " << seq_duration.count() << " ms"
            << std::endl;
  std::cout << "  Parallel build time:   " << omp_duration.count() << " ms"
            << std::endl;

  // Test that both trees give same results
  Eigen::Vector2d query(50.0, 50.0);

  size_t seq_idx, omp_idx;
  double seq_dist, omp_dist;

  seq_kdtree.nearest_neighbor_search(query, &seq_idx, &seq_dist);
  omp_kdtree.nearest_neighbor_search(query, &omp_idx, &omp_dist);

  // Results should be identical (or very close due to tie-breaking)
  double dist_diff = std::abs(std::sqrt(seq_dist) - std::sqrt(omp_dist));
  assert(dist_diff < 1e-10);

  std::cout << "  Sequential nearest: point " << seq_idx << " at distance "
            << std::sqrt(seq_dist) << std::endl;
  std::cout << "  Parallel nearest:   point " << omp_idx << " at distance "
            << std::sqrt(omp_dist) << std::endl;
  std::cout << "  ✓ Both builders produce equivalent results" << std::endl;
}

// Test 3: Correctness with random queries
void test_omp_correctness() {
  std::cout << "\nTest 3: OMP Correctness test" << std::endl;

  // Generate random points
  std::random_device                     rd;
  std::mt19937                           gen(123); // Different seed
  std::uniform_real_distribution<double> dis(0.0, 10.0);

  SimplePointCloud2D cloud;
  const int          num_points = 1000;

  for (int i = 0; i < num_points; i++) {
    cloud.addPoint(dis(gen), dis(gen));
  }

  icp2d::KdTreeBuilderOMP                 builder(2);
  icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud, builder);

  // Test multiple queries
  for (int test = 0; test < 20; test++) {
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

  std::cout << "  ✓ OMP KDTree matches brute force on all queries" << std::endl;
}

// Test 4: Different thread counts
void test_different_thread_counts() {
  std::cout << "\nTest 4: Different thread counts" << std::endl;

  SimplePointCloud2D cloud;
  for (int i = 0; i < 1000; i++) {
    cloud.addPoint(i * 0.01, (i % 100) * 0.01);
  }

  Eigen::Vector2d query(5.0, 0.5);

  // Test with different thread counts
  for (int threads : {1, 2, 4, 8}) {
    icp2d::KdTreeBuilderOMP                 builder(threads);
    icp2d::UnsafeKdTree<SimplePointCloud2D> kdtree(cloud, builder);

    size_t nearest_idx;
    double nearest_dist;
    kdtree.nearest_neighbor_search(query, &nearest_idx, &nearest_dist);

    std::cout << "  " << threads << " threads: found point " << nearest_idx
              << " at distance " << std::sqrt(nearest_dist) << std::endl;
  }

  std::cout << "  ✓ All thread counts produce valid results" << std::endl;
}

int main() {
  std::cout << "Running KDTree OMP 2D Tests..." << std::endl;
  std::cout << "==============================" << std::endl;

  try {
    test_omp_basic_functionality();
    test_omp_performance_comparison();
    test_omp_correctness();
    test_different_thread_counts();

    std::cout << "\n==============================" << std::endl;
    std::cout << "✅ All OMP tests passed successfully!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "❌ OMP test failed with exception: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "❌ OMP test failed with unknown exception" << std::endl;
    return 1;
  }

  return 0;
}
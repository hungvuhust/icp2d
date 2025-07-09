#include <gtest/gtest.h>
#include <random>

#include "icp2d/core/kdtree.hpp"
#include "simple_point_cloud_2d.hpp"

using namespace icp2d;

TEST(KdTreeTest, BasicFunctionality) {
  // Create a simple point cloud
  SimplePointCloud2D cloud;
  cloud.points = {
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
      {2, 2},
      {2, 3},
      {3, 2},
  };

  // Create a KdTree
  UnsafeKdTree<SimplePointCloud2D, AxisAlignedProjection> kdtree(cloud);

  // Test nearest neighbor search
  Eigen::Vector2d query(0.1, 0.1);
  size_t          nearest_idx;
  double          nearest_dist;

  std::cout << "Test 1: Basic functionality" << std::endl;
  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;

  ASSERT_TRUE(kdtree.nearest(query, &nearest_idx, &nearest_dist));
  EXPECT_EQ(nearest_idx, 0);
  EXPECT_NEAR(nearest_dist, 0.02, 1e-6);

  const auto &nearest_point = cloud.points[nearest_idx];
  std::cout << "  Nearest: (" << nearest_point.x() << ", " << nearest_point.y()
            << ") at distance " << std::sqrt(nearest_dist) << std::endl;

  std::cout << "  ✓ Basic nearest neighbor search passed" << std::endl;
}

TEST(KdTreeTest, KnnSearch) {
  // Create a simple point cloud
  SimplePointCloud2D cloud;
  cloud.points = {
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
      {2, 2},
      {2, 3},
      {3, 2},
  };

  // Create a KdTree
  UnsafeKdTree<SimplePointCloud2D, AxisAlignedProjection> kdtree(cloud);

  // Test k-nearest neighbor search
  Eigen::Vector2d     query(2.1, 2.1);
  const size_t        k = 3;
  std::vector<size_t> k_indices(k);
  std::vector<double> k_distances(k);

  std::cout << "\nTest 2: KNN search" << std::endl;
  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;
  std::cout << "  " << k << " nearest neighbors:" << std::endl;

  kdtree.knn_search(query, k, k_indices.data(), k_distances.data());

  // Check if the distances are sorted
  for (size_t i = 1; i < k; ++i) {
    EXPECT_LE(k_distances[i - 1], k_distances[i]);
  }

  // Print results
  for (size_t i = 0; i < k; ++i) {
    const auto &point = cloud.points[k_indices[i]];
    std::cout << "    " << i + 1 << ": (" << point.x() << ", " << point.y()
              << ") at distance " << std::sqrt(k_distances[i]) << std::endl;
  }

  std::cout << "  ✓ KNN search passed (distances sorted)" << std::endl;
}

TEST(KdTreeTest, StaticKnnSearch) {
  // Create a simple point cloud
  SimplePointCloud2D cloud;
  cloud.points = {
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
      {2, 2},
      {2, 3},
      {3, 2},
  };

  // Create a KdTree
  UnsafeKdTree<SimplePointCloud2D, AxisAlignedProjection> kdtree(cloud);

  // Test k-nearest neighbor search with static k
  Eigen::Vector2d     query(0.1, 0.1);
  constexpr size_t    k = 2;
  std::vector<size_t> k_indices(k);
  std::vector<double> k_distances(k);

  std::cout << "\nTest 3: Static KNN search" << std::endl;
  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;
  std::cout << "  " << k << " nearest neighbors (static):" << std::endl;

  kdtree.knn_search<k>(query, k_indices.data(), k_distances.data());

  // Check if the distances are sorted
  for (size_t i = 1; i < k; ++i) {
    EXPECT_LE(k_distances[i - 1], k_distances[i]);
  }

  // Print results
  for (size_t i = 0; i < k; ++i) {
    const auto &point = cloud.points[k_indices[i]];
    std::cout << "    " << i + 1 << ": (" << point.x() << ", " << point.y()
              << ") at distance " << std::sqrt(k_distances[i]) << std::endl;
  }

  std::cout << "  ✓ Static KNN search passed" << std::endl;
}

TEST(KdTreeTest, RandomPoints) {
  // Create a random point cloud
  SimplePointCloud2D cloud;
  const size_t       N = 1000;
  cloud.points.reserve(N);

  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (size_t i = 0; i < N; ++i) {
    cloud.points.push_back({dis(gen), dis(gen)});
  }

  // Create a KdTree
  UnsafeKdTree<SimplePointCloud2D, AxisAlignedProjection> kdtree(cloud);

  // Test random queries
  const size_t num_queries = 100;
  for (size_t i = 0; i < num_queries; ++i) {
    // Generate a random query point
    Eigen::Vector2d query(dis(gen), dis(gen));

    // Find nearest neighbor using KdTree
    size_t kd_nearest_idx;
    double kd_nearest_dist;
    ASSERT_TRUE(kdtree.nearest(query, &kd_nearest_idx, &kd_nearest_dist));

    // Find nearest neighbor using brute force
    size_t bf_nearest_idx  = 0;
    double bf_nearest_dist = std::numeric_limits<double>::max();
    for (size_t j = 0; j < N; ++j) {
      const double dist = (cloud.points[j] - query).squaredNorm();
      if (dist < bf_nearest_dist) {
        bf_nearest_dist = dist;
        bf_nearest_idx  = j;
      }
    }

    // Compare results
    EXPECT_EQ(kd_nearest_idx, bf_nearest_idx);
    EXPECT_NEAR(kd_nearest_dist, bf_nearest_dist, 1e-6);
  }

  std::cout << "\nTest 4: Random points test" << std::endl;
  std::cout << "  ✓ Random points test passed (KDTree matches brute force)"
            << std::endl;
}

TEST(KdTreeTest, DifferentProjections) {
  // Create a simple point cloud
  SimplePointCloud2D cloud;
  cloud.points = {
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
      {2, 2},
      {2, 3},
      {3, 2},
  };

  // Create KdTrees with different projection types
  UnsafeKdTree<SimplePointCloud2D, AxisAlignedProjection> kdtree_aa(cloud);
  UnsafeKdTree<SimplePointCloud2D, NormalProjection>      kdtree_np(cloud);

  // Test nearest neighbor search
  Eigen::Vector2d query(2.05, 2.05);
  size_t          aa_nearest_idx, np_nearest_idx;
  double          aa_nearest_dist, np_nearest_dist;

  ASSERT_TRUE(kdtree_aa.nearest(query, &aa_nearest_idx, &aa_nearest_dist));
  ASSERT_TRUE(kdtree_np.nearest(query, &np_nearest_idx, &np_nearest_dist));

  std::cout << "\nTest 5: Different projections test" << std::endl;
  std::cout << "  Query: (" << query.x() << ", " << query.y() << ")"
            << std::endl;
  std::cout << "  Axis-aligned nearest: (" << cloud.points[aa_nearest_idx].x()
            << ", " << cloud.points[aa_nearest_idx].y() << ") at distance "
            << std::sqrt(aa_nearest_dist) << std::endl;
  std::cout << "  Normal projection nearest: ("
            << cloud.points[np_nearest_idx].x() << ", "
            << cloud.points[np_nearest_idx].y() << ") at distance "
            << std::sqrt(np_nearest_dist) << std::endl;

  std::cout << "  ✓ Different projections test passed" << std::endl;
}

TEST(KdTreeTest, SafeKdTree) {
  // Create a simple point cloud
  SimplePointCloud2D cloud;
  cloud.points = {
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
  };

  // Create a thread-safe KdTree
  KdTree<SimplePointCloud2D, AxisAlignedProjection> safe_kdtree(cloud);

  // Test nearest neighbor search
  Eigen::Vector2d query(0.1, 0.1);
  size_t          nearest_idx;
  double          nearest_dist;

  ASSERT_TRUE(safe_kdtree.nearest(query, &nearest_idx, &nearest_dist));
  EXPECT_EQ(nearest_idx, 0);
  EXPECT_NEAR(nearest_dist, 0.02, 1e-6);

  std::cout << "\nTest 6: Safe KdTree test" << std::endl;
  std::cout << "  ✓ Safe KdTree test passed" << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
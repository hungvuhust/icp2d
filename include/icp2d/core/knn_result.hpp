#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <vector> // Added for std::vector

namespace icp2d {

/// @brief K-nearest neighbor search result.
/// @tparam N Number of neighbors (N > 0: static, N <= 0: dynamic)
template <int N> struct KnnResult {
  /// @brief Constructor for static k.
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  KnnResult(size_t *k_indices, double *k_sq_dists)
      : k_indices(k_indices), k_sq_dists(k_sq_dists), k(N) {}

  /// @brief Constructor for dynamic k.
  /// @param k_indices [out] Indices of k-nearest neighbors
  /// @param k_sq_dists [out] Squared distances to k-nearest neighbors
  /// @param k Number of neighbors
  KnnResult(size_t *k_indices, double *k_sq_dists, size_t k)
      : k_indices(k_indices), k_sq_dists(k_sq_dists), k(k) {}

  /// @brief Push a new neighbor.
  /// @param index Index of the neighbor
  /// @param sq_dist Squared distance to the neighbor
  void push(size_t index, double sq_dist) {
    if constexpr (N > 0) {
      // Static k
      if (count < k) {
        k_indices[count]  = index;
        k_sq_dists[count] = sq_dist;
        count++;

        if (count == k) {
          // Build max heap
          for (size_t i = k / 2; i > 0; i--) {
            size_t parent = i - 1;
            sift_down(parent);
          }
        }
      } else if (sq_dist < k_sq_dists[0]) {
        // Replace root
        k_indices[0]  = index;
        k_sq_dists[0] = sq_dist;
        sift_down(0);
      }
    } else {
      // Dynamic k
      if (count < k) {
        k_indices[count]  = index;
        k_sq_dists[count] = sq_dist;
        count++;

        if (count == k) {
          // Build max heap
          for (size_t i = k / 2; i > 0; i--) {
            size_t parent = i - 1;
            sift_down(parent);
          }
        }
      } else if (sq_dist < k_sq_dists[0]) {
        // Replace root
        k_indices[0]  = index;
        k_sq_dists[0] = sq_dist;
        sift_down(0);
      }
    }
  }

  /// @brief Get the worst distance.
  /// @return Worst distance
  double worst_distance() const {
    if constexpr (N == 1) {
      return count == 0 ? std::numeric_limits<double>::max() : k_sq_dists[0];
    } else {
      return count < k ? std::numeric_limits<double>::max() : k_sq_dists[0];
    }
  }

  /// @brief Get the number of found neighbors.
  /// @return Number of found neighbors
  size_t num_found() const { return count; }

  /// @brief Check if the result is full.
  /// @return True if the result is full
  bool is_full() const { return count >= k; }

  /// @brief Sort the results in ascending order of distances.
  void sort() {
    // Create pairs of (distance, index)
    std::vector<std::pair<double, size_t>> pairs(count);
    for (size_t i = 0; i < count; i++) {
      pairs[i] = std::make_pair(k_sq_dists[i], k_indices[i]);
    }

    // Sort pairs by distance
    std::sort(pairs.begin(), pairs.end());

    // Copy back to arrays
    for (size_t i = 0; i < count; i++) {
      k_sq_dists[i] = pairs[i].first;
      k_indices[i]  = pairs[i].second;
    }
  }

private:
  /// @brief Sift down the element at index.
  /// @param index Index of the element
  void sift_down(size_t index) {
    while (true) {
      size_t child = 2 * index + 1;
      if (child >= count) {
        break;
      }

      if (child + 1 < count && k_sq_dists[child + 1] > k_sq_dists[child]) {
        child++;
      }

      if (k_sq_dists[index] >= k_sq_dists[child]) {
        break;
      }

      std::swap(k_indices[index], k_indices[child]);
      std::swap(k_sq_dists[index], k_sq_dists[child]);
      index = child;
    }
  }

private:
  size_t *k_indices;  ///< Indices of k-nearest neighbors
  double *k_sq_dists; ///< Squared distances to k-nearest neighbors
  size_t  k;          ///< Number of neighbors
  size_t  count = 0;  ///< Current number of neighbors
};

} // namespace icp2d

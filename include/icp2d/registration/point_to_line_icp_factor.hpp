#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <icp2d/core/traits.hpp>
#include <icp2d/util/lie.hpp>
#include <icp2d/util/robust_kernel.hpp>

namespace icp2d {

/// @brief Point-to-line ICP factor with robust kernel and point distance
/// scaling
template <typename RobustKernelType = Huber> struct PointToLineICPFactor {
  struct Setting {
    typename RobustKernelType::Setting robust_kernel;
    double                             distance_scale_factor =
        0.1; ///< Scale factor for point distance weighting
    Setting(const typename RobustKernelType::Setting &kernel_setting =
                typename RobustKernelType::Setting(),
            double scale_factor = 0.1)
        : robust_kernel(kernel_setting), distance_scale_factor(scale_factor) {}
  };

  explicit PointToLineICPFactor(const Setting &setting = Setting())
      : robust_kernel_(setting.robust_kernel),
        distance_scale_factor_(setting.distance_scale_factor) {}

  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector>
  bool linearize(const TargetPointCloud &target, const SourcePointCloud &source,
                 const TargetTree &target_tree, const Eigen::Isometry2d &T,
                 size_t source_idx, const CorrespondenceRejector &rejector,
                 Eigen::Matrix<double, 3, 3> *H, Eigen::Matrix<double, 3, 1> *b,
                 double *e) const {
    // Get source point and transform it
    const Eigen::Vector2d source_pt      = traits::point(source, source_idx);
    const Eigen::Vector2d transformed_pt = T * source_pt;

    // Find two nearest neighbors in target (similar to MRPT approach)
    std::vector<size_t> nn_indices(2);
    std::vector<double> nn_dists(2);
    size_t found = target_tree.knn_search(transformed_pt, 2, nn_indices.data(),
                                          nn_dists.data());

    if (found < 2) {
      return false; // Need at least 2 points to form a line
    }

    // Check if correspondence should be rejected
    if (rejector(target, source, T, nn_indices[0], source_idx, nn_dists[0])) {
      return false;
    }

    // Get two nearest points to form a line
    const Eigen::Vector2d pt1 = traits::point(target, nn_indices[0]);
    const Eigen::Vector2d pt2 = traits::point(target, nn_indices[1]);

    // Compute line direction and normal
    Eigen::Vector2d line_dir    = pt2 - pt1;
    const double    line_length = line_dir.norm();
    if (line_length < 1e-8) {
      return false; // Degenerate line
    }
    line_dir = line_dir / line_length;

    // Normal is perpendicular to line direction
    Eigen::Vector2d line_normal(-line_dir.y(), line_dir.x());

    // Compute point-to-line distance (similar to MRPT)
    // Using the standard point-to-line distance formula
    const Eigen::Vector2d diff = transformed_pt - pt1;
    const double signed_distance = diff.dot(line_normal);
    
    // Use signed distance (like original approach)
    const double error = signed_distance;

    // Apply robust kernel
    const double w = robust_kernel_.weight(std::abs(error));

    // Apply point distance scaling: points further from origin get less weight
    const double distance_scale =
        1.0 / (1.0 + distance_scale_factor_ * source_pt.norm());
    const double combined_weight = w * distance_scale;

    // Compute Jacobian for signed distance: e = d where d = (T*p_s - p_t)^T * n_t
    Eigen::Matrix<double, 1, 3> J;

    // Compute jacobian of signed distance
    const Eigen::Vector2d rotated_source = T.linear() * source_pt;
    double J_signed_rotation = line_normal.x() * (-rotated_source.y()) +
                               line_normal.y() * rotated_source.x();
    double J_signed_tx = line_normal.x();
    double J_signed_ty = line_normal.y();
    
    // Use signed distance jacobian directly
    J(0) = J_signed_rotation;
    J(1) = J_signed_tx;
    J(2) = J_signed_ty;

    // Apply weighting
    *H = (J.transpose() * J) * combined_weight;
    *b = -J.transpose() * error * combined_weight;
    *e = robust_kernel_.robust_error(error) * distance_scale;

    return true;
  }

  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree>
  double error(const TargetPointCloud &target, const SourcePointCloud &source,
               const TargetTree        &target_tree,
               const Eigen::Isometry2d &T) const {
    double total_error = 0.0;
    size_t num_valid   = 0;

    // For each source point
    for (size_t i = 0; i < traits::size(source); ++i) {
      const Eigen::Vector2d source_pt      = traits::point(source, i);
      const Eigen::Vector2d transformed_pt = T * source_pt;

      // Find two nearest neighbors to form a line (same as linearize)
      std::vector<size_t> nn_indices(2);
      std::vector<double> nn_dists(2);
      size_t found = target_tree.knn_search(transformed_pt, 2, nn_indices.data(),
                                            nn_dists.data());

      if (found < 2) {
        continue; // Need at least 2 points to form a line
      }

      // Get two nearest points to form a line
      const Eigen::Vector2d pt1 = traits::point(target, nn_indices[0]);
      const Eigen::Vector2d pt2 = traits::point(target, nn_indices[1]);

      // Compute line direction and normal
      Eigen::Vector2d line_dir    = pt2 - pt1;
      const double    line_length = line_dir.norm();
      if (line_length < 1e-8) {
        continue; // Degenerate line
      }
      line_dir = line_dir / line_length;

      // Normal is perpendicular to line direction
      Eigen::Vector2d line_normal(-line_dir.y(), line_dir.x());

      // Compute point-to-line distance using absolute distance
      const Eigen::Vector2d diff = transformed_pt - pt1;
      const double signed_distance = diff.dot(line_normal);
      const double error = std::abs(signed_distance);

      // Apply robust kernel and distance scaling
      const double distance_scale =
          1.0 / (1.0 + distance_scale_factor_ * source_pt.norm());
      total_error += robust_kernel_.robust_error(error) * distance_scale;
      num_valid++;
    }

    return num_valid > 0 ? total_error / num_valid
                         : std::numeric_limits<double>::max();
  }

private:
  RobustKernelType robust_kernel_;
  double           distance_scale_factor_;

public:
  bool   inlier() const { return true; }       // Simple implementation for now
  double get_step_size() const { return 1.0; } // Default step size
};

// Convenience type aliases for different robust kernels
using HuberPointToLineICPFactor        = PointToLineICPFactor<Huber>;
using CauchyPointToLineICPFactor       = PointToLineICPFactor<Cauchy>;
using TukeyPointToLineICPFactor        = PointToLineICPFactor<Tukey>;
using FairPointToLineICPFactor         = PointToLineICPFactor<Fair>;
using GemanMcClurePointToLineICPFactor = PointToLineICPFactor<GemanMcClure>;
using WelschPointToLineICPFactor       = PointToLineICPFactor<Welsch>;

} // namespace icp2d
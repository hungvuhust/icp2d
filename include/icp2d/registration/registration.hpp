
#pragma once

#include <icp2d/core/kdtree.hpp>
#include <icp2d/core/projection.hpp>
#include <icp2d/core/traits.hpp>
#include <icp2d/registration/optimizer.hpp>
#include <icp2d/registration/point_to_line_icp_factor.hpp>
#include <icp2d/registration/reduction.hpp>
#include <icp2d/registration/registration_result.hpp>
#include <icp2d/registration/rejector.hpp>
#include <icp2d/registration/termination_criteria.hpp>
#include <icp2d/util/general_factor.hpp>
#include <icp2d/util/point_icp_factor.hpp>
#include <icp2d/util/robust_kernel.hpp>
#include <memory>

namespace icp2d {

/// @brief Point cloud registration in 2D
class Registration {
public:
  struct Setting {
    std::shared_ptr<RobustKernel> robust_kernel;
    TerminationCriteria           criteria;
    CombinedRejector              rejector;

    Setting()
        : robust_kernel(std::make_shared<Huber>()),
          criteria(TerminationCriteria()), rejector(CombinedRejector()) {}
  };

  explicit Registration(const Setting &setting = Setting())
      : factor_(setting.robust_kernel), criteria_(setting.criteria),
        rejector_(setting.rejector) {}

  /// @brief Estimate transformation between point clouds
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param T_init Initial transformation guess (optional)
  /// @return true if estimation successful
  template <typename TargetPointCloud, typename SourcePointCloud>
  bool estimate(const TargetPointCloud &target, const SourcePointCloud &source,
                Eigen::Isometry2d *T_est) const {
    if (!T_est) {
      return false;
    }

    if (target.size() < 3 || traits::size(source) < 3) {
      std::cerr << "Error: Point clouds too small (target: " << target.size()
                << ", source: " << traits::size(source) << ")" << std::endl;
      return false;
    }

    // Create KD-tree for target point cloud
    auto target_ptr = std::make_shared<TargetPointCloud>(target);
    KdTree<TargetPointCloud> target_tree(target_ptr);

    // Initialize optimizer
    GaussNewtonOptimizer optimizer;
    optimizer.max_iterations = criteria_.max_iterations;
    // optimizer.min_delta      = criteria_.min_delta;

    // Create point-to-line factors
    std::vector<HuberPointToLineICPFactor> factors(traits::size(source));

    // Create reduction and general factor
    PointFactorReduction reduction;
    NullFactor           general_factor;

    // Run optimization
    auto result =
        optimizer.optimize(target, source, target_tree, rejector_, criteria_,
                           reduction, *T_est, factors, general_factor);

    if (!result.converged) {
      std::cerr << "Warning: Optimization failed" << std::endl;
      return false;
    }

    *T_est = result.T_target_source;
    return result.error < criteria_.eps_error;
  }

private:
  std::shared_ptr<RobustKernel> factor_;
  TerminationCriteria           criteria_;
  CombinedRejector              rejector_;
};

} // namespace icp2d


#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

#include "../util/lie.hpp"
#include "registration_result.hpp"

namespace icp2d {

/// @brief Gauss-Newton optimizer
class GaussNewtonOptimizer {
public:
  /// @brief Constructor
  GaussNewtonOptimizer() : verbose(false), max_iterations(100), lambda(1e-6) {}

  /// @brief Tối ưu hóa phép biến đổi
  /// @tparam TargetPointCloud Kiểu point cloud đích
  /// @tparam SourcePointCloud Kiểu point cloud nguồn
  /// @tparam TargetTree Kiểu cây tìm kiếm lân cận
  /// @tparam CorrespondenceRejector Kiểu bộ loại bỏ điểm tương ứng
  /// @tparam TerminationCriteria Kiểu tiêu chí dừng
  /// @tparam Reduction Kiểu reduction
  /// @tparam Factor Kiểu factor
  /// @tparam GeneralFactor Kiểu general factor
  /// @param target Point cloud đích
  /// @param source Point cloud nguồn
  /// @param target_tree Cây tìm kiếm lân cận cho point cloud đích
  /// @param rejector Bộ loại bỏ điểm tương ứng
  /// @param criteria Tiêu chí dừng
  /// @param reduction Reduction
  /// @param init_T Phép biến đổi ban đầu
  /// @param factors Vector các factor
  /// @param general_factor General factor
  /// @return Kết quả đăng ký
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector,
            typename TerminationCriteria, typename Reduction, typename Factor,
            typename GeneralFactor>
  RegistrationResult
  optimize(const TargetPointCloud &target, const SourcePointCloud &source,
           const TargetTree             &target_tree,
           const CorrespondenceRejector &rejector,
           const TerminationCriteria &criteria, Reduction &reduction,
           const Eigen::Isometry2d &init_T, std::vector<Factor> &factors,
           GeneralFactor &general_factor) const {
    //
    if (verbose) {
      std::cout << "--- GN optimization ---" << std::endl;
    }

    RegistrationResult result(init_T);
    double             prev_error     = std::numeric_limits<double>::max();
    int                num_bad_steps  = 0;
    double             current_lambda = lambda; // Use local copy of lambda
    double             trust_radius   = 0.3;    // Initial trust region radius

    for (int i = 0; i < max_iterations && !result.converged; i++) {
      // Linearize
      auto [H, b, e] =
          reduction.linearize(target, source, target_tree, rejector,
                              result.T_target_source, factors);
      general_factor.update_linearized_system(
          target, source, target_tree, result.T_target_source, &H, &b, &e);

      // Add damping to improve numerical stability
      const double min_eig = H.eigenvalues().real().minCoeff();
      if (min_eig < 1e-6) {
        H.diagonal().array() += std::max(1e-6 - min_eig, 0.0);
      }
      H.diagonal() = H.diagonal() * (1.0 + current_lambda);

      // Solve linear system
      Eigen::Matrix<double, 3, 1> delta = H.ldlt().solve(-b);

      // Scale step to trust radius if necessary
      const double step_norm =
          std::sqrt(delta(0) * delta(0) + delta.tail<2>().squaredNorm());
      if (step_norm > trust_radius) {
        delta *= trust_radius / step_norm;
      }

      // Try different step sizes with more granular steps for better
      // convergence
      double            best_error    = e;
      double            best_step     = 1.0;
      Eigen::Isometry2d best_T        = result.T_target_source;
      const double      step_scales[] = {2.0, 1.5, 1.0, 0.8, 0.6,
                                         0.4, 0.2, 0.1, 0.05};

      for (double step_scale : step_scales) {
        const Eigen::Isometry2d T_new =
            result.T_target_source * se2_exp_isometry(step_scale * delta);
        double new_error =
            reduction.error(target, source, target_tree, T_new, factors);
        general_factor.update_error(target, source, T_new, &new_error);

        if (new_error < best_error) {
          best_error = new_error;
          best_step  = step_scale;
          best_T     = T_new;
        }
      }

      if (verbose) {
        std::cout << "iter=" << i << " e=" << e << " new_e=" << best_error
                  << " lambda=" << current_lambda << " step=" << best_step
                  << " dt=" << (best_step * delta.template segment<2>(1)).norm()
                  << " dr=" << (best_step * delta.head<1>()).norm()
                  << " radius=" << trust_radius << std::endl;
      }

      // Compute actual vs predicted reduction with better scaling
      const double actual_reduction    = e - best_error;
      const double predicted_reduction = -0.5 * delta.dot(2.0 * b + H * delta);
      const double reduction_ratio =
          actual_reduction /
          (predicted_reduction + 1e-10); // Avoid division by zero

      // Update trust region radius and damping with more aggressive updates
      if (reduction_ratio > 0.75) {
        // Very good step - increase trust region more aggressively
        trust_radius   = std::min(3.0 * trust_radius, 1.0);
        current_lambda = std::max(current_lambda * 0.1, 1e-10);
      } else if (reduction_ratio > 0.25) {
        // Good step - increase trust region moderately
        trust_radius   = std::min(1.5 * trust_radius, 0.8);
        current_lambda = std::max(current_lambda * 0.5, 1e-8);
      } else if (reduction_ratio > 0.0) {
        // Okay step - maintain current trust region
        trust_radius   = std::min(trust_radius, 0.5);
        current_lambda = std::min(current_lambda * 2.0, 1e-6);
      } else {
        // Bad step - shrink trust region aggressively
        trust_radius   = 0.25 * trust_radius;
        current_lambda = std::min(current_lambda * 10.0, 1e-4);
      }

      // Accept step if we made any progress
      if (best_error < e) {
        result.converged = criteria.converged(best_step * delta, best_error);
        result.T_target_source = best_T;
        prev_error             = best_error;
        num_bad_steps          = 0;
      } else {
        num_bad_steps++;
        if (num_bad_steps > 3) { // Reduced from 5 to 3 for faster termination
                                 // on bad convergence
          break;
        }
      }

      result.iterations = i;
      result.H          = H;
      result.b          = b;
      result.error      = e;
    }

    result.num_inliers =
        std::count_if(factors.begin(), factors.end(),
                      [](const auto &factor) { return factor.inlier(); });

    return result;
  }

  bool   verbose;        ///< If true, print debug messages
  int    max_iterations; ///< Max number of optimization iterations
  double lambda;         ///< Initial damping factor
};

/// @brief Levenberg-Marquardt optimizer
class LevenbergMarquardtOptimizer {
public:
  /// @brief Constructor
  LevenbergMarquardtOptimizer()
      : is_verbose(false), max_iterations(50), max_inner_iterations(20),
        init_lambda(1e-4), lambda_factor(5.0) {}

  /// @brief Tối ưu hóa phép biến đổi
  /// @tparam TargetPointCloud Kiểu point cloud đích
  /// @tparam SourcePointCloud Kiểu point cloud nguồn
  /// @tparam TargetTree Kiểu cây tìm kiếm lân cận
  /// @tparam CorrespondenceRejector Kiểu bộ loại bỏ điểm tương ứng
  /// @tparam TerminationCriteria Kiểu tiêu chí dừng
  /// @tparam Reduction Kiểu reduction
  /// @tparam Factor Kiểu factor
  /// @tparam GeneralFactor Kiểu general factor
  /// @param target Point cloud đích
  /// @param source Point cloud nguồn
  /// @param target_tree Cây tìm kiếm lân cận cho point cloud đích
  /// @param rejector Bộ loại bỏ điểm tương ứng
  /// @param criteria Tiêu chí dừng
  /// @param reduction Reduction
  /// @param init_T Phép biến đổi ban đầu
  /// @param factors Vector các factor
  /// @param general_factor General factor
  /// @return Kết quả đăng ký
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector,
            typename TerminationCriteria, typename Reduction, typename Factor,
            typename GeneralFactor>
  RegistrationResult
  optimize(const TargetPointCloud &target, const SourcePointCloud &source,
           const TargetTree             &target_tree,
           const CorrespondenceRejector &rejector,
           const TerminationCriteria &criteria, Reduction &reduction,
           const Eigen::Isometry2d &init_T, std::vector<Factor> &factors,
           GeneralFactor &general_factor) const {
    //
    if (is_verbose) {
      std::cout << "--- LM optimization ---" << std::endl;
    }

    double             lambda = init_lambda;
    RegistrationResult result(init_T);
    for (int i = 0; i < max_iterations && !result.converged; i++) {
      // Linearize
      auto [H, b, e] =
          reduction.linearize(target, source, target_tree, rejector,
                              result.T_target_source, factors);
      general_factor.update_linearized_system(
          target, source, target_tree, result.T_target_source, &H, &b, &e);

      // Lambda iteration
      bool success = false;
      for (int j = 0; j < max_inner_iterations; j++) {
        // Solve with damping
        Eigen::Matrix<double, 3, 1> delta =
            (H + lambda * Eigen::Matrix<double, 3, 3>::Identity())
                .ldlt()
                .solve(-b);

        // Điều chỉnh step size cho góc xoay và dịch chuyển
        const double step_size = factors[0].get_step_size();
        const double rot_scale = 0.1; // Hệ số điều chỉnh cho góc xoay
        delta(0) *= step_size * rot_scale;
        delta.template segment<2>(1) *= step_size;

        // Validate new solution
        const Eigen::Isometry2d new_T =
            result.T_target_source * se2_exp_isometry(delta);
        double new_e =
            reduction.error(target, source, target_tree, new_T, factors);
        general_factor.update_error(target, source, new_T, &new_e);

        if (is_verbose) {
          std::cout << "iter=" << i << " inner=" << j << " e=" << e
                    << " new_e=" << new_e << " lambda=" << lambda
                    << " dt=" << delta.template segment<2>(1).norm()
                    << " dr=" << delta.head<1>().norm() << std::endl;
        }

        if (new_e <= e) {
          // Error decreased, decrease lambda
          result.converged       = criteria.converged(delta, new_e);
          result.T_target_source = new_T;
          lambda /= lambda_factor;
          success = true;
          e       = new_e;

          break;
        } else {
          // Failed to decrease error, increase lambda
          lambda *= lambda_factor;
        }
      }

      result.iterations = i;
      result.H          = H;
      result.b          = b;
      result.error      = e;

      if (!success) {
        break;
      }
    }

    result.num_inliers =
        std::count_if(factors.begin(), factors.end(),
                      [](const auto &factor) { return factor.inlier(); });

    return result;
  }

  bool is_verbose;           ///< If true, print debug messages
  int  max_iterations;       ///< Max number of optimization iterations
  int  max_inner_iterations; ///< Max number of inner iterations (lambda-trial)
  double init_lambda;        ///< Initial lambda (damping factor)
  double lambda_factor;      ///< Lambda increase factor
};

} // namespace icp2d

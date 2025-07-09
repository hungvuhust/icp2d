
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace icp2d {

/// @brief Huber robust kernel for 2D ICP
/// @note  Robust kernels help handle outliers in point correspondences
struct Huber {
public:
  /// @brief Huber robust kernel setting
  struct Setting {
    double c; ///< Kernel width parameter (threshold)
    Setting(double c = 1.0) : c(c) {}
  };

  /// @brief Constructor
  /// @param setting Kernel configuration
  Huber(const Setting &setting = Setting()) : c(setting.c) {}

  /// @brief Compute the weight for an error
  /// @param e Squared error (distance^2)
  /// @return  Weight to apply to the error term
  double weight(double e) const {
    const double e_sqrt = std::sqrt(e);
    return e_sqrt < c ? 1.0 : c / e_sqrt;
  }

  /// @brief Compute the robust error (weighted error)
  /// @param e Squared error
  /// @return  Robust error value
  double robust_error(double e) const {
    const double e_sqrt = std::sqrt(e);
    if (e_sqrt < c) {
      return e; // Quadratic region
    } else {
      return 2.0 * c * e_sqrt - c * c; // Linear region
    }
  }

public:
  const double c; ///< Kernel width parameter
};

/// @brief Cauchy robust kernel for 2D ICP
/// @note  More aggressive outlier rejection than Huber
struct Cauchy {
public:
  /// @brief Cauchy robust kernel setting
  struct Setting {
    double c; ///< Kernel width parameter
    Setting(double c = 1.0) : c(c) {}
  };

  /// @brief Constructor
  /// @param setting Kernel configuration
  Cauchy(const Setting &setting = Setting()) : c(setting.c) {}

  /// @brief Compute the weight for an error
  /// @param e Squared error (distance^2)
  /// @return  Weight to apply to the error term
  double weight(double e) const { return c * c / (c * c + e); }

  /// @brief Compute the robust error (weighted error)
  /// @param e Squared error
  /// @return  Robust error value
  double robust_error(double e) const {
    return c * c * std::log(1.0 + e / (c * c));
  }

public:
  const double c; ///< Kernel width parameter
};

/// @brief Tukey (Bisquare) robust kernel for 2D ICP
/// @note  Completely rejects outliers beyond threshold
struct Tukey {
public:
  /// @brief Tukey robust kernel setting
  struct Setting {
    double c; ///< Kernel width parameter
    Setting(double c = 1.0) : c(c) {}
  };

  /// @brief Constructor
  /// @param setting Kernel configuration
  Tukey(const Setting &setting = Setting()) : c(setting.c) {}

  /// @brief Compute the weight for an error
  /// @param e Squared error (distance^2)
  /// @return  Weight to apply to the error term
  double weight(double e) const {
    const double e_sqrt = std::sqrt(e);
    if (e_sqrt < c) {
      const double t = 1.0 - (e_sqrt / c) * (e_sqrt / c);
      return t * t;
    } else {
      return 0.0; // Complete rejection
    }
  }

  /// @brief Compute the robust error (weighted error)
  /// @param e Squared error
  /// @return  Robust error value
  double robust_error(double e) const {
    const double e_sqrt = std::sqrt(e);
    if (e_sqrt < c) {
      const double t = e_sqrt / c;
      return (c * c / 6.0) * (1.0 - std::pow(1.0 - t * t, 3.0));
    } else {
      return c * c / 6.0; // Constant for outliers
    }
  }

public:
  const double c; ///< Kernel width parameter
};

/// @brief Fair robust kernel for 2D ICP
/// @note  Good balance between outlier rejection and smoothness
struct Fair {
public:
  /// @brief Fair robust kernel setting
  struct Setting {
    double c; ///< Kernel width parameter
    Setting(double c = 1.0) : c(c) {}
  };

  /// @brief Constructor
  /// @param setting Kernel configuration
  Fair(const Setting &setting = Setting()) : c(setting.c) {}

  /// @brief Compute the weight for an error
  /// @param e Squared error (distance^2)
  /// @return  Weight to apply to the error term
  double weight(double e) const {
    const double e_sqrt = std::sqrt(e);
    return c / (c + e_sqrt);
  }

  /// @brief Compute the robust error (weighted error)
  /// @param e Squared error
  /// @return  Robust error value
  double robust_error(double e) const {
    const double e_sqrt = std::sqrt(e);
    return c * c * (e_sqrt / c - std::log(1.0 + e_sqrt / c));
  }

public:
  const double c; ///< Kernel width parameter
};

/// @brief No robust kernel (identity) for 2D ICP
/// @note  Standard least squares without robustification
struct None {
public:
  /// @brief No robust kernel setting (empty)
  struct Setting {};

  /// @brief Constructor
  None(const Setting &setting = Setting()) {}

  /// @brief Compute the weight for an error (always 1.0)
  /// @param e Squared error (distance^2)
  /// @return  Weight = 1.0 (no robustification)
  double weight(double e) const {
    (void)e; // Suppress unused parameter warning
    return 1.0;
  }

  /// @brief Compute the robust error (same as input)
  /// @param e Squared error
  /// @return  Same error (no robustification)
  double robust_error(double e) const { return e; }
};

/// @brief Robustify a factor with a robust kernel for 2D ICP
/// @tparam Kernel  Robust kernel type
/// @tparam Factor  Factor type (e.g., point-to-point, point-to-line)
template <typename Kernel, typename Factor> struct RobustFactor {
public:
  /// @brief Robust factor setting
  struct Setting {
    typename Kernel::Setting robust_kernel; ///< Robust kernel setting
    typename Factor::Setting factor;        ///< Factor setting
  };

  /// @brief Constructor
  /// @param setting Configuration for both kernel and factor
  RobustFactor(const Setting &setting = Setting())
      : robust_kernel(setting.robust_kernel), factor(setting.factor) {}

  /// @brief Linearize the factor with robust kernel for 2D
  /// @tparam TargetPointCloud Target point cloud type
  /// @tparam SourcePointCloud Source point cloud type
  /// @tparam TargetTree Target tree type (e.g., KDTree)
  /// @tparam CorrespondenceRejector Correspondence rejection type
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param target_tree Target KDTree for nearest neighbor search
  /// @param T Current 2D transformation estimate
  /// @param source_index Index of source point
  /// @param rejector Correspondence rejector
  /// @param H Output Hessian matrix (3x3 for SE(2))
  /// @param b Output gradient vector (3x1 for SE(2))
  /// @param e Output error value
  /// @return true if linearization successful
  template <typename TargetPointCloud, typename SourcePointCloud,
            typename TargetTree, typename CorrespondenceRejector>
  bool linearize(const TargetPointCloud &target, const SourcePointCloud &source,
                 const TargetTree &target_tree, const Eigen::Isometry2d &T,
                 size_t source_index, const CorrespondenceRejector &rejector,
                 Eigen::Matrix<double, 3, 3> *H, Eigen::Matrix<double, 3, 1> *b,
                 double *e) {
    // First, linearize with the base factor
    if (!factor.linearize(target, source, target_tree, T, source_index,
                          rejector, H, b, e)) {
      return false;
    }

    // Apply robust kernel weighting
    const double w = robust_kernel.weight(*e);
    *H *= w;
    *b *= w;
    *e = robust_kernel.robust_error(*e);

    return true;
  }

  /// @brief Evaluate error with robust kernel for 2D
  /// @tparam TargetPointCloud Target point cloud type
  /// @tparam SourcePointCloud Source point cloud type
  /// @param target Target point cloud
  /// @param source Source point cloud
  /// @param T Current 2D transformation
  /// @return Robust error value
  template <typename TargetPointCloud, typename SourcePointCloud>
  double error(const TargetPointCloud &target, const SourcePointCloud &source,
               const Eigen::Isometry2d &T) const {
    const double e = factor.error(target, source, T);
    return robust_kernel.robust_error(e);
  }

  /// @brief Check if the factor is considered inlier
  /// @return true if factor is inlier
  bool inlier() const { return factor.inlier(); }

  /// @brief Get the robust kernel weight for given error
  /// @param e Squared error
  /// @return Weight value
  double get_weight(double e) const { return robust_kernel.weight(e); }

public:
  Kernel robust_kernel; ///< Robust kernel instance
  Factor factor;        ///< Base factor instance
};

// Type aliases for convenience
using HuberFactor =
    RobustFactor<Huber, void>; // Will be specialized with actual factors
using CauchyFactor = RobustFactor<Cauchy, void>;
using TukeyFactor  = RobustFactor<Tukey, void>;
using FairFactor   = RobustFactor<Fair, void>;

} // namespace icp2d


#pragma once

#include <cmath>

namespace icp2d {

/// @brief Base class for robust kernels
struct RobustKernel {
  virtual ~RobustKernel() = default;

  /// @brief Compute weight for error value
  /// @param error Error value
  /// @return Weight in [0,1]
  virtual double weight(double error) const = 0;

  /// @brief Compute robust error value
  /// @param error Original error value
  /// @return Robust error value
  virtual double robust_error(double error) const = 0;
};

/// @brief No robustification (standard least squares)
struct None : public RobustKernel {
  struct Setting {};

  explicit None(const Setting & = Setting()) {}

  double weight(double) const override { return 1.0; }

  double robust_error(double error) const override {
    return 0.5 * error * error;
  }
};

/// @brief Huber kernel (quadratic near zero, linear far from zero)
struct Huber : public RobustKernel {
  struct Setting {
    double c; ///< Threshold between quadratic and linear regions
    Setting(double c = 0.5)
        : c(c) {} // Reduced from 1.0 to 0.5 for better outlier rejection
  };

  explicit Huber(const Setting &setting = Setting()) : c(setting.c) {}

  double weight(double error) const override {
    const double abs_error = std::abs(error);
    return abs_error <= c ? 1.0 : c / abs_error;
  }

  double robust_error(double error) const override {
    const double abs_error = std::abs(error);
    if (abs_error <= c) {
      return 0.5 * error * error;
    }
    return c * abs_error - 0.5 * c * c;
  }

  double c;
};

/// @brief Cauchy kernel (more aggressive than Huber)
struct Cauchy : public RobustKernel {
  struct Setting {
    double c; ///< Scale parameter
    Setting(double c = 0.3)
        : c(c) {} // Reduced from 0.5 to 0.3 for stronger outlier rejection
  };

  explicit Cauchy(const Setting &setting = Setting()) : c(setting.c) {}

  double weight(double error) const override {
    const double scaled_error = error / c;
    return 1.0 / (1.0 + scaled_error * scaled_error);
  }

  double robust_error(double error) const override {
    const double scaled_error = error / c;
    return 0.5 * c * c * std::log(1.0 + scaled_error * scaled_error);
  }

  double c;
};

/// @brief Tukey kernel (completely rejects large errors)
struct Tukey : public RobustKernel {
  struct Setting {
    double c; ///< Cutoff parameter
    Setting(double c = 0.7)
        : c(c) {} // Reduced from 1.0 to 0.7 for better balance
  };

  explicit Tukey(const Setting &setting = Setting()) : c(setting.c) {}

  double weight(double error) const override {
    const double abs_error = std::abs(error);
    if (abs_error > c)
      return 0.0;
    const double s = abs_error / c;
    const double w = 1.0 - s * s;
    return w * w;
  }

  double robust_error(double error) const override {
    const double abs_error = std::abs(error);
    if (abs_error > c)
      return c * c / 6.0;
    const double s = abs_error / c;
    const double w = 1.0 - s * s;
    return (c * c / 6.0) * (1.0 - w * w * w);
  }

  double c;
};

/// @brief Fair kernel (similar to Huber but smoother transition)
struct Fair : public RobustKernel {
  struct Setting {
    double c; ///< Scale parameter
    Setting(double c = 0.4)
        : c(c) {} // Reduced from 0.6 to 0.4 for better performance
  };

  explicit Fair(const Setting &setting = Setting()) : c(setting.c) {}

  double weight(double error) const override {
    const double abs_error = std::abs(error);
    return 1.0 / (1.0 + abs_error / c);
  }

  double robust_error(double error) const override {
    const double abs_error = std::abs(error);
    return c * c * (abs_error / c - std::log(1.0 + abs_error / c));
  }

  double c;
};

/// @brief Geman-McClure kernel (very aggressive outlier rejection)
struct GemanMcClure : public RobustKernel {
  struct Setting {
    double c; ///< Scale parameter
    Setting(double c = 0.2)
        : c(c) {} // Small value for aggressive outlier rejection
  };

  explicit GemanMcClure(const Setting &setting = Setting()) : c(setting.c) {}

  double weight(double error) const override {
    const double s = error / c;
    const double w = 1.0 / (1.0 + s * s);
    return w * w;
  }

  double robust_error(double error) const override {
    const double s = error / c;
    return 0.5 * (s * s) / (1.0 + s * s);
  }

  double c;
};

/// @brief Welsch kernel (smooth transition with complete outlier rejection)
struct Welsch : public RobustKernel {
  struct Setting {
    double c;                         ///< Scale parameter
    Setting(double c = 0.6) : c(c) {} // Balanced value for general use
  };

  explicit Welsch(const Setting &setting = Setting()) : c(setting.c) {}

  double weight(double error) const override {
    const double s = error / c;
    return std::exp(-s * s);
  }

  double robust_error(double error) const override {
    const double s = error / c;
    return 0.5 * c * c * (1.0 - std::exp(-s * s));
  }

  double c;
};

} // namespace icp2d

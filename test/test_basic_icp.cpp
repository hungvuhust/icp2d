#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <icp2d/core/kdtree.hpp>
#include <icp2d/core/projection.hpp>
#include <icp2d/core/traits.hpp>
#include <icp2d/registration/optimizer.hpp>
#include <icp2d/registration/point_to_line_icp_factor.hpp>
#include <icp2d/registration/reduction.hpp>
#include <icp2d/registration/registration.hpp>
#include <icp2d/registration/rejector.hpp>
#include <icp2d/registration/termination_criteria.hpp>
#include <icp2d/util/general_factor.hpp>
#include <icp2d/util/lie.hpp>
#include <icp2d/util/robust_kernel.hpp>

#include <cmath>
#include <random>

namespace icp2d {

/// @brief Add Gaussian noise to a point
/// @param point Point to add noise to
/// @param std_dev Standard deviation of noise
/// @param gen Random number generator
/// @return Point with noise added
inline Eigen::Vector2d add_gaussian_noise(const Eigen::Vector2d &point,
                                          double std_dev, std::mt19937 &gen) {
  std::normal_distribution<double> dist(0.0, std_dev);
  return point + Eigen::Vector2d(dist(gen), dist(gen));
}

// Test robust kernels
TEST(BasicICPTest, RobustKernels) {
  // Test Huber kernel
  Huber::Setting huber_setting;
  huber_setting.c = 1.0;

  Huber huber_kernel(huber_setting);

  // Test with small error (should get weight ~1)
  double small_error = 0.1;
  double weight1     = huber_kernel.weight(small_error);
  EXPECT_NEAR(weight1, 1.0, 1e-6);

  // Test with large error (should get reduced weight)
  double large_error = 2.0;
  double weight2     = huber_kernel.weight(large_error);
  EXPECT_LT(weight2, 1.0);
  EXPECT_GT(weight2, 0.0);

  // Test robust error
  double robust_err = huber_kernel.robust_error(small_error);
  EXPECT_GT(robust_err, 0.0);

  std::cout << "Huber kernel - Small error weight: " << weight1 << std::endl;
  std::cout << "Huber kernel - Large error weight: " << weight2 << std::endl;
}

TEST(BasicICPTest, CauchyKernel) {
  // Test Cauchy kernel
  Cauchy::Setting cauchy_setting;
  cauchy_setting.c = 1.0;

  Cauchy cauchy_kernel(cauchy_setting);

  double small_error   = 0.1;
  double cauchy_weight = cauchy_kernel.weight(small_error);
  EXPECT_GT(cauchy_weight, 0.0);
  EXPECT_LE(cauchy_weight, 1.0);

  double large_error         = 10.0;
  double cauchy_weight_large = cauchy_kernel.weight(large_error);
  EXPECT_LT(cauchy_weight_large,
            cauchy_weight); // Should be smaller for large error

  std::cout << "Cauchy kernel - Small error weight: " << cauchy_weight
            << std::endl;
  std::cout << "Cauchy kernel - Large error weight: " << cauchy_weight_large
            << std::endl;
}

TEST(BasicICPTest, TukeyKernel) {
  // Test Tukey kernel
  Tukey::Setting tukey_setting;
  tukey_setting.c = 1.0;

  Tukey tukey_kernel(tukey_setting);

  double small_error  = 0.5;
  double weight_small = tukey_kernel.weight(small_error);
  EXPECT_GT(weight_small, 0.0);
  EXPECT_LE(weight_small, 1.0);

  double large_error  = 2.0; // Beyond threshold
  double weight_large = tukey_kernel.weight(large_error);
  EXPECT_EQ(weight_large, 0.0); // Should completely reject

  std::cout << "Tukey kernel - Small error weight: " << weight_small
            << std::endl;
  std::cout << "Tukey kernel - Large error weight: " << weight_large
            << std::endl;
}

TEST(BasicICPTest, FairKernel) {
  // Test Fair kernel
  Fair::Setting fair_setting;
  fair_setting.c = 1.0;

  Fair fair_kernel(fair_setting);

  double error  = 1.0;
  double weight = fair_kernel.weight(error);
  EXPECT_GT(weight, 0.0);
  EXPECT_LE(weight, 1.0);

  double robust_err = fair_kernel.robust_error(error);
  EXPECT_GT(robust_err, 0.0);

  std::cout << "Fair kernel - Weight: " << weight << std::endl;
  std::cout << "Fair kernel - Robust error: " << robust_err << std::endl;
}

TEST(BasicICPTest, NoKernel) {
  // Test None kernel (no robustification)
  None::Setting none_setting;
  None          none_kernel(none_setting);

  double error  = 5.0;
  double weight = none_kernel.weight(error);
  EXPECT_EQ(weight, 1.0); // Always 1.0

  double robust_err = none_kernel.robust_error(error);
  EXPECT_EQ(robust_err, 0.5 * error * error); // 0.5 * error^2 for None kernel

  std::cout << "None kernel - Weight: " << weight << std::endl;
}

// Test DoF restriction
TEST(BasicICPTest, RestrictDoF) {
  RestrictDoFFactor constraint;

  // Test default state (all DoF active)
  EXPECT_EQ(constraint.mask(0), 1.0); // rotation
  EXPECT_EQ(constraint.mask(1), 1.0); // tx
  EXPECT_EQ(constraint.mask(2), 1.0); // ty

  // Test fixing rotation
  constraint.fix_rotation();
  EXPECT_EQ(constraint.mask(0), 0.0); // rotation fixed
  EXPECT_EQ(constraint.mask(1), 1.0); // tx free
  EXPECT_EQ(constraint.mask(2), 1.0); // ty free

  // Test fixing translation
  constraint.fix_translation();
  EXPECT_EQ(constraint.mask(0), 1.0); // rotation free
  EXPECT_EQ(constraint.mask(1), 0.0); // tx fixed
  EXPECT_EQ(constraint.mask(2), 0.0); // ty fixed

  // Test fixing all
  constraint.fix_all();
  EXPECT_EQ(constraint.mask(0), 0.0);
  EXPECT_EQ(constraint.mask(1), 0.0);
  EXPECT_EQ(constraint.mask(2), 0.0);

  std::cout << "DoF constraints test passed!" << std::endl;
}

// Test correspondence rejectors
TEST(BasicICPTest, CorrespondenceRejectors) {
  // Test null rejector (accepts all)
  NullRejector null_rejector;

  std::vector<Eigen::Vector2d> target, source;
  Eigen::Isometry2d            T = Eigen::Isometry2d::Identity();

  bool rejected =
      null_rejector(target, source, T, 0, 0, 100.0); // Large distance
  EXPECT_FALSE(rejected); // Should accept even large distances

  // Test distance rejector
  DistanceRejector distance_rejector(1.0); // max distance = 1.0

  // Small distance should be accepted
  bool rejected_small =
      distance_rejector(target, source, T, 0, 0, 0.5 * 0.5); // sq_dist = 0.25
  EXPECT_FALSE(rejected_small);

  // Large distance should be rejected
  bool rejected_large =
      distance_rejector(target, source, T, 0, 0, 2.0 * 2.0); // sq_dist = 4.0
  EXPECT_TRUE(rejected_large);

  std::cout << "Correspondence rejectors test passed!" << std::endl;
}

// Test SE(2) transformations
TEST(BasicICPTest, SE2Transformations) {
  // Create transformation
  Eigen::Isometry2d T = Eigen::Isometry2d::Identity();

  // Set rotation (45 degrees)
  double angle = M_PI / 4.0;
  T.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);

  // Set translation
  T.translation() << 1.0, 2.0;

  // Test point transformation
  Eigen::Vector2d point(1, 0);
  Eigen::Vector2d transformed = T * point;

  // Expected: rotation + translation
  Eigen::Vector2d expected;
  expected << std::cos(angle) + 1.0, std::sin(angle) + 2.0;

  EXPECT_NEAR(transformed.x(), expected.x(), 1e-10);
  EXPECT_NEAR(transformed.y(), expected.y(), 1e-10);

  std::cout << "Original point: " << point.transpose() << std::endl;
  std::cout << "Transformed point: " << transformed.transpose() << std::endl;
  std::cout << "Expected point: " << expected.transpose() << std::endl;
}

// Test robust factor template
TEST(BasicICPTest, RobustFactorTemplate) {
  // Test that robust factor template compiles and basic functionality works
  Huber::Setting huber_setting;
  huber_setting.c = 1.0;

  Huber kernel(huber_setting);

  // Test weight function
  double error      = 1.5;
  double weight     = kernel.weight(error);
  double robust_err = kernel.robust_error(error);

  EXPECT_GT(weight, 0.0);
  EXPECT_LT(weight, 1.0); // Should be reduced for error > c
  EXPECT_GT(robust_err, 0.0);

  std::cout << "Robust factor template test passed!" << std::endl;
}

// Test mathematical properties
TEST(BasicICPTest, MathematicalProperties) {
  Huber::Setting setting;
  setting.c = 1.0;
  Huber kernel(setting);

  // Test monotonicity: weight should decrease as error increases
  std::vector<double> errors = {0.1, 0.5, 1.0, 1.5, 2.0, 5.0};
  std::vector<double> weights;

  for (double err : errors) {
    weights.push_back(kernel.weight(err));
  }

  // Check that weights are non-increasing
  for (size_t i = 1; i < weights.size(); ++i) {
    EXPECT_LE(weights[i],
              weights[i - 1] + 1e-10); // Allow small numerical errors
  }

  // Print weights for visualization
  std::cout << "Huber weights for increasing errors:" << std::endl;
  for (size_t i = 0; i < errors.size(); ++i) {
    std::cout << "  Error: " << errors[i] << " -> Weight: " << weights[i]
              << std::endl;
  }
}

// Test registration with sine curve
TEST(BasicICPTest, SineCurveRegistration) {
  // Tạo điểm trên đường cong sin gốc
  std::vector<Eigen::Vector2d> source_points;
  const int                    num_points = 100;
  const double                 x_start    = -M_PI;
  const double                 x_end      = M_PI;
  const double                 dx = (x_end - x_start) / (num_points - 1);

  for (int i = 0; i < num_points; ++i) {
    double x = x_start + i * dx;
    double y = std::sin(x);
    source_points.push_back(Eigen::Vector2d(x, y));
  }

  // Tạo điểm đích bằng cách áp dụng phép biến đổi
  std::vector<Eigen::Vector2d> target_points;
  Eigen::Isometry2d            T_true = Eigen::Isometry2d::Identity();

  // Xoay 15 độ
  double angle = 15.0 * M_PI / 180.0;
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);

  // Dịch chuyển (0.5, 0.3)
  T_true.translation() << 0.5, 0.3;

  // Áp dụng biến đổi để tạo target points
  for (const auto &p : source_points) {
    target_points.push_back(T_true * p);
  }

  // In thông tin để kiểm tra
  std::cout << "Sine curve test:" << std::endl;
  std::cout << "Number of points: " << num_points << std::endl;
  std::cout << "True transformation:" << std::endl;
  std::cout << "Rotation angle: " << (angle * 180.0 / M_PI) << " degrees"
            << std::endl;
  std::cout << "Translation: " << T_true.translation().transpose() << std::endl;

  // TODO: Thêm phần đăng ký ICP khi đã có implementation đầy đủ
  // Hiện tại chỉ kiểm tra việc tạo dữ liệu
  EXPECT_EQ(source_points.size(), target_points.size());
  EXPECT_EQ(source_points.size(), num_points);

  // Kiểm tra một số điểm đầu và cuối
  EXPECT_NEAR(source_points.front().x(), -M_PI, 1e-10);
  EXPECT_NEAR(source_points.back().x(), M_PI, 1e-10);
}

// Test registration với đường cong sin có nhiễu
TEST(BasicICPTest, NoisySineCurveRegistration) {
  // Tạo điểm trên đường cong sin gốc
  std::vector<Eigen::Vector2d> source_points;
  const int                    num_points = 100;
  const double                 x_start    = -M_PI;
  const double                 x_end      = M_PI;
  const double                 dx = (x_end - x_start) / (num_points - 1);

  for (int i = 0; i < num_points; ++i) {
    double x = x_start + i * dx;
    double y = std::sin(x);
    source_points.push_back(Eigen::Vector2d(x, y));
  }

  // Tạo điểm đích với biến đổi và nhiễu
  std::vector<Eigen::Vector2d> target_points;
  Eigen::Isometry2d            T_true = Eigen::Isometry2d::Identity();

  // Xoay 15 độ
  double angle = 15.0 * M_PI / 180.0;
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);

  // Dịch chuyển (0.5, 0.3)
  T_true.translation() << 0.5, 0.3;

  // Thêm nhiễu Gaussian
  std::random_device rd;
  std::mt19937       gen(rd());
  const double       noise_std_dev = 0.05; // Độ lệch chuẩn của nhiễu

  // Áp dụng biến đổi và thêm nhiễu
  for (const auto &p : source_points) {
    Eigen::Vector2d transformed = T_true * p;
    target_points.push_back(
        add_gaussian_noise(transformed, noise_std_dev, gen));
  }

  // In thông tin để kiểm tra
  std::cout << "Noisy sine curve test:" << std::endl;
  std::cout << "Number of points: " << num_points << std::endl;
  std::cout << "True transformation:" << std::endl;
  std::cout << "Rotation angle: " << (angle * 180.0 / M_PI) << " degrees"
            << std::endl;
  std::cout << "Translation: " << T_true.translation().transpose() << std::endl;
  std::cout << "Noise std dev: " << noise_std_dev << std::endl;

  // TODO: Thêm phần đăng ký ICP khi đã có implementation đầy đủ
  EXPECT_EQ(source_points.size(), target_points.size());
  EXPECT_EQ(source_points.size(), num_points);
}

// Test point-to-point ICP với đường cong sin có nhiễu
TEST(BasicICPTest, PointToPointSineCurve) {
  // Tạo dữ liệu giống như test trước
  std::vector<Eigen::Vector2d> source_points;
  std::vector<Eigen::Vector2d> target_points;
  const int                    num_points = 100;
  const double                 x_start    = -M_PI;
  const double                 x_end      = M_PI;
  const double                 dx = (x_end - x_start) / (num_points - 1);

  // Tạo source points
  for (int i = 0; i < num_points; ++i) {
    double x = x_start + i * dx;
    double y = std::sin(x);
    source_points.push_back(Eigen::Vector2d(x, y));
  }

  // Tạo transformation thật
  Eigen::Isometry2d T_true = Eigen::Isometry2d::Identity();
  double            angle  = 15.0 * M_PI / 180.0;
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.5, 0.3;

  // Thêm nhiễu
  std::random_device rd;
  std::mt19937       gen(rd());
  const double       noise_std_dev = 0.05;

  for (const auto &p : source_points) {
    Eigen::Vector2d transformed = T_true * p;
    target_points.push_back(
        add_gaussian_noise(transformed, noise_std_dev, gen));
  }

  // Khởi tạo KDTree cho target points
  using KDTree = icp2d::KdTree<std::vector<Eigen::Vector2d>>;
  auto target_points_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KDTree target_tree(target_points_ptr);

  // Khởi tạo point-to-point ICP factor
  icp2d::PointICPFactor::Setting factor_setting;
  factor_setting.weight    = 1.0;
  //   factor_setting.lambda    = 0.1; // Regularization
  //   factor_setting.damping   = 0.1; // Damping
  factor_setting.step_size = 0.5; // Step size
  icp2d::PointICPFactor p2p_factor(factor_setting);

  // Khởi tạo correspondence rejector
  icp2d::DistanceRejector rejector(0.5); // Tăng ngưỡng lên 0.5

  // Khởi tạo biến đổi ban đầu (identity)
  Eigen::Isometry2d T_est = Eigen::Isometry2d::Identity();

  // Các tham số cho optimization
  const int    max_iterations        = 50;
  const double convergence_threshold = 1e-4; // Tăng ngưỡng hội tụ lên
  double       prev_error            = std::numeric_limits<double>::max();

  std::cout << "Point-to-point ICP test with sine curve:" << std::endl;
  std::cout << "Initial transformation:" << std::endl;
  std::cout << "Translation: " << T_est.translation().transpose() << std::endl;
  std::cout << "Rotation: "
            << std::atan2(T_est.linear()(1, 0), T_est.linear()(0, 0)) * 180.0 /
                   M_PI
            << " degrees" << std::endl;

  // Vòng lặp ICP
  for (int iter = 0; iter < max_iterations; ++iter) {
    Eigen::Matrix<double, 3, 3> H = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Matrix<double, 3, 1> b = Eigen::Matrix<double, 3, 1>::Zero();
    double                      total_error     = 0.0;
    int                         num_valid_pairs = 0;

    // Tích lũy H và b cho tất cả các điểm
    for (size_t i = 0; i < source_points.size(); ++i) {
      Eigen::Matrix<double, 3, 3> Hi;
      Eigen::Matrix<double, 3, 1> bi;
      double                      ei;

      if (p2p_factor.linearize(target_points, source_points, target_tree, T_est,
                               i, rejector, &Hi, &bi, &ei)) {
        H += Hi;
        b += bi;
        total_error += ei;
        num_valid_pairs++;
      }
    }

    // Kiểm tra convergence
    if (num_valid_pairs < 20) { // Tăng số lượng điểm tối thiểu lên 20
      std::cout << "Too few valid pairs, stopping at iteration " << iter
                << std::endl;
      break;
    }

    double error_change = std::abs(total_error - prev_error);
    if (error_change < convergence_threshold && iter > 0) {
      std::cout << "Converged at iteration " << iter << std::endl;
      break;
    }
    prev_error = total_error;

    // Giải hệ phương trình để tìm delta
    Eigen::Vector3d delta = H.ldlt().solve(b);

    // Áp dụng step size
    delta *= p2p_factor.get_step_size();

    // Cập nhật transformation
    Eigen::Isometry2d delta_T     = Eigen::Isometry2d::Identity();
    double            delta_theta = delta(0);
    delta_T.linear() << std::cos(delta_theta), -std::sin(delta_theta),
        std::sin(delta_theta), std::cos(delta_theta);
    delta_T.translation() = delta.tail<2>();
    T_est                 = delta_T * T_est;

    // In thông tin iteration
    if (iter % 5 == 0) {
      std::cout << "Iteration " << iter << ":" << std::endl;
      std::cout << "  Error: " << total_error << std::endl;
      std::cout << "  Valid pairs: " << num_valid_pairs << std::endl;
      std::cout << "  Translation: " << T_est.translation().transpose()
                << std::endl;
      std::cout << "  Rotation: "
                << std::atan2(T_est.linear()(1, 0), T_est.linear()(0, 0)) *
                       180.0 / M_PI
                << " degrees" << std::endl;
    }
  }

  // So sánh kết quả với ground truth
  Eigen::Vector2d translation_error =
      T_est.translation() - T_true.translation();
  double rotation_error =
      std::abs(std::atan2(T_est.linear()(1, 0), T_est.linear()(0, 0)) -
               std::atan2(T_true.linear()(1, 0), T_true.linear()(0, 0)));
  rotation_error =
      std::min(rotation_error, 2 * M_PI - rotation_error) * 180.0 / M_PI;

  std::cout << "\nFinal results:" << std::endl;
  std::cout << "Translation error: " << translation_error.norm() << std::endl;
  std::cout << "Rotation error: " << rotation_error << " degrees" << std::endl;

  // Kiểm tra độ chính xác
  EXPECT_LT(translation_error.norm(), 0.1); // Sai số dịch chuyển < 0.1
  EXPECT_LT(rotation_error, 5.0);           // Sai số góc < 5 độ
}

// Test point-to-line ICP với hình vuông có nhiễu
TEST(BasicICPTest, PointToLineSineCurve) {
  // Create square shape for point-to-line ICP
  std::vector<Eigen::Vector2d> source_points;
  std::vector<Eigen::Vector2d> target_points;
  const int                    points_per_side = 25;
  const double                 side_length     = 2.0;

  // Create source points - square shape
  // Bottom edge: y = 0, x from 0 to side_length
  for (int i = 0; i < points_per_side; ++i) {
    double x = (double)i / (points_per_side - 1) * side_length;
    source_points.push_back(Eigen::Vector2d(x, 0.0));
  }
  // Right edge: x = side_length, y from 0 to side_length
  for (int i = 1; i < points_per_side; ++i) {
    double y = (double)i / (points_per_side - 1) * side_length;
    source_points.push_back(Eigen::Vector2d(side_length, y));
  }
  // Top edge: y = side_length, x from side_length to 0
  for (int i = 1; i < points_per_side; ++i) {
    double x = side_length - (double)i / (points_per_side - 1) * side_length;
    source_points.push_back(Eigen::Vector2d(x, side_length));
  }
  // Left edge: x = 0, y from side_length to 0
  for (int i = 1; i < points_per_side - 1; ++i) {
    double y = side_length - (double)i / (points_per_side - 1) * side_length;
    source_points.push_back(Eigen::Vector2d(0.0, y));
  }

  // Create ground truth transformation
  Eigen::Isometry2d T_true = Eigen::Isometry2d::Identity();
  double            angle  = 15.0 * M_PI / 180.0;
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.5, 0.3;

  // Add noise
  std::random_device rd;
  std::mt19937       gen(rd());
  const double       noise_std_dev = 0.05;

  for (const auto &p : source_points) {
    Eigen::Vector2d transformed = T_true * p;
    target_points.push_back(
        add_gaussian_noise(transformed, noise_std_dev, gen));
  }

  // Initialize KDTree for target points directly (no normals needed for
  // point-to-line ICP)
  using KDTree = KdTree<std::vector<Eigen::Vector2d>>;
  auto target_points_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KDTree target_tree(target_points_ptr);

  // Initialize point-to-line ICP factor
  PointToLineICPFactor<Huber>::Setting factor_setting;
  factor_setting.robust_kernel.c       = 0.5;
  factor_setting.distance_scale_factor = 0.5;
  std::vector<PointToLineICPFactor<Huber>> factors(
      source_points.size(), PointToLineICPFactor<Huber>(factor_setting));

  // Initialize other components
  DistanceRejector    rejector(0.5);
  TerminationCriteria criteria;
  criteria.max_iterations = 100;
  criteria.eps_rot        = 1e-4;
  criteria.eps_trans      = 1e-4;
  criteria.eps_error      = 1e-6;
  NullFactor           general_factor;
  PointFactorReduction reduction;

  // Run optimization with initial guess closer to ground truth
  Eigen::Isometry2d T_init = Eigen::Isometry2d::Identity();
  T_init.linear() << std::cos(18.0 * M_PI / 180.0),
      -std::sin(18.0 * M_PI / 180.0), std::sin(18.0 * M_PI / 180.0),
      std::cos(18.0 * M_PI / 180.0); // 18 degrees (closer to 15° ground truth)
  T_init.translation() << 0.55, 0.35; // Closer to (0.5, 0.3) ground truth

  // Run both optimizers for comparison
  std::cout << "\nRunning GaussNewton optimizer:" << std::endl;
  GaussNewtonOptimizer gn_optimizer;
  auto                 gn_result = gn_optimizer.optimize(
                      target_points, source_points, target_tree, rejector, criteria, reduction,
                      T_init, factors, general_factor);

  std::cout << "\nRunning LevenbergMarquardt optimizer:" << std::endl;
  LevenbergMarquardtOptimizer lm_optimizer;
  auto                        lm_result = lm_optimizer.optimize(
                             target_points, source_points, target_tree, rejector, criteria, reduction,
                             T_init, factors, general_factor);

  // Check GaussNewton results
  std::cout << "\nGaussNewton Results:" << std::endl;
  std::cout << "Converged: " << gn_result.converged << std::endl;
  std::cout << "Error: " << gn_result.error << std::endl;
  std::cout << "Iterations: " << gn_result.iterations << std::endl;
  std::cout << "Inliers: " << gn_result.num_inliers << std::endl;

  Eigen::Vector2d gn_trans_error =
      gn_result.T_target_source.translation() - T_true.translation();
  double gn_rot_error =
      std::abs(std::atan2(gn_result.T_target_source.linear()(1, 0),
                          gn_result.T_target_source.linear()(0, 0)) -
               angle);
  gn_rot_error = std::min(gn_rot_error, 2 * M_PI - gn_rot_error);

  std::cout << "Translation error: " << gn_trans_error.norm() << std::endl;
  std::cout << "Rotation error: " << gn_rot_error * 180.0 / M_PI << " degrees"
            << std::endl;

  // Check LevenbergMarquardt results
  std::cout << "\nLevenbergMarquardt Results:" << std::endl;
  std::cout << "Converged: " << lm_result.converged << std::endl;
  std::cout << "Error: " << lm_result.error << std::endl;
  std::cout << "Iterations: " << lm_result.iterations << std::endl;
  std::cout << "Inliers: " << lm_result.num_inliers << std::endl;

  Eigen::Vector2d lm_trans_error =
      lm_result.T_target_source.translation() - T_true.translation();
  double lm_rot_error =
      std::abs(std::atan2(lm_result.T_target_source.linear()(1, 0),
                          lm_result.T_target_source.linear()(0, 0)) -
               angle);
  lm_rot_error = std::min(lm_rot_error, 2 * M_PI - lm_rot_error);

  std::cout << "Translation error: " << lm_trans_error.norm() << std::endl;
  std::cout << "Rotation error: " << lm_rot_error * 180.0 / M_PI << " degrees"
            << std::endl;

  // Check accuracy (allow larger errors due to noise)
  EXPECT_TRUE(gn_result.converged);
  EXPECT_LT(gn_result.error, 0.1);
  EXPECT_LT(gn_trans_error.norm(), 0.2);
  EXPECT_LT(gn_rot_error, 0.2);

  EXPECT_TRUE(lm_result.converged);
  EXPECT_LT(lm_result.error, 0.1);
  EXPECT_LT(lm_trans_error.norm(), 0.2);
  EXPECT_LT(lm_rot_error, 0.2);
}

// Test GaussNewton optimizer with point-to-point ICP
TEST(BasicICPTest, GaussNewtonOptimizer) {
  // Create test data - square with more points
  std::vector<Eigen::Vector2d> source_points;
  const int                    num_points_per_side = 10; // 10 points per side
  const double                 step = 1.0 / (num_points_per_side - 1);

  // Bottom edge
  for (int i = 0; i < num_points_per_side; ++i) {
    source_points.push_back({i * step, 0.0});
  }
  // Right edge
  for (int i = 1; i < num_points_per_side; ++i) {
    source_points.push_back({1.0, i * step});
  }
  // Top edge
  for (int i = num_points_per_side - 2; i >= 0; --i) {
    source_points.push_back({i * step, 1.0});
  }
  // Left edge
  for (int i = num_points_per_side - 2; i > 0; --i) {
    source_points.push_back({0.0, i * step});
  }

  // Create ground truth transformation
  Eigen::Isometry2d T_true = Eigen::Isometry2d::Identity();
  double            angle  = 30.0 * M_PI / 180.0; // 30 degrees rotation
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.5, 0.3;

  // Create target points by applying transformation
  std::vector<Eigen::Vector2d> target_points;
  for (const auto &p : source_points) {
    target_points.push_back(T_true * p);
  }

  // Initialize KDTree for target points
  using KDTree = icp2d::KdTree<std::vector<Eigen::Vector2d>>;
  auto target_points_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KDTree target_tree(target_points_ptr);

  // Initialize point-to-point ICP factor
  icp2d::PointICPFactor::Setting factor_setting;
  factor_setting.weight    = 1.0;
  factor_setting.step_size = 0.3; // Reduce step size
  std::vector<icp2d::PointICPFactor> factors(
      source_points.size(), icp2d::PointICPFactor(factor_setting));

  // Initialize GaussNewton optimizer
  icp2d::GaussNewtonOptimizer optimizer;
  optimizer.verbose        = true;
  optimizer.max_iterations = 100;  // Increase iterations
  optimizer.lambda         = 1e-8; // Reduce damping

  // Initialize other components
  icp2d::DistanceRejector    rejector(0.5);
  icp2d::TerminationCriteria criteria;
  criteria.max_iterations = 100; // Increase iterations
  criteria.eps_rot        = 1e-4;
  criteria.eps_trans      = 1e-4;
  criteria.eps_error      = 1e-6;
  icp2d::NullFactor           general_factor;
  icp2d::PointFactorReduction reduction;

  // Run optimization with initial guess closer to ground truth
  Eigen::Isometry2d T_init = Eigen::Isometry2d::Identity();
  T_init.linear() << std::cos(28.0 * M_PI / 180.0),
      -std::sin(28.0 * M_PI / 180.0), std::sin(28.0 * M_PI / 180.0),
      std::cos(28.0 * M_PI / 180.0);  // 28 degrees
  T_init.translation() << 0.45, 0.25; // Closer to true translation

  auto result =
      optimizer.optimize(target_points, source_points, target_tree, rejector,
                         criteria, reduction, T_init, factors, general_factor);

  // Compare with ground truth
  Eigen::Vector2d trans_error =
      result.T_target_source.translation() - T_true.translation();
  double rot_error =
      std::abs(std::atan2(result.T_target_source.linear()(1, 0),
                          result.T_target_source.linear()(0, 0)) -
               angle);

  EXPECT_LT(trans_error.norm(), 0.1); // Translation error < 0.1
  EXPECT_LT(rot_error, 0.1);          // Rotation error < 0.1 rad (~5.7 degrees)

  // Check results
  EXPECT_TRUE(result.converged);

  std::cout << "GaussNewton Results:" << std::endl;
  std::cout << "Iterations: " << result.iterations << std::endl;
  std::cout << "Final error: " << result.error << std::endl;
  std::cout << "Translation error: " << trans_error.norm() << std::endl;
  std::cout << "Rotation error (rad): " << rot_error << std::endl;
}

// Test LevenbergMarquardt optimizer with point-to-point ICP
TEST(BasicICPTest, LevenbergMarquardtOptimizer) {
  // Create test data - square with more points
  std::vector<Eigen::Vector2d> source_points;
  const int                    num_points_per_side = 10; // 10 points per side
  const double                 step = 1.0 / (num_points_per_side - 1);

  // Bottom edge
  for (int i = 0; i < num_points_per_side; ++i) {
    source_points.push_back({i * step, 0.0});
  }
  // Right edge
  for (int i = 1; i < num_points_per_side; ++i) {
    source_points.push_back({1.0, i * step});
  }
  // Top edge
  for (int i = num_points_per_side - 2; i >= 0; --i) {
    source_points.push_back({i * step, 1.0});
  }
  // Left edge
  for (int i = num_points_per_side - 2; i > 0; --i) {
    source_points.push_back({0.0, i * step});
  }

  // Create ground truth transformation
  Eigen::Isometry2d T_true = Eigen::Isometry2d::Identity();
  double            angle  = 45.0 * M_PI / 180.0; // 45 degrees rotation
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.8, -0.5;

  // Add noise to target points
  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::normal_distribution<double> noise(0.0, 0.05);

  std::vector<Eigen::Vector2d> target_points;
  for (const auto &p : source_points) {
    Eigen::Vector2d noisy_point = T_true * p;
    noisy_point += Eigen::Vector2d(noise(gen), noise(gen));
    target_points.push_back(noisy_point);
  }

  // Initialize KDTree for target points
  using KDTree = icp2d::KdTree<std::vector<Eigen::Vector2d>>;
  auto target_points_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KDTree target_tree(target_points_ptr);

  // Initialize point-to-point ICP factor
  icp2d::PointICPFactor::Setting factor_setting;
  factor_setting.weight    = 1.0;
  factor_setting.step_size = 0.3; // Reduce step size
  std::vector<icp2d::PointICPFactor> factors(
      source_points.size(), icp2d::PointICPFactor(factor_setting));

  // Initialize LevenbergMarquardt optimizer
  icp2d::LevenbergMarquardtOptimizer optimizer;
  optimizer.max_iterations       = 100; // Increase iterations
  optimizer.max_inner_iterations = 20;
  optimizer.init_lambda          = 1e-4;
  optimizer.lambda_factor        = 5.0;

  // Initialize other components
  icp2d::DistanceRejector    rejector(0.5);
  icp2d::TerminationCriteria criteria;
  criteria.max_iterations = 100; // Increase iterations
  criteria.eps_rot        = 1e-4;
  criteria.eps_trans      = 1e-4;
  criteria.eps_error      = 1e-6;
  icp2d::NullFactor           general_factor;
  icp2d::PointFactorReduction reduction;

  // Run optimization with initial guess closer to ground truth
  Eigen::Isometry2d T_init = Eigen::Isometry2d::Identity();
  T_init.linear() << std::cos(42.0 * M_PI / 180.0),
      -std::sin(42.0 * M_PI / 180.0), std::sin(42.0 * M_PI / 180.0),
      std::cos(42.0 * M_PI / 180.0);   // 42 degrees
  T_init.translation() << 0.75, -0.45; // Closer to true translation

  auto result =
      optimizer.optimize(target_points, source_points, target_tree, rejector,
                         criteria, reduction, T_init, factors, general_factor);

  // Check results
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.error, 0.1); // Allow larger error due to noise

  // Compare with ground truth
  Eigen::Vector2d trans_error =
      result.T_target_source.translation() - T_true.translation();
  double rot_error =
      std::abs(std::atan2(result.T_target_source.linear()(1, 0),
                          result.T_target_source.linear()(0, 0)) -
               angle);

  EXPECT_LT(trans_error.norm(), 0.2); // Allow larger error due to noise
  EXPECT_LT(rot_error, 0.2);          // Allow larger error due to noise

  std::cout << "LevenbergMarquardt Results:" << std::endl;
  std::cout << "Iterations: " << result.iterations << std::endl;
  std::cout << "Final error: " << result.error << std::endl;
  std::cout << "Translation error: " << trans_error.norm() << std::endl;
  std::cout << "Rotation error (rad): " << rot_error << std::endl;
}

// Test point cloud traits system
TEST(BasicICPTest, PointCloudTraits) {
  using namespace icp2d;

  // Test std::vector<Eigen::Vector2d> traits
  std::vector<Eigen::Vector2d> vec_points;
  vec_points.push_back(Eigen::Vector2d(1.0, 0.0));
  vec_points.push_back(Eigen::Vector2d(0.0, 1.0));

  EXPECT_EQ(traits::point(vec_points, 0), Eigen::Vector2d(1.0, 0.0));
  EXPECT_EQ(traits::point(vec_points, 1), Eigen::Vector2d(0.0, 1.0));
  EXPECT_THROW(traits::normal(vec_points, 0), std::runtime_error);

  // Test that normal access throws for std::vector<Eigen::Vector2d>
  EXPECT_THROW(traits::normal(vec_points, 0), std::runtime_error);
}

// Test point-to-line ICP with degenerate cases
TEST(BasicICPTest, PointToLineDegenerateCases) {
  // Test with collinear points
  std::vector<Eigen::Vector2d> source_points;
  const int                    num_points = 10;
  for (int i = 0; i < num_points; ++i) {
    source_points.push_back(Eigen::Vector2d(i * 0.1, 0.0));
  }

  // Create target by rotating 90 degrees
  Eigen::Isometry2d T_true = Eigen::Isometry2d::Identity();
  double            angle  = M_PI / 2.0; // 90 degrees
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.1, 0.2;

  std::vector<Eigen::Vector2d> target_points;
  for (const auto &p : source_points) {
    target_points.push_back(T_true * p);
  }

  // Initialize ICP components directly with target points
  using KDTree = KdTree<std::vector<Eigen::Vector2d>>;
  auto target_points_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KDTree target_tree(target_points_ptr);

  PointToLineICPFactor<Huber>::Setting factor_setting;
  factor_setting.robust_kernel.c       = 0.5;
  factor_setting.distance_scale_factor = 0.3;
  std::vector<PointToLineICPFactor<Huber>> factors(
      source_points.size(), PointToLineICPFactor<Huber>(factor_setting));

  DistanceRejector    rejector(0.5);
  TerminationCriteria criteria;
  criteria.max_iterations = 100;
  criteria.eps_rot        = 1e-4;
  criteria.eps_trans      = 1e-4;
  criteria.eps_error      = 1e-6;
  NullFactor           general_factor;
  PointFactorReduction reduction;

  // Test with initial guess far from ground truth
  Eigen::Isometry2d T_init = Eigen::Isometry2d::Identity();
  T_init.linear() << std::cos(-angle), -std::sin(-angle), std::sin(-angle),
      std::cos(-angle); // -90 degrees
  T_init.translation() << -0.1, -0.2;

  // Try both optimizers
  GaussNewtonOptimizer gn_optimizer;
  auto                 gn_result = gn_optimizer.optimize(
                      target_points, source_points, target_tree, rejector, criteria, reduction,
                      T_init, factors, general_factor);

  LevenbergMarquardtOptimizer lm_optimizer;
  auto                        lm_result = lm_optimizer.optimize(
                             target_points, source_points, target_tree, rejector, criteria, reduction,
                             T_init, factors, general_factor);

  // Check results - should converge despite degenerate case
  EXPECT_TRUE(gn_result.converged);
  EXPECT_TRUE(lm_result.converged);

  // Errors should be small but may not be zero due to collinearity
  EXPECT_LT(gn_result.error, 0.1);
  EXPECT_LT(lm_result.error, 0.1);

  // Print detailed results
  std::cout << "\nDegenerate Case Results:" << std::endl;
  std::cout << "GaussNewton error: " << gn_result.error << std::endl;
  std::cout << "LevenbergMarquardt error: " << lm_result.error << std::endl;

  // Test with circle points (another degenerate case)
  source_points.clear();
  const int    num_circle_points = 20;
  const double radius            = 1.0;
  for (int i = 0; i < num_circle_points; ++i) {
    double theta = 2.0 * M_PI * i / num_circle_points;
    source_points.push_back(
        Eigen::Vector2d(radius * std::cos(theta), radius * std::sin(theta)));
  }

  // Create target by rotating 45 degrees and translating
  T_true = Eigen::Isometry2d::Identity();
  angle  = M_PI / 4.0; // 45 degrees
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.5, -0.3;

  target_points.clear();
  for (const auto &p : source_points) {
    target_points.push_back(T_true * p);
  }

  // Re-initialize KDTree for circle points
  target_points_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KDTree target_tree_circle(target_points_ptr);

  // Reset factors for new point count
  factors.clear();
  factors.resize(source_points.size(), PointToLineICPFactor(factor_setting));

  // Test with bad initial guess
  T_init = Eigen::Isometry2d::Identity();
  T_init.linear() << std::cos(M_PI), -std::sin(M_PI), std::sin(M_PI),
      std::cos(M_PI); // 180 degrees (completely wrong)
  T_init.translation() << -1.0, 1.0;

  gn_result = gn_optimizer.optimize(target_points, source_points,
                                    target_tree_circle, rejector, criteria,
                                    reduction, T_init, factors, general_factor);

  lm_result = lm_optimizer.optimize(target_points, source_points,
                                    target_tree_circle, rejector, criteria,
                                    reduction, T_init, factors, general_factor);

  // Check results for circle case
  EXPECT_TRUE(gn_result.converged);
  EXPECT_TRUE(lm_result.converged);
  EXPECT_LT(gn_result.error, 0.1);
  EXPECT_LT(lm_result.error, 0.1);

  std::cout << "\nCircle Case Results:" << std::endl;
  std::cout << "GaussNewton error: " << gn_result.error << std::endl;
  std::cout << "LevenbergMarquardt error: " << lm_result.error << std::endl;

  // Compare transformations with ground truth
  Eigen::Vector2d gn_trans_error =
      gn_result.T_target_source.translation() - T_true.translation();
  double gn_rot_error =
      std::abs(std::atan2(gn_result.T_target_source.linear()(1, 0),
                          gn_result.T_target_source.linear()(0, 0)) -
               angle);
  gn_rot_error = std::min(gn_rot_error, 2 * M_PI - gn_rot_error);

  Eigen::Vector2d lm_trans_error =
      lm_result.T_target_source.translation() - T_true.translation();
  double lm_rot_error =
      std::abs(std::atan2(lm_result.T_target_source.linear()(1, 0),
                          lm_result.T_target_source.linear()(0, 0)) -
               angle);
  lm_rot_error = std::min(lm_rot_error, 2 * M_PI - lm_rot_error);

  // Check accuracy (allow larger errors due to degeneracy)
  EXPECT_LT(gn_trans_error.norm(), 0.2);
  EXPECT_LT(gn_rot_error, 0.2);
  EXPECT_LT(lm_trans_error.norm(), 0.2);
  EXPECT_LT(lm_rot_error, 0.2);

  std::cout << "GaussNewton transformation error:" << std::endl;
  std::cout << "  Translation: " << gn_trans_error.norm() << std::endl;
  std::cout << "  Rotation: " << gn_rot_error * 180.0 / M_PI << " degrees"
            << std::endl;
  std::cout << "LevenbergMarquardt transformation error:" << std::endl;
  std::cout << "  Translation: " << lm_trans_error.norm() << std::endl;
  std::cout << "  Rotation: " << lm_rot_error * 180.0 / M_PI << " degrees"
            << std::endl;
}

// Test point-to-line ICP with different robust kernels
TEST(BasicICPTest, PointToLineRobustKernels) {
  using namespace traits;

  // Create sine curve test case with outliers
  std::vector<Eigen::Vector2d> source_points;
  const int                    num_points = 100;
  const double                 x_min      = -M_PI;
  const double                 x_max      = M_PI;
  const double                 dx         = (x_max - x_min) / (num_points - 1);

  std::random_device                     rd;
  std::mt19937                           gen(rd());
  std::normal_distribution<double>       noise(0.0, 0.05);
  std::uniform_real_distribution<double> outlier_dist(-1.0, 1.0);

  // Generate source points (sine curve with noise and outliers)
  for (int i = 0; i < num_points; ++i) {
    const double x = x_min + i * dx;
    const double y = std::sin(x);

    // Add 10% outliers
    if (i % 10 == 0) {
      source_points.push_back(Eigen::Vector2d(x, y + outlier_dist(gen)));
    } else {
      source_points.push_back(Eigen::Vector2d(x, y + noise(gen)));
    }
  }

  // Create ground truth transform
  Eigen::Isometry2d T_true = Eigen::Isometry2d::Identity();
  const double      angle  = 15.0 * M_PI / 180.0; // 15 degrees
  T_true.linear() << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  T_true.translation() << 0.5, 0.3;

  // Transform source points
  std::vector<Eigen::Vector2d> target_points;
  for (const auto &p : source_points) {
    target_points.push_back(T_true * p);
  }

  // No need to estimate normals for point-to-line ICP

  // Initial guess closer to ground truth (18 degrees vs 15 degrees)
  Eigen::Isometry2d T_init     = Eigen::Isometry2d::Identity();
  const double      init_angle = 18.0 * M_PI / 180.0;
  T_init.linear() << std::cos(init_angle), -std::sin(init_angle),
      std::sin(init_angle), std::cos(init_angle);
  T_init.translation() << 0.55, 0.35; // Closer to (0.5, 0.3) ground truth

  // Create registration with current kernel
  LevenbergMarquardtOptimizer optimizer;
  optimizer.max_iterations = 100;
  // optimizer.lambda         = 1e-6;

  TerminationCriteria criteria;
  criteria.max_iterations = 50;
  criteria.eps_rot        = 1e-4;
  criteria.eps_trans      = 1e-4;
  criteria.eps_error      = 1e-6;

  DistanceRejector     rejector(0.5);
  PointFactorReduction reduction;

  std::vector<PointToLineICPFactor<Huber>> factors(source_points.size());

  // Run registration
  Eigen::Isometry2d T_est = T_init;
  auto              target_ptr =
      std::make_shared<std::vector<Eigen::Vector2d>>(target_points);
  KdTree<std::vector<Eigen::Vector2d>> target_tree(target_ptr);

  NullFactor general_factor;
  auto       result =
      optimizer.optimize(target_points, source_points, target_tree, rejector,
                         criteria, reduction, T_init, factors, general_factor);

  // Check results
  const Eigen::Vector2d translation_error =
      result.T_target_source.translation() - T_true.translation();
  const double rotation_error = std::abs(
      std::acos(Eigen::Rotation2Dd(result.T_target_source.linear()).angle() -
                Eigen::Rotation2Dd(T_true.linear()).angle()));

  std::cout << "Translation error: " << translation_error.norm() << std::endl;
  std::cout << "Rotation error: " << rotation_error * 180.0 / M_PI << " degrees"
            << std::endl;
}

} // namespace icp2d

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
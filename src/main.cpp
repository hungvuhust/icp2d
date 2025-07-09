#include <icp2d/registration/optimizer.hpp>
#include <icp2d/registration/registration.hpp>
#include <icp2d/registration/registration_result.hpp>
#include <icp2d/registration/rejector.hpp>
#include <icp2d/registration/termination_criteria.hpp>
#include <icp2d/util/general_factor.hpp>
#include <icp2d/util/line_icp_factor.hpp>
#include <icp2d/util/point_icp_factor.hpp>
#include <icp2d/util/robust_kernel.hpp>
#include <iostream>

int main(int argc, char **argv) {
  icp2d::GaussNewtonOptimizer optimizer;
  std::cout << "Hello, World!" << std::endl;
  return 0;
}
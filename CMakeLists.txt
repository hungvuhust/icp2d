cmake_minimum_required(VERSION 3.8)
project(icp2d)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenMP REQUIRED)

# Add Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add Google Test
find_package(GTest REQUIRED)

# Add include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)

# Add executables
add_executable(main src/main.cpp)
target_link_libraries(main Eigen3::Eigen OpenMP::OpenMP_CXX)

# Add tests
add_executable(test_kdtree test/test_kdtree.cpp)
target_link_libraries(test_kdtree Eigen3::Eigen OpenMP::OpenMP_CXX GTest::GTest GTest::Main)

add_executable(test_kdtree_omp test/test_kdtree_omp.cpp)
target_link_libraries(test_kdtree_omp Eigen3::Eigen OpenMP::OpenMP_CXX)

add_executable(test_basic_icp test/test_basic_icp.cpp)
target_link_libraries(test_basic_icp Eigen3::Eigen OpenMP::OpenMP_CXX GTest::GTest GTest::Main)

# Add test target
enable_testing()
add_test(NAME test_kdtree COMMAND test_kdtree)
add_test(NAME test_kdtree_omp COMMAND test_kdtree_omp)
add_test(NAME test_basic_icp COMMAND test_basic_icp)

# Install
install(TARGETS main
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include
)

ament_package()

cmake_minimum_required(VERSION 3.8)
project(icp2d)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(main src/main.cpp)

# Add test executable
add_executable(test_kdtree test/test_kdtree.cpp)
target_link_libraries(test_kdtree ${EIGEN3_LIBRARIES})

# Enable testing
enable_testing()
add_test(NAME kdtree_test COMMAND test_kdtree)

install(TARGETS main test_kdtree
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()

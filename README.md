# ICP 2D - Iterative Closest Point for 2D

Thư viện ICP (Iterative Closest Point) được thiết kế cho bài toán alignment point cloud 2D.

## Tính năng

- **KDTree 2D**: Cấu trúc dữ liệu cho nearest neighbor search hiệu quả trong không gian 2D
- **Projection Methods**: Hỗ trợ AxisAlignedProjection và NormalProjection
- **Lie Algebra**: Implement SE(2) cho rigid transformation 2D 
- **Optimization**: Gauss-Newton và Levenberg-Marquardt optimizers

## Cấu trúc dự án

```
icp2d/
├── include/icp2d/
│   ├── core/
│   │   ├── kdtree.hpp          # KDTree implementation for 2D
│   │   ├── projection.hpp      # Projection methods
│   │   ├── traits.hpp          # Type traits
│   │   └── knn_result.hpp      # KNN search results
│   ├── registration/
│   │   ├── optimizaer.hpp      # Optimization algorithms
│   │   ├── registration_result.hpp  # Registration results
│   │   └── termination_criteria.hpp # Convergence criteria
│   └── util/
│       └── lie.hpp             # SE(2) Lie algebra
├── src/
│   └── main.cpp                # Main application
├── test/
│   └── test_kdtree.cpp         # KDTree tests
└── CMakeLists.txt
```

## Dependencies

- **Eigen3**: Ma trận và vector operations
- **CMake**: Build system (>= 3.8)
- **C++17**: Standard required

## Build và chạy

### Build thường
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build và chạy test
```bash
# Cách 1: Sử dụng script
chmod +x run_tests.sh
./run_tests.sh

# Cách 2: Manual
mkdir build && cd build
cmake ..
make test_kdtree
./test_kdtree
```

### Chạy ứng dụng chính
```bash
cd build
./main
```

## Sử dụng KDTree 2D

```cpp
#include <icp2d/core/kdtree.hpp>

// Tạo point cloud 2D
std::vector<Eigen::Vector2d> points = {
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}
};

// Build KDTree
icp2d::UnsafeKdTree<std::vector<Eigen::Vector2d>> kdtree(points);

// Nearest neighbor search
Eigen::Vector2d query(0.1, 0.1);
size_t nearest_idx;
double nearest_dist;
kdtree.nearest_neighbor_search(query, &nearest_idx, &nearest_dist);

// K-nearest neighbors search
const int k = 3;
std::vector<size_t> k_indices(k);
std::vector<double> k_distances(k);
kdtree.knn_search(query, k, k_indices.data(), k_distances.data());
```

## Test Coverage

File `test/test_kdtree.cpp` bao gồm các test sau:

1. **Basic Functionality**: Test cơ bản với nearest neighbor search
2. **KNN Search**: Test k-nearest neighbors với dynamic memory
3. **Static KNN Search**: Test với fixed memory allocation
4. **Random Points**: Verification với brute force search trên 1000 điểm ngẫu nhiên
5. **Different Projections**: Test AxisAlignedProjection và NormalProjection
6. **Safe KdTree**: Test phiên bản thread-safe với shared_ptr

## Performance

- **Complexity**: O(log n) cho nearest neighbor search
- **Memory**: Tối ưu với static allocation cho KNN search nhỏ
- **Accuracy**: Verified với brute force trên random datasets

## Known Issues

- Hiện tại chỉ hỗ trợ 2D points (Eigen::Vector2d)
- Cần implement traits cho point cloud types khác nhau

## Contributing

1. Tạo feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Create pull request

## License

MIT License - xem file LICENSE để biết chi tiết. 
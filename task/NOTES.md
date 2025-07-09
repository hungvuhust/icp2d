# NOTES - Ghi chú dự án ICP 2D

## Thông tin chung
- **Tên dự án**: ICP 2D (Iterative Closest Point for 2D)
- **Ngôn ngữ**: C++
- **Dependencies**: Eigen3
- **Build system**: CMake

## Cấu trúc Lie Algebra đã implement
### SO(2) - Special Orthogonal Group 2D
- Nhóm ma trận xoay 2D
- Parameterized bằng góc θ
- exp: ℝ → SO(2), θ ↦ [cos θ, -sin θ; sin θ, cos θ]

### SE(2) - Special Euclidean Group 2D  
- Nhóm biến đổi cứng 2D (rigid transformation)
- Parameterized bằng twist vector [tx, ty, θ]
- Bao gồm cả xoay và tịnh tiến

## Ý tưởng triển khai
### ICP Algorithm variants:
1. **Point-to-Point ICP**: Minimize khoảng cách Euclidean
2. **Point-to-Line ICP**: Minimize khoảng cách từ điểm đến đường thẳng (chính xác hơn)

### Robust Cost Functions:
- Least Squares (L2): Nhạy cảm với outliers
- Huber: Kết hợp L2 và L1, robust hơn
- Cauchy: Rất robust với outliers

### Optimization Methods:
- Gauss-Newton: Nhanh, cần good initialization
- Levenberg-Marquardt: Robust hơn, slower

## Cấu trúc file dự kiến
```
include/icp2d/
├── core/
│   ├── point_cloud.hpp
│   ├── icp_solver.hpp
│   └── cost_functions.hpp
├── optimization/
│   ├── gauss_newton.hpp
│   └── levenberg_marquardt.hpp
├── io/
│   ├── file_reader.hpp
│   └── file_writer.hpp
└── util/
    ├── lie.hpp ✅
    └── visualization.hpp
```

## Benchmark datasets
- Stanford Bunny (2D slice)
- Synthetic data với known ground truth
- Real sensor data (LiDAR scans)

---
*Ngày tạo: 2024-12-19* 
#!/bin/bash

echo "Building and running KDTree tests..."
echo "===================================="

# # Delete build directory if it exists
# if [ -d "build" ]; then
#     rm -rf build
# fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ --no-warn-unused-cli ..

# Build the project
echo "Building..."
cmake --build . --config Release

# Check if test executables exist
if [ ! -f "test_kdtree" ]; then
    echo " Error: test_kdtree executable not found!"
    exit 1
fi

if [ ! -f "test_kdtree_omp" ]; then
    echo " Error: test_kdtree_omp executable not found!"
    exit 1
fi

if [ ! -f "test_basic_icp" ]; then
    echo " Error: test_basic_icp executable not found!"
    exit 1
fi

# Run the sequential tests
echo ""
echo "Running sequential KDTree tests..."
echo "=================================="
./test_kdtree

# Check the result
if [ $? -ne 0 ]; then
    echo ""
    echo "Sequential tests failed!"
    exit 1
fi

# Run the OpenMP tests
echo ""
echo "Running OpenMP KDTree tests..."
echo "=============================="
./test_kdtree_omp

# Check the result
if [ $? -ne 0 ]; then
    echo ""
    echo " OpenMP tests failed!"
    exit 1
fi

# Run Basic ICP tests
echo ""
echo "Running Basic ICP tests..."
echo "========================="
./test_basic_icp

# Check the result
if [ $? -eq 0 ]; then
    echo ""
    echo " All tests completed successfully!"
    echo " KDTree (Sequential): PASSED"
    echo " KDTree (OpenMP): PASSED" 
    echo " Basic ICP: PASSED"
else
    echo ""
    echo " Basic ICP tests failed!"
    exit 1
fi 
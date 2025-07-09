#!/bin/bash

echo "Building and running KDTree tests..."
echo "===================================="

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

# Check if test executable exists
if [ ! -f "test_kdtree" ]; then
    echo "❌ Error: test_kdtree executable not found!"
    exit 1
fi

# Run the tests
echo ""
echo "Running tests..."
echo "=================="
./test_kdtree

# Check the result
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 All tests completed successfully!"
else
    echo ""
    echo "❌ Some tests failed!"
    exit 1
fi 
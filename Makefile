.PHONY: all clean test build

# Default target
all: build

# Create build directory and build project
build:
	@echo "Building project..."
	@mkdir -p build
	@cd build && cmake .. && make -j$$(nproc)

# Build and run tests
test: build
	@echo "Running tests..."
	@cd build && ./test_kdtree

# Clean build directory
clean:
	@echo "Cleaning build directory..."
	@rm -rf build

# Quick test using script
test-script:
	@chmod +x run_tests.sh
	@./run_tests.sh

# Run main application
run: build
	@cd build && ./main

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the project (default)"
	@echo "  build       - Build the project"
	@echo "  test        - Build and run tests"
	@echo "  test-script - Run tests using script"
	@echo "  run         - Run main application"
	@echo "  clean       - Clean build directory"
	@echo "  help        - Show this help message" 
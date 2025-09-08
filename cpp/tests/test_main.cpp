#include <iostream>
#include <cassert>
#include <cmath>

// Test declarations
void test_oscillator();
void test_bin_handler();
void test_splines();
void test_mc_load();
void test_spline_syst();

// Simple test framework
int passed_tests = 0;
int total_tests = 0;

#define ASSERT_TRUE(condition) do { \
    total_tests++; \
    if (condition) { \
        passed_tests++; \
        std::cout << "✓ Test passed: " << #condition << std::endl; \
    } else { \
        std::cout << "✗ Test failed: " << #condition << std::endl; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tolerance) do { \
    total_tests++; \
    if (std::abs((a) - (b)) < (tolerance)) { \
        passed_tests++; \
        std::cout << "✓ Test passed: " << #a << " ≈ " << #b << " (diff: " << std::abs((a) - (b)) << ")" << std::endl; \
    } else { \
        std::cout << "✗ Test failed: " << #a << " ≈ " << #b << " (diff: " << std::abs((a) - (b)) << " > " << (tolerance) << ")" << std::endl; \
    } \
} while(0)

#define ASSERT_VECTOR_NEAR(v1, v2, tolerance) do { \
    total_tests++; \
    bool vectors_equal = true; \
    if ((v1).size() != (v2).size()) { \
        vectors_equal = false; \
    } else { \
        for (int i = 0; i < (v1).size(); ++i) { \
            if (std::abs((v1)[i] - (v2)[i]) > (tolerance)) { \
                vectors_equal = false; \
                break; \
            } \
        } \
    } \
    if (vectors_equal) { \
        passed_tests++; \
        std::cout << "✓ Test passed: vectors are approximately equal" << std::endl; \
    } else { \
        std::cout << "✗ Test failed: vectors are not approximately equal" << std::endl; \
    } \
} while(0)

int main() {
    std::cout << "Running C++ MAGPY tests..." << std::endl;
    std::cout << "=========================" << std::endl;
    
    std::cout << "\n--- Testing Oscillator ---" << std::endl;
    test_oscillator();
    
    std::cout << "\n--- Testing Bin Handler ---" << std::endl;
    test_bin_handler();
    
    std::cout << "\n--- Testing Splines ---" << std::endl;
    test_splines();
    
    std::cout << "\n--- Testing MC Load ---" << std::endl;
    test_mc_load();
    
    std::cout << "\n--- Testing Spline Syst ---" << std::endl;
    test_spline_syst();
    
    std::cout << "\n=========================" << std::endl;
    std::cout << "Tests completed: " << passed_tests << "/" << total_tests << " passed" << std::endl;
    
    if (passed_tests == total_tests) {
        std::cout << "🎉 All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed." << std::endl;
        return 1;
    }
}

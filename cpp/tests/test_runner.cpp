#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "test_functions.h"

// Global test counters
int passed_tests = 0;
int total_tests = 0;

// Test assertion functions
void ASSERT_TRUE(bool condition) {
    if (!condition) {
        throw std::runtime_error("Assertion failed: condition is false");
    }
}

void ASSERT_NEAR(double actual, double expected, double tolerance) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error("Assertion failed: values not within tolerance");
    }
}

void ASSERT_VECTOR_NEAR(const Eigen::VectorXd& actual, const Eigen::VectorXd& expected, double tolerance) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Assertion failed: vector sizes don't match");
    }
    for (int i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], tolerance);
    }
}

int main() {
    std::cout << "=== MAGPY C++ TEST SUITE ===" << std::endl;
    std::cout << "Running critical tests to match Python behavior exactly..." << std::endl << std::endl;
    
    try {
        // Run all oscillator tests - THE MOST CRITICAL ONES
        std::cout << "--- OSCILLATOR TESTS ---" << std::endl;
        magpy::test_oscillator_creation();
        magpy::test_oscillator_energy_setting();
        
        // THE CRITICAL TEST - must pass exactly!
        std::cout << std::endl << "*** RUNNING CRITICAL TEST ***" << std::endl;
        magpy::test_oscillator_probabilities();
        std::cout << "*** CRITICAL TEST COMPLETED ***" << std::endl << std::endl;
        
        magpy::test_oscillator_neutrino_types();
        magpy::test_oscillator_consistency();
        magpy::test_oscillator_performance();
        
        std::cout << std::endl << "=== TEST SUMMARY ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << total_tests << " tests" << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "🎉 ALL TESTS PASSED! C++ implementation matches Python exactly!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ SOME TESTS FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ TEST FAILED WITH EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}

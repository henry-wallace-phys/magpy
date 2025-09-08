#include "oscillator.h"
#include "mc_event.h"
#include "test_functions.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cassert>

extern int passed_tests, total_tests;
extern void ASSERT_TRUE(bool);
extern void ASSERT_NEAR(double, double, double);
extern void ASSERT_VECTOR_NEAR(const Eigen::VectorXd&, const Eigen::VectorXd&, double);

namespace magpy {

// Test data from Python implementation - CRITICAL for exact compatibility
struct TestData {
    double s12, s13, s23, delta_cp, dmsq21, dmsq31;
    std::vector<double> expected;
};

std::vector<TestData> get_test_data() {
    return {
        {
            0.31, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3,
            {0.025859313064903713, 0.40276761016651386, 0.1739814018841045, 
             0.05528143409429227, 0.005903280202548378, 0.8964759326330115}
        }
    };
}

void test_oscillator_creation() {
    std::cout << "Testing oscillator creation..." << std::endl;
    Oscillator osc(1300, 0.5, 3.0, 1000);
    std::cout << "✓ Oscillator created successfully" << std::endl;
    passed_tests++;
    total_tests++;
}

void test_oscillator_energy_setting() {
    std::cout << "Testing oscillator energy setting..." << std::endl;
    Oscillator osc(1300, 0.5, 3.0, 1000);
    
    // Create test energies
    Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(100, 0.1, 10.0);
    Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(100, 14);  // muon neutrino
    Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(100, 14);    // muon neutrino
    
    osc.set_energy_osc(energies, start_nu, end_nu);
    std::cout << "✓ Energy setting successful" << std::endl;
    passed_tests++;
    total_tests++;
}

void test_oscillator_probabilities() {
    std::cout << "Testing oscillator probabilities - CRITICAL TEST..." << std::endl;
    
    auto test_cases = get_test_data();
    const double tolerance = 1e-10;
    
    for (const auto& test_case : test_cases) {
        Oscillator osc(1300, 0.5, 3.0, 1000);
        
        // Create test energies matching Python
        Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(6, 0.5, 3.0);
        std::vector<int> nu_in = {14, 14, 14, 14, 14, 14};  // muon neutrino
        std::vector<int> nu_out = {12, 14, 16, 12, 14, 16}; // e, mu, tau, e, mu, tau
        
        // Convert to Eigen
        Eigen::VectorXi start_nu(6), end_nu(6);
        for (int i = 0; i < 6; ++i) {
            start_nu[i] = nu_in[i];
            end_nu[i] = nu_out[i];
        }
        
        osc.set_energy_osc(energies, start_nu, end_nu);
        
        // Set oscillation parameters
        Eigen::VectorXd osc_params(6);
        osc_params << test_case.s12, test_case.s13, test_case.s23, 
                      test_case.delta_cp, test_case.dmsq21, test_case.dmsq31;
        
        // Calculate probabilities
        Eigen::VectorXd results = osc.calc_probability(osc_params);
        
        // Debug output - show actual vs expected
        std::cout << "Comparing results:" << std::endl;
        for (int i = 0; i < std::min(6, (int)test_case.expected.size()); ++i) {
            std::cout << "Event " << i << ": Got " << results[i] 
                      << ", Expected " << test_case.expected[i] 
                      << ", Diff " << std::abs(results[i] - test_case.expected[i]) << std::endl;
        }
        
        // Compare with expected results
        for (int i = 0; i < std::min(6, (int)test_case.expected.size()); ++i) {
            ASSERT_NEAR(results[i], test_case.expected[i], tolerance);
        }
    }
    
    std::cout << "✓ Probability test PASSED!" << std::endl;
    passed_tests++;
    total_tests++;
}

void test_oscillator_neutrino_types() {
    std::cout << "Testing oscillator neutrino types..." << std::endl;
    
    Oscillator osc(1300, 0.5, 3.0, 1000);
    
    // Test all neutrino types
    Eigen::VectorXd energies = Eigen::VectorXd::Constant(9, 1.0);
    std::vector<int> nu_types = {12, 14, 16, -12, -14, -16, 12, 14, 16};
    std::vector<int> nu_out_types = {12, 14, 16, -12, -14, -16, 14, 16, 12};
    
    Eigen::VectorXi start_nu(9), end_nu(9);
    for (int i = 0; i < 9; ++i) {
        start_nu[i] = nu_types[i];
        end_nu[i] = nu_out_types[i];
    }
    
    osc.set_energy_osc(energies, start_nu, end_nu);
    
    Eigen::VectorXd osc_params(6);
    osc_params << 0.31, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3;
    
    Eigen::VectorXd results = osc.calc_probability(osc_params);
    
    // Check that all probabilities are valid (0 <= P <= 1)
    for (int i = 0; i < results.size(); ++i) {
        ASSERT_TRUE(results[i] >= 0.0);
        ASSERT_TRUE(results[i] <= 1.0);
    }
    
    std::cout << "✓ Neutrino types test passed" << std::endl;
    passed_tests++;
    total_tests++;
}

void test_oscillator_consistency() {
    std::cout << "Testing oscillator consistency..." << std::endl;
    
    Oscillator osc(1300, 0.5, 3.0, 1000);
    
    Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(10, 0.5, 5.0);
    Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(10, 14);
    Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(10, 12);
    
    osc.set_energy_osc(energies, start_nu, end_nu);
    
    Eigen::VectorXd osc_params(6);
    osc_params << 0.31, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3;
    
    // Calculate twice to ensure consistency
    Eigen::VectorXd results1 = osc.calc_probability(osc_params);
    Eigen::VectorXd results2 = osc.calc_probability(osc_params);
    
    ASSERT_VECTOR_NEAR(results1, results2, 1e-15);
    
    std::cout << "✓ Consistency test passed" << std::endl;
    passed_tests++;
    total_tests++;
}

void test_oscillator_performance() {
    std::cout << "Testing oscillator performance..." << std::endl;
    
    Oscillator osc(1300, 0.5, 3.0, 1000);
    
    // Large-scale performance test
    const int n_events = 10000;
    Eigen::VectorXd energies = Eigen::VectorXd::Random(n_events).array().abs() * 5.0 + 0.1;
    Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, 14);
    Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, 12);
    
    osc.set_energy_osc(energies, start_nu, end_nu);
    
    Eigen::VectorXd osc_params(6);
    osc_params << 0.31, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3;
    
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd results = osc.calc_probability(osc_params);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "✓ Performance test: " << n_events << " events in " 
              << duration.count() << " μs" << std::endl;
    
    // Verify results are reasonable
    for (int i = 0; i < std::min(100, (int)results.size()); ++i) {
        ASSERT_TRUE(results[i] >= 0.0);
        ASSERT_TRUE(results[i] <= 1.0);
    }
    
    std::cout << "✓ Performance test passed" << std::endl;
    passed_tests++;
    total_tests++;
}

} // namespace magpy

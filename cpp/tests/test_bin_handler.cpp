#include "bin_handler.h"
#include <iostream>
#include <chrono>

extern int passed_tests, total_tests;
extern void ASSERT_TRUE(bool);
extern void ASSERT_NEAR(double, double, double);

void test_bin_handler() {
    using namespace magpy;
    
    std::cout << "Creating bin handler..." << std::endl;
    
    std::vector<Eigen::VectorXd> bin_edges(3);
    bin_edges[0] = Eigen::VectorXd(4);
    bin_edges[0] << 0, 1.5, 2.5, 3.5;
    
    bin_edges[1] = Eigen::VectorXd(6); 
    bin_edges[1] << 0, 1.5, 2.5, 3.5, 4.5, 5.5;
    
    bin_edges[2] = Eigen::VectorXd(2);
    bin_edges[2] << 0, 1.5;
    
    BinHandler bin_handler(bin_edges);
    
    // Test bin index conversion
    std::cout << "Testing bin index conversion..." << std::endl;
    Eigen::VectorXi indices(3);
    indices << 0, 1, 2;
    
    auto bins = bin_handler.get_bin_from_int(indices);
    
    // Check dimensions
    ASSERT_TRUE(bins.dimension(0) == 3);  // 3 events
    ASSERT_TRUE(bins.dimension(1) == 3);  // 3 dimensions
    ASSERT_TRUE(bins.dimension(2) == 2);  // 2 edges per bin
    
    // Test kinematic bin finding
    std::cout << "Testing kinematic bin finding..." << std::endl;
    Eigen::MatrixXd kinematics(1, 3);
    kinematics << 0.1, 2.0, 1.0;
    
    Eigen::MatrixXi bin_indices = bin_handler.find_bin(kinematics);
    
    ASSERT_TRUE(bin_indices.rows() == 1);
    ASSERT_TRUE(bin_indices.cols() == 3);
    ASSERT_TRUE(bin_indices(0, 0) == 0);  // 0.1 is in bin 0 for first dimension
    ASSERT_TRUE(bin_indices(0, 1) == 1);  // 2.0 is in bin 1 for second dimension
    ASSERT_TRUE(bin_indices(0, 2) == 0);  // 1.0 is in bin 0 for third dimension
    
    // Test multiple events
    std::cout << "Testing multiple events..." << std::endl;
    Eigen::MatrixXd kinematics_multi(3, 3);
    kinematics_multi << 0.1, 2.0, 1.0,
                       2.0, 4.0, 0.5,
                       3.0, 5.0, 1.2;
    
    Eigen::MatrixXi bin_indices_multi = bin_handler.find_bin(kinematics_multi);
    
    ASSERT_TRUE(bin_indices_multi.rows() == 3);
    ASSERT_TRUE(bin_indices_multi.cols() == 3);
    
    // Check first event
    ASSERT_TRUE(bin_indices_multi(0, 0) == 0);
    ASSERT_TRUE(bin_indices_multi(0, 1) == 1); 
    ASSERT_TRUE(bin_indices_multi(0, 2) == 0);
    
    // Check second event
    ASSERT_TRUE(bin_indices_multi(1, 0) == 1);
    ASSERT_TRUE(bin_indices_multi(1, 1) == 3);
    ASSERT_TRUE(bin_indices_multi(1, 2) == 0);
    
    // Check third event
    ASSERT_TRUE(bin_indices_multi(2, 0) == 2);
    ASSERT_TRUE(bin_indices_multi(2, 1) == 4);
    ASSERT_TRUE(bin_indices_multi(2, 2) == 0);
    
    std::cout << "Bin indices: " << std::endl << bin_indices_multi << std::endl;
}

void test_bin_handler_properties() {
    using namespace magpy;
    
    std::cout << "Testing bin handler properties..." << std::endl;
    
    std::vector<Eigen::VectorXd> bin_edges(3);
    bin_edges[0] = Eigen::VectorXd(4);
    bin_edges[0] << 0, 1.5, 2.5, 3.5;
    
    bin_edges[1] = Eigen::VectorXd(6);
    bin_edges[1] << 0, 1.5, 2.5, 3.5, 4.5, 5.5;
    
    bin_edges[2] = Eigen::VectorXd(2);
    bin_edges[2] << 0, 1.5;
    
    BinHandler bin_handler(bin_edges);
    
    // Test that properties exist and have correct dimensions
    const auto& bin_edge_tensor = bin_handler.get_bin_edge_tensor();
    const auto& bin_indices = bin_handler.get_bin_indices();
    
    ASSERT_TRUE(bin_edge_tensor.dimension(0) > 0);
    ASSERT_TRUE(bin_edge_tensor.dimension(1) == 3);
    ASSERT_TRUE(bin_edge_tensor.dimension(2) == 2);
    
    ASSERT_TRUE(bin_indices.rows() > 0);
    ASSERT_TRUE(bin_indices.cols() == 3);
}

void test_bin_handler_performance() {
    using namespace magpy;
    
    std::cout << "Testing bin handler performance..." << std::endl;
    
    std::vector<Eigen::VectorXd> bin_edges(3);
    bin_edges[0] = Eigen::VectorXd::LinSpaced(6, 0.0, 5.0);
    bin_edges[1] = Eigen::VectorXd::LinSpaced(6, 0.0, 5.0);
    bin_edges[2] = Eigen::VectorXd::LinSpaced(3, 0.0, 2.0);
    
    BinHandler bin_handler(bin_edges);
    
    // Large array test
    const int n_events = 10000;
    Eigen::MatrixXd test_kinematics(n_events, 3);
    
    // Fill with random-like data (deterministic for testing)
    for (int i = 0; i < n_events; ++i) {
        test_kinematics(i, 0) = 3.5 * (static_cast<double>(i % 1000) / 1000.0);
        test_kinematics(i, 1) = 5.5 * (static_cast<double>((i * 17) % 1000) / 1000.0);
        test_kinematics(i, 2) = 1.5 * (static_cast<double>((i * 31) % 1000) / 1000.0);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXi bin_indices = bin_handler.find_bin(test_kinematics);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "C++ BinHandler performance: " << duration.count() << " μs for " << n_events << " events" << std::endl;
    
    // Check results
    ASSERT_TRUE(bin_indices.rows() == n_events);
    ASSERT_TRUE(bin_indices.cols() == 3);
    
    // Performance check - should be reasonably fast
    ASSERT_TRUE(duration.count() < 100000);  // Less than 100ms
}

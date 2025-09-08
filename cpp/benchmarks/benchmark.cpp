#include "oscillator.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>

int main() {
    std::cout << "=== MAGPY C++ BENCHMARKS ===" << std::endl;
    
    // Benchmark parameters from Python
    const Eigen::VectorXd osc_params = (Eigen::VectorXd(6) << 
        0.3, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3).finished();
    
    // Create oscillator
    magpy::Oscillator osc(1300, 0.5, 3.0, 1000);
    
    // Benchmark 1: 50 events timing (as requested)
    std::cout << "\n--- BENCHMARK 1: 50 Events Timing ---" << std::endl;
    {
        const int n_events = 50;
        Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n_events, 0.1, 10.0);
        Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, 14);  // muon neutrino
        Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, 14);    // muon neutrino
        
        osc.set_energy_osc(energies, start_nu, end_nu);
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            osc.calc_probability(osc_params);
        }
        
        // Timing
        const int n_iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < n_iterations; ++i) {
            auto result = osc.calc_probability(osc_params);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double time_per_call = static_cast<double>(duration.count()) / n_iterations;
        double time_per_event = time_per_call / n_events;
        
        std::cout << "Time per call (50 events): " << time_per_call << " μs" << std::endl;
        std::cout << "Time per event: " << time_per_event << " μs" << std::endl;
        std::cout << "Events per second: " << (1e6 / time_per_event) << std::endl;
    }
    
    // Benchmark 2: Scaling test (like Python benchmark)
    std::cout << "\n--- BENCHMARK 2: Scaling Performance ---" << std::endl;
    {
        std::vector<int> event_counts = {100, 500, 1000, 5000, 10000, 50000};
        
        for (int n_events : event_counts) {
            Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n_events, 0.1, 10.0);
            Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, 14);
            Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, 14);
            
            osc.set_energy_osc(energies, start_nu, end_nu);
            
            // Warm up
            osc.calc_probability(osc_params);
            
            // Timing
            const int n_iterations = std::max(1, 10000 / n_events);  // Adaptive iterations
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < n_iterations; ++i) {
                auto result = osc.calc_probability(osc_params);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double time_per_call = static_cast<double>(duration.count()) / n_iterations;
            double time_per_event = time_per_call / n_events;
            
            std::cout << "Events: " << n_events 
                      << ", Time per call: " << time_per_call << " μs"
                      << ", Time per event: " << time_per_event << " μs"
                      << ", Events/sec: " << (1e6 / time_per_event) << std::endl;
        }
    }
    
    // Benchmark 3: Memory efficiency test
    std::cout << "\n--- BENCHMARK 3: Memory Efficiency ---" << std::endl;
    {
        const int n_events = 100000;
        Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n_events, 0.1, 10.0);
        Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, 14);
        Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, 14);
        
        auto start = std::chrono::high_resolution_clock::now();
        osc.set_energy_osc(energies, start_nu, end_nu);
        auto result = osc.calc_probability(osc_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_per_event = static_cast<double>(duration.count()) / n_events;
        
        std::cout << "Large scale test (" << n_events << " events):" << std::endl;
        std::cout << "Total time: " << duration.count() << " μs" << std::endl;
        std::cout << "Time per event: " << time_per_event << " μs" << std::endl;
        std::cout << "Events per second: " << (1e6 / time_per_event) << std::endl;
        
        // Memory usage estimate
        size_t memory_usage = sizeof(double) * n_events * 3 + sizeof(int) * n_events * 2;
        std::cout << "Estimated memory usage: " << memory_usage / 1024 << " KB" << std::endl;
    }
    
    std::cout << "\n=== BENCHMARKS COMPLETED ===" << std::endl;
    
    return 0;
}

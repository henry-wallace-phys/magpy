#include "oscillator.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

int main() {
    std::cout << "=== MAGPY C++ BENCHMARKS WITH DATA COLLECTION ===" << std::endl;
    
    // Benchmark parameters from Python
    const Eigen::VectorXd osc_params = (Eigen::VectorXd(6) << 
        0.3, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3).finished();
    
    // Create oscillator
    magpy::Oscillator osc(1300, 0.5, 3.0, 1000);
    
    // Benchmark 1: 50 events timing with 100 iterations for histogram
    std::cout << "\n--- BENCHMARK 1: 50 Events Timing (100 iterations) ---" << std::endl;
    {
        const int n_events = 50;
        const int n_iterations = 100;
        
        Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n_events, 0.1, 10.0);
        Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, 14);  // muon neutrino
        Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, 14);    // muon neutrino
        
        osc.set_energy_osc(energies, start_nu, end_nu);
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            osc.calc_probability(osc_params);
        }
        
        // Collect timing data
        std::vector<double> times_microseconds;
        times_microseconds.reserve(n_iterations);
        
        for (int i = 0; i < n_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = osc.calc_probability(osc_params);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            double time_microseconds = static_cast<double>(duration.count()) / 1000.0;
            times_microseconds.push_back(time_microseconds);
        }
        
        // Calculate statistics
        double total_time = 0.0;
        double min_time = times_microseconds[0];
        double max_time = times_microseconds[0];
        
        for (double time : times_microseconds) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double avg_time = total_time / n_iterations;
        double avg_time_per_event = avg_time / n_events;
        
        std::cout << "Average time per call: " << avg_time << " μs" << std::endl;
        std::cout << "Average time per event: " << avg_time_per_event << " μs" << std::endl;
        std::cout << "Min time: " << min_time << " μs" << std::endl;
        std::cout << "Max time: " << max_time << " μs" << std::endl;
        std::cout << "Events per second: " << (1e6 / avg_time_per_event) << std::endl;
        
        // Save timing data for plotting
        std::ofstream file("benchmark_50_events.csv");
        file << "iteration,time_microseconds,time_per_event\n";
        for (int i = 0; i < n_iterations; ++i) {
            file << i << "," << times_microseconds[i] << "," << (times_microseconds[i] / n_events) << "\n";
        }
        file.close();
        std::cout << "Timing data saved to benchmark_50_events.csv" << std::endl;
    }
    
    // Benchmark 2: Scaling test with data collection
    std::cout << "\n--- BENCHMARK 2: Scaling Performance ---" << std::endl;
    {
        std::vector<int> event_counts = {100, 500, 1000, 2000, 5000, 10000, 20000, 50000};
        
        std::ofstream file("benchmark_scaling.csv");
        file << "n_events,time_microseconds,time_per_event,events_per_second\n";
        
        for (int n_events : event_counts) {
            Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n_events, 0.1, 10.0);
            Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, 14);
            Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, 14);
            
            osc.set_energy_osc(energies, start_nu, end_nu);
            
            // Warm up
            osc.calc_probability(osc_params);
            
            // Timing with multiple iterations for accuracy
            const int n_iterations = std::max(1, 1000 / (n_events / 1000));  // Adaptive iterations
            std::vector<double> times;
            
            for (int i = 0; i < n_iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = osc.calc_probability(osc_params);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                times.push_back(static_cast<double>(duration.count()) / 1000.0);  // Convert to microseconds
            }
            
            // Calculate average
            double avg_time = 0.0;
            for (double time : times) {
                avg_time += time;
            }
            avg_time /= n_iterations;
            
            double time_per_event = avg_time / n_events;
            double events_per_second = 1e6 / time_per_event;
            
            std::cout << "Events: " << n_events 
                      << ", Avg time: " << avg_time << " μs"
                      << ", Time per event: " << time_per_event << " μs"
                      << ", Events/sec: " << events_per_second << std::endl;
            
            file << n_events << "," << avg_time << "," << time_per_event << "," << events_per_second << "\n";
        }
        
        file.close();
        std::cout << "Scaling data saved to benchmark_scaling.csv" << std::endl;
    }
    
    // Benchmark 3: Different oscillation channels
    std::cout << "\n--- BENCHMARK 3: Different Oscillation Channels ---" << std::endl;
    {
        const int n_events = 10000;
        const int n_iterations = 10;
        
        // Different oscillation channels
        std::vector<std::pair<std::string, std::pair<int, int>>> channels = {
            {"nu_mu_to_nu_e", {14, 12}},
            {"nu_mu_to_nu_mu", {14, 14}},
            {"nu_mu_to_nu_tau", {14, 16}},
            {"nu_e_to_nu_mu", {12, 14}},
            {"nu_e_to_nu_e", {12, 12}},
            {"nu_e_to_nu_tau", {12, 16}},
            {"nu_tau_to_nu_mu", {16, 14}},
            {"nu_tau_to_nu_e", {16, 12}},
            {"nu_tau_to_nu_tau", {16, 16}}
        };
        
        std::ofstream file("benchmark_channels.csv");
        file << "channel,avg_time_microseconds,time_per_event,events_per_second\n";
        
        for (const auto& channel : channels) {
            std::string name = channel.first;
            int in_pdg = channel.second.first;
            int out_pdg = channel.second.second;
            
            Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n_events, 0.1, 10.0);
            Eigen::VectorXi start_nu = Eigen::VectorXi::Constant(n_events, in_pdg);
            Eigen::VectorXi end_nu = Eigen::VectorXi::Constant(n_events, out_pdg);
            
            osc.set_energy_osc(energies, start_nu, end_nu);
            
            // Timing
            std::vector<double> times;
            for (int i = 0; i < n_iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto result = osc.calc_probability(osc_params);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                times.push_back(static_cast<double>(duration.count()) / 1000.0);
            }
            
            double avg_time = 0.0;
            for (double time : times) avg_time += time;
            avg_time /= n_iterations;
            
            double time_per_event = avg_time / n_events;
            double events_per_second = 1e6 / time_per_event;
            
            std::cout << "Channel " << name << ": " << avg_time << " μs, " 
                      << time_per_event << " μs/event, " << events_per_second << " events/s" << std::endl;
            
            file << name << "," << avg_time << "," << time_per_event << "," << events_per_second << "\n";
        }
        
        file.close();
        std::cout << "Channel data saved to benchmark_channels.csv" << std::endl;
    }
    
    std::cout << "\n=== BENCHMARKS COMPLETED - Data saved for plotting ===" << std::endl;
    
    return 0;
}

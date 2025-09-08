/**
 * C++ Performance Benchmark
 * 
 * Comprehensive performance testing for various components.
 * Measures throughput and timing for oscillations, MC loading, etc.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <Eigen/Dense>
#include "oscillator.h"
#include "mc_file.h"

#ifdef HAS_ROOT
#include "TFile.h"
#include "TTree.h"
#endif

// JSON parsing 
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class PerformanceBenchmark {
private:
    std::mt19937 rng;
    
public:
    PerformanceBenchmark() : rng(42) {} // Fixed seed for reproducibility
    
    json benchmark_oscillator_performance() {
        std::cout << "🚀 Benchmarking Oscillator Performance" << std::endl;
        
        json results;
        std::vector<int> event_counts = {1000, 5000, 10000, 50000, 100000};
        
        // Test parameters
        std::vector<double> osc_params = {0.3, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3};
        Eigen::VectorXd eigen_osc_params = Eigen::Map<Eigen::VectorXd>(osc_params.data(), osc_params.size());
        
        double L = 1300.0;
        double rho = 0.5;
        double Y_e = 3.0;
        int n_layers = 1000;
        
        for (int n_events : event_counts) {
            std::cout << "  Testing " << n_events << " events..." << std::endl;
            
            // Generate test data
            std::uniform_real_distribution<double> energy_dist(0.1, 10.0);
            
            Eigen::VectorXd energies(n_events);
            Eigen::VectorXi start_nu(n_events);
            Eigen::VectorXi end_nu(n_events);
            
            for (int i = 0; i < n_events; ++i) {
                energies[i] = energy_dist(rng);
                start_nu[i] = 14; // muon neutrino
                end_nu[i] = 14;   // muon neutrino
            }
            
            // Time the calculation
            auto start_time = std::chrono::high_resolution_clock::now();
            
            magpy::Oscillator osc(L, rho, Y_e, n_layers);
            osc.set_energy_osc(energies, start_nu, end_nu);
            Eigen::VectorXd probs = osc.calc_probability(eigen_osc_params);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double time_seconds = duration.count() / 1000000.0;
            
            double events_per_second = n_events / time_seconds;
            
            results["oscillator"][std::to_string(n_events)] = {
                {"time_seconds", time_seconds},
                {"events_per_second", events_per_second},
                {"n_events", n_events}
            };
            
            std::cout << "    " << std::fixed << std::setprecision(2) 
                     << events_per_second / 1000000.0 << "M events/sec" << std::endl;
        }
        
        return results;
    }
    
    json benchmark_mc_loading_performance() {
        std::cout << "📊 Benchmarking MC Loading Performance" << std::endl;
        
        json results;
        
        // Create test CSV files of different sizes
        std::vector<int> event_counts = {1000, 5000, 10000, 50000};
        
        for (int n_events : event_counts) {
            std::cout << "  Testing " << n_events << " events..." << std::endl;
            
                        // Generate test ROOT file
            std::string root_filename = "perf_test_" + std::to_string(n_events) + ".root";
            generate_test_root_file(root_filename, n_events);
            
            // Time the loading
            auto start_time = std::chrono::high_resolution_clock::now();
            
            magpy::MCFile mc_file(root_filename, "mc_tree");
            
            // Configure MC branches (matching the ROOT file structure)
            mc_file.set_mc_branch(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY, "true_neutrino_energy");
            mc_file.set_mc_branch(magpy::MCEventIndices::TRUE_Q2, "true_q2");
            mc_file.set_mc_branch(magpy::MCEventIndices::RECO_NEUTRINO_ENERGY, "reco_neutrino_energy");
            mc_file.set_mc_branch(magpy::MCEventIndices::INTERACTION_MODE, "interaction_mode");
            mc_file.set_mc_branch(magpy::MCEventIndices::START_NU, "start_nu");
            mc_file.set_mc_branch(magpy::MCEventIndices::END_NU, "end_nu");
            mc_file.set_mc_branch(magpy::MCEventIndices::TARGET, "target");
            mc_file.set_mc_branch(magpy::MCEventIndices::WEIGHT, "weight");
            
            mc_file.fill_monolith();
            auto monolith = mc_file.get_monolith();
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double time_seconds = duration.count() / 1000000.0;
            
            double events_per_second = n_events / time_seconds;
            
            results["mc_loading"][std::to_string(n_events)] = {
                {"time_seconds", time_seconds},
                {"events_per_second", events_per_second},
                {"n_events", n_events}
            };
            
            std::cout << "    " << std::fixed << std::setprecision(2) 
                     << events_per_second / 1000000.0 << "M events/sec" << std::endl;
            
            // Clean up
            std::remove(root_filename.c_str());
        }
        
        return results;
    }
    
    void generate_test_root_file(const std::string& filename, int n_events) {
#ifdef HAS_ROOT
        // Create ROOT file and tree
        TFile* file = new TFile(filename.c_str(), "RECREATE");
        TTree* tree = new TTree("mc_tree", "Monte Carlo Events");
        
        // Variables for branches
        double true_neutrino_energy, true_q2, reco_neutrino_energy, weight;
        int interaction_mode, start_nu, end_nu, target;
        
        // Create branches
        tree->Branch("true_neutrino_energy", &true_neutrino_energy, "true_neutrino_energy/D");
        tree->Branch("true_q2", &true_q2, "true_q2/D");
        tree->Branch("reco_neutrino_energy", &reco_neutrino_energy, "reco_neutrino_energy/D");
        tree->Branch("interaction_mode", &interaction_mode, "interaction_mode/I");
        tree->Branch("start_nu", &start_nu, "start_nu/I");
        tree->Branch("end_nu", &end_nu, "end_nu/I");
        tree->Branch("target", &target, "target/I");
        tree->Branch("weight", &weight, "weight/D");
        
        // Generate random data
        std::uniform_real_distribution<double> energy_dist(0.1, 10.0);
        std::uniform_real_distribution<double> q2_dist(0.01, 2.0);
        std::uniform_int_distribution<int> mode_dist(1, 10);
        std::uniform_int_distribution<int> target_dist(0, 1);
        
        for (int i = 0; i < n_events; ++i) {
            true_neutrino_energy = energy_dist(rng);
            true_q2 = q2_dist(rng);
            reco_neutrino_energy = energy_dist(rng);
            interaction_mode = mode_dist(rng);
            start_nu = 14; // muon neutrino
            end_nu = 14;   // muon neutrino
            target = target_dist(rng) ? 1000060120 : 1000010010; // C or H
            weight = 1.0;
            
            tree->Fill();
        }
        
        file->Write();
        file->Close();
        delete file;
#else
        throw std::runtime_error("ROOT support not available for performance testing");
#endif
    }
};

int main() {
    try {
        std::cout << "⚡ C++ Performance Benchmark Suite" << std::endl;
        std::cout << "===================================" << std::endl;
        
        PerformanceBenchmark benchmark;
        
        json all_results;
        
        // Run oscillator benchmarks
        json osc_results = benchmark.benchmark_oscillator_performance();
        all_results.update(osc_results);
        
        // Run MC loading benchmarks  
        json mc_results = benchmark.benchmark_mc_loading_performance();
        all_results.update(mc_results);
        
        // Save comprehensive results
        std::ofstream results_file("performance_comparison.json");
        results_file << all_results.dump(2);
        results_file.close();
        
        std::cout << "\n✅ Performance benchmark completed" << std::endl;
        std::cout << "Results saved to performance_comparison.json" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

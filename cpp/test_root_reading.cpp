#include "mc_file.h"
#include "oscillator.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== MAGPY C++ ROOT FILE READING TEST ===" << std::endl;
    
    try {
        // Test with example ROOT file path (adjust as needed)
        std::string root_file_path = "../src/magpy/tests/data/NuWro_FlatTree.root";
        std::string tree_name = "FlatTree_VARS";
        
        std::cout << "\n--- Loading ROOT file ---" << std::endl;
        magpy::MCFile mc_file(root_file_path, tree_name);
        
        std::cout << "\n--- Available branches ---" << std::endl;
        auto branch_names = mc_file.get_branch_names();
        std::cout << "Found " << branch_names.size() << " branches:" << std::endl;
        for (const auto& name : branch_names) {
            std::cout << "  - " << name << std::endl;
        }
        
        std::cout << "\n--- Configuring branch mappings ---" << std::endl;
        // Set up branch mappings like Python implementation
        mc_file.set_mc_branch(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY, "Enu_true");
        mc_file.set_mc_branch(magpy::MCEventIndices::TRUE_Q2, "Q2");
        mc_file.set_mc_const(magpy::MCEventIndices::RECO_NEUTRINO_ENERGY, 0.0);
        mc_file.set_mc_branch(magpy::MCEventIndices::INTERACTION_MODE, "Mode");
        mc_file.set_mc_branch(magpy::MCEventIndices::TARGET, "tgt");
        mc_file.set_mc_const(magpy::MCEventIndices::START_NU, 14);  // muon neutrino
        mc_file.set_mc_const(magpy::MCEventIndices::END_NU, 14);   // muon neutrino
        mc_file.set_mc_const(magpy::MCEventIndices::WEIGHT, 1.0);
        
        std::cout << "\n--- Filling monolith ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        mc_file.fill_monolith();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Filled monolith in " << duration.count() << " ms" << std::endl;
        
        const auto& monolith = mc_file.get_monolith();
        std::cout << "Loaded " << monolith.size() << " events" << std::endl;
        
        std::cout << "\n--- Data summary ---" << std::endl;
        auto energies = monolith.get_column(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY);
        auto q2_values = monolith.get_column(magpy::MCEventIndices::TRUE_Q2);
        auto start_nu_dbl = monolith.get_column(magpy::MCEventIndices::START_NU);
        auto end_nu_dbl = monolith.get_column(magpy::MCEventIndices::END_NU);
        
        // Convert to integers
        Eigen::VectorXi start_nu = start_nu_dbl.cast<int>();
        Eigen::VectorXi end_nu = end_nu_dbl.cast<int>();
        
        std::cout << "Energy range: " << energies.minCoeff() << " - " << energies.maxCoeff() << " GeV" << std::endl;
        std::cout << "Q2 range: " << q2_values.minCoeff() << " - " << q2_values.maxCoeff() << " GeV²" << std::endl;
        std::cout << "Start neutrino PDG: " << start_nu[0] << " (all same: " << 
                     (start_nu.array() == start_nu[0]).all() << ")" << std::endl;
        std::cout << "End neutrino PDG: " << end_nu[0] << " (all same: " << 
                     (end_nu.array() == end_nu[0]).all() << ")" << std::endl;
        
        std::cout << "\n--- First 5 events ---" << std::endl;
        auto full_matrix = monolith.get_monolith();
        for (size_t i = 0; i < std::min(size_t(5), monolith.size()); ++i) {
            std::cout << "Event " << i << ": E=" << full_matrix(i, static_cast<int>(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY))
                      << " GeV, Q²=" << full_matrix(i, static_cast<int>(magpy::MCEventIndices::TRUE_Q2))
                      << " GeV², Mode=" << full_matrix(i, static_cast<int>(magpy::MCEventIndices::INTERACTION_MODE))
                      << ", Target=" << full_matrix(i, static_cast<int>(magpy::MCEventIndices::TARGET)) << std::endl;
        }
        
        std::cout << "\n--- Testing oscillator integration ---" << std::endl;
        if (monolith.size() > 0) {
            // Test with first 1000 events or all if fewer
            size_t n_test = std::min(size_t(1000), monolith.size());
            
            // Extract data for oscillator
            Eigen::VectorXd test_energies = energies.head(n_test);
            Eigen::VectorXi test_start_nu = start_nu.head(n_test);
            Eigen::VectorXi test_end_nu = end_nu.head(n_test);
            
            // Create oscillator
            magpy::Oscillator osc(1300, 0.5, 3.0, 1000);
            osc.set_energy_osc(test_energies, test_start_nu, test_end_nu);
            
            // Oscillation parameters
            const Eigen::VectorXd osc_params = (Eigen::VectorXd(6) << 
                0.3, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3).finished();
            
            // Calculate probabilities
            auto osc_start = std::chrono::high_resolution_clock::now();
            auto probabilities = osc.calc_probability(osc_params);
            auto osc_end = std::chrono::high_resolution_clock::now();
            
            auto osc_duration = std::chrono::duration_cast<std::chrono::microseconds>(osc_end - osc_start);
            
            std::cout << "Calculated oscillation probabilities for " << n_test << " events" << std::endl;
            std::cout << "Oscillation calculation took " << osc_duration.count() << " μs" << std::endl;
            std::cout << "Probability range: " << probabilities.minCoeff() << " - " << probabilities.maxCoeff() << std::endl;
            std::cout << "Mean probability: " << probabilities.mean() << std::endl;
            
            // Performance metrics
            double events_per_second = n_test * 1e6 / osc_duration.count();
            std::cout << "Performance: " << events_per_second << " events/second" << std::endl;
        }
        
        std::cout << "\n--- Data matrix test ---" << std::endl;
        auto data_matrix = monolith.get_monolith();
        std::cout << "Data matrix shape: " << data_matrix.rows() << " x " << data_matrix.cols() << std::endl;
        std::cout << "Matrix memory usage: " << 
                     (data_matrix.rows() * data_matrix.cols() * sizeof(double)) / (1024.0 * 1024.0) 
                     << " MB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "\nNote: This test requires a ROOT file at the specified path." << std::endl;
        std::cout << "If you don't have the test data, you can create a simple ROOT file" << std::endl;
        std::cout << "or modify the path to point to an existing neutrino MC ROOT file." << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 ROOT file reading test completed successfully!" << std::endl;
    return 0;
}

/**
 * C++ MC Event Comparison Test
 * 
 * Loads MC even        // Extract data for oscillation calculation
        Eigen::VectorXd energies = monolith.get_column(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY);
        Eigen::VectorXd start_nu_double = monolith.get_column(magpy::MCEventIndices::START_NU);
        Eigen::VectorXd end_nu_double = monolith.get_column(magpy::MCEventIndices::END_NU);
        
        // Convert to integer vectors
        Eigen::VectorXi start_nu = start_nu_double.cast<int>();
        Eigen::VectorXi end_nu = end_nu_double.cast<int>();rom CSV, processes them with C++ MCFile,
 * and compares against Python results for data consistency.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>
#include "mc_file.h"
#include "oscillator.h"

// JSON parsing 
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mc_events.root>" << std::endl;
        return 1;
    }

    std::string root_file = argv[1];
    
    try {
        std::cout << "🧪 C++ MC Event Comparison Test" << std::endl;
        std::cout << "Loading MC events from: " << root_file << std::endl;
        
        // Time the C++ processing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Load MC events using MCFile
        magpy::MCFile mc_file(root_file);
        
        // Set up branch mappings for ROOT file
        mc_file.set_mc_branch(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY, "true_neutrino_energy");
        mc_file.set_mc_branch(magpy::MCEventIndices::TRUE_Q2, "true_q2");
        mc_file.set_mc_branch(magpy::MCEventIndices::RECO_NEUTRINO_ENERGY, "reco_neutrino_energy");
        mc_file.set_mc_branch(magpy::MCEventIndices::INTERACTION_MODE, "interaction_mode");
        mc_file.set_mc_branch(magpy::MCEventIndices::START_NU, "start_nu");
        mc_file.set_mc_branch(magpy::MCEventIndices::END_NU, "end_nu");
        mc_file.set_mc_branch(magpy::MCEventIndices::TARGET, "target");
        mc_file.set_mc_branch(magpy::MCEventIndices::WEIGHT, "weight");
        
        // Fill the monolith
        mc_file.fill_monolith();
        
        // Get the monolith
        auto monolith = mc_file.get_monolith();
        
        std::cout << "Loaded " << monolith.size() << " MC events" << std::endl;
        
        // Extract data for oscillation calculation
        Eigen::VectorXd energies = monolith.get_column(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY);
        Eigen::VectorXd start_nu_double = monolith.get_column(magpy::MCEventIndices::START_NU);
        Eigen::VectorXd end_nu_double = monolith.get_column(magpy::MCEventIndices::END_NU);
        
        // Convert to int vectors (with proper casting)
        Eigen::VectorXi start_nu = start_nu_double.cast<int>();
        Eigen::VectorXi end_nu = end_nu_double.cast<int>();
        
        // Load Python results for comparison
        std::ifstream python_file("python_mc_results.json");
        if (!python_file.is_open()) {
            throw std::runtime_error("Cannot open Python results file");
        }
        
        json python_results;
        python_file >> python_results;
        python_file.close();
        
        // Extract oscillation parameters from comparison script
        std::vector<double> osc_params = {0.3, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3};
        double L = 1300.0;
        double rho = 0.5;
        double Y_e = 3.0;
        int n_layers = 1000;
        
        std::cout << "Calculating oscillation probabilities..." << std::endl;
        
        // Create oscillator and calculate probabilities
        magpy::Oscillator cpp_osc(L, rho, Y_e, n_layers);
        cpp_osc.set_energy_osc(energies, start_nu, end_nu);
        
        Eigen::VectorXd osc_params_eigen = Eigen::Map<Eigen::VectorXd>(osc_params.data(), osc_params.size());
        Eigen::VectorXd cpp_probs = cpp_osc.calc_probability(osc_params_eigen);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto cpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double cpp_time = cpp_duration.count() / 1000000.0;
        
        std::cout << "C++ processing completed in " << cpp_time << " seconds" << std::endl;
        
        // Save results for comparison
        json results;
        results["energies"] = std::vector<double>(energies.data(), energies.data() + energies.size());
        results["start_nu"] = std::vector<int>(start_nu.data(), start_nu.data() + start_nu.size());
        results["end_nu"] = std::vector<int>(end_nu.data(), end_nu.data() + end_nu.size());
        results["probabilities"] = std::vector<double>(cpp_probs.data(), cpp_probs.data() + cpp_probs.size());
        results["processing_time"] = cpp_time;
        results["n_events"] = monolith.size();
        
        std::ofstream results_file("cpp_mc_results.json");
        results_file << results.dump(2);
        results_file.close();
        
        std::cout << "✅ Results saved to cpp_mc_results.json" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

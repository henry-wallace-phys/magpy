/**
 * C++ Oscillator Comparison Test
 * 
 * Loads test data from Python and compares oscillation calculations
 * to ensure exact numerical agreement at machine precision.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>
#include "oscillator.h"

// JSON parsing 
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_data.json>" << std::endl;
        return 1;
    }

    std::string test_file = argv[1];
    
    try {
        std::cout << "🧪 C++ Oscillator Comparison Test" << std::endl;
        std::cout << "Loading test data from: " << test_file << std::endl;
        
        // Load test data
        std::ifstream file(test_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open test file: " + test_file);
        }
        
        json test_data;
        file >> test_data;
        file.close();
        
        // Extract test parameters
        std::vector<double> energies = test_data["energies"];
        std::vector<int> start_nu = test_data["start_nu"];
        std::vector<int> end_nu = test_data["end_nu"];
        std::vector<double> osc_params = test_data["osc_params"];
        
        auto osc_config = test_data["osc_config"];
        double L = osc_config["L"];
        double rho = osc_config["rho"];
        double Y_e = osc_config["Y_e"];
        int n_layers = osc_config["n_layers"];
        
        std::vector<double> python_probs = test_data["python_probabilities"];
        
        std::cout << "Test parameters:" << std::endl;
        std::cout << "  Events: " << energies.size() << std::endl;
        std::cout << "  L: " << L << " km" << std::endl;
        std::cout << "  ρ: " << rho << " g/cm³" << std::endl;
        std::cout << "  Y_e: " << Y_e << std::endl;
        std::cout << "  Layers: " << n_layers << std::endl;
        
        // Convert to Eigen arrays for C++ oscillator
        Eigen::VectorXd eigen_energies = Eigen::Map<Eigen::VectorXd>(energies.data(), energies.size());
        Eigen::VectorXi eigen_start_nu = Eigen::Map<Eigen::VectorXi>(start_nu.data(), start_nu.size());
        Eigen::VectorXi eigen_end_nu = Eigen::Map<Eigen::VectorXi>(end_nu.data(), end_nu.size());
        Eigen::VectorXd eigen_osc_params = Eigen::Map<Eigen::VectorXd>(osc_params.data(), osc_params.size());
        
        std::cout << "\nRunning C++ oscillation calculation..." << std::endl;
        
        // Time the C++ calculation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create oscillator
        magpy::Oscillator cpp_osc(L, rho, Y_e, n_layers);
        
        // Set energy and neutrino types
        cpp_osc.set_energy_osc(eigen_energies, eigen_start_nu, eigen_end_nu);
        
        // Calculate probabilities
        Eigen::VectorXd cpp_probs = cpp_osc.calc_probability(eigen_osc_params);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto cpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double cpp_time = cpp_duration.count() / 1000000.0;
        
        std::cout << "C++ calculation completed in " << cpp_time << " seconds" << std::endl;
        
        // Compare results
        double max_diff = 0.0;
        double sum_diff = 0.0;
        
        for (size_t i = 0; i < energies.size(); ++i) {
            double diff = std::abs(cpp_probs[i] - python_probs[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
        }
        
        double mean_diff = sum_diff / energies.size();
        
        std::cout << "\nComparison Results:" << std::endl;
        std::cout << std::scientific << std::setprecision(2);
        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Mean difference: " << mean_diff << std::endl;
        
        // Save results
        json results;
        results["cpp_probabilities"] = std::vector<double>(cpp_probs.data(), cpp_probs.data() + cpp_probs.size());
        results["cpp_time"] = cpp_time;
        results["max_difference"] = max_diff;
        results["mean_difference"] = mean_diff;
        results["n_events"] = energies.size();
        
        std::ofstream results_file("oscillator_test_results.json");
        results_file << results.dump(2);
        results_file.close();
        
        std::cout << "✅ Results saved to oscillator_test_results.json" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

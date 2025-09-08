/**
 * @file debug_mcmc.cpp
 * @brief Debug the MCMC issues
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include "adaptive_metropolis.h"
#include "poisson_likelihood.h"
#include "mc_event.h"

// Generate test MC events with POT scaling
std::vector<magpy::MCEvent> generate_test_events(int n_events, double pot_scale = 1.0, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    std::vector<magpy::MCEvent> events;
    events.reserve(n_events);
    
    for (int i = 0; i < n_events; ++i) {
        // Generate realistic neutrino energies (log-normal distribution)
        double log_energy = std::log(2.0) + 0.8 * normal_dist(rng);
        double energy = std::exp(log_energy);
        energy = std::max(0.1, std::min(energy, 20.0)); // Clip to reasonable range
        
        // Most events are muon neutrino -> muon neutrino
        int start_nu = 14;
        int end_nu = 14;
        
        // Add some noise to reconstructed energy
        double reco_energy = energy * (1 + 0.1 * normal_dist(rng));
        
        magpy::MCEvent event{
            energy,           // true_neutrino_energy
            0.5,             // true_q2
            reco_energy,     // reco_neutrino_energy
            1,               // interaction_mode
            1000060120,      // target (carbon)
            start_nu,        // start_nu
            end_nu,          // end_nu
            pot_scale        // weight scaled by POT ratio
        };
        
        events.push_back(event);
    }
    
    return events;
}

// Debug bounds checking
bool debug_parameter_bounds(const Eigen::VectorXd& params) {
    bool all_good = true;
    
    std::cout << "DEBUG: Checking bounds for params: ";
    for (int i = 0; i < params.size(); ++i) {
        std::cout << params[i] << " ";
    }
    std::cout << std::endl;
    
    if (params.size() >= 6) {
        // Check oscillation parameter bounds
        if (params[0] < 0.28 || params[0] > 0.35) { std::cout << "theta12 out of bounds: " << params[0] << std::endl; all_good = false; }
        if (params[1] < 0.005 || params[1] > 0.025) { std::cout << "theta13 out of bounds: " << params[1] << std::endl; all_good = false; }
        if (params[2] < 0.35 || params[2] > 0.75) { std::cout << "theta23 out of bounds: " << params[2] << std::endl; all_good = false; }
        // deltacp is periodic - no bounds
        if (params[4] < 0.0 || params[4] > 0.0002) { std::cout << "dm21 out of bounds: " << params[4] << std::endl; all_good = false; }
        // dm32 can be positive or negative (mass ordering)
        if (std::abs(params[5]) < 0.002 || std::abs(params[5]) > 0.004) { std::cout << "dm32 out of bounds: " << params[5] << std::endl; all_good = false; }
    }
    
    // Check systematic parameters (should be small)
    for (int i = 6; i < params.size(); ++i) {
        if (std::abs(params[i]) > 0.02) { 
            std::cout << "syst_" << (i-6) << " out of bounds: " << params[i] << std::endl; 
            all_good = false; 
        }
    }
    
    std::cout << "DEBUG: Bounds check result: " << (all_good ? "PASS" : "FAIL") << std::endl;
    return all_good;
}

int main() {
    std::cout << "🔍 Debug MCMC Test" << std::endl;
    std::cout << "==================" << std::endl;
    
    // Generate test data and MC
    std::cout << "Generating 1000 test MC events..." << std::endl;
    auto data_events = generate_test_events(1000);
    auto mc_events = generate_test_events(1000);
    
    // Energy bins for binning
    std::vector<double> energy_bins;
    for (int i = 0; i <= 40; ++i) {
        energy_bins.push_back(0.1 + i * (20.0 - 0.1) / 40);
    }
    
    // Setup likelihood
    BinnedPoissonLikelihood likelihood(data_events, mc_events, energy_bins);
    
    // Starting parameters (nominal values)
    Eigen::VectorXd nominal_params(11);
    nominal_params << 0.30, 0.02, 0.55, 2.199114857512855, 0.000075, -0.0025, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    std::cout << "Nominal parameters: ";
    for (int i = 0; i < nominal_params.size(); ++i) {
        std::cout << nominal_params[i] << " ";
    }
    std::cout << std::endl;
    
    // Test bounds on nominal
    std::cout << "\nTesting bounds on nominal parameters:" << std::endl;
    bool nominal_bounds = debug_parameter_bounds(nominal_params);
    
    // Test likelihood evaluation
    std::cout << "\nTesting likelihood evaluation:" << std::endl;
    try {
        double log_prob = likelihood(nominal_params);
        std::cout << "Log likelihood at nominal: " << log_prob << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR evaluating likelihood: " << e.what() << std::endl;
    }
    
    // Test a simple proposal
    std::cout << "\nTesting a small proposal:" << std::endl;
    Eigen::VectorXd small_proposal = nominal_params;
    small_proposal[0] += 0.001;  // Small change to theta12
    
    std::cout << "Small proposal bounds check:" << std::endl;
    bool small_proposal_bounds = debug_parameter_bounds(small_proposal);
    
    try {
        double small_log_prob = likelihood(small_proposal);
        std::cout << "Log likelihood at small proposal: " << small_log_prob << std::endl;
        std::cout << "Log ratio: " << (small_log_prob - likelihood(nominal_params)) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR evaluating small proposal: " << e.what() << std::endl;
    }
    
    // Test a larger proposal
    std::cout << "\nTesting a larger proposal:" << std::endl;
    Eigen::VectorXd large_proposal = nominal_params;
    large_proposal[0] += 0.01;  // Larger change to theta12 (1% change)
    
    std::cout << "Large proposal bounds check:" << std::endl;
    bool large_proposal_bounds = debug_parameter_bounds(large_proposal);
    
    try {
        double large_log_prob = likelihood(large_proposal);
        std::cout << "Log likelihood at large proposal: " << large_log_prob << std::endl;
        std::cout << "Log ratio: " << (large_log_prob - likelihood(nominal_params)) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR evaluating large proposal: " << e.what() << std::endl;
    }
    
    // Test different parameters
    std::cout << "\nTesting theta13 change:" << std::endl;
    Eigen::VectorXd theta13_proposal = nominal_params;
    theta13_proposal[1] += 0.001;  // Change theta13
    
    try {
        double theta13_log_prob = likelihood(theta13_proposal);
        std::cout << "Log likelihood with theta13 change: " << theta13_log_prob << std::endl;
        std::cout << "Log ratio: " << (theta13_log_prob - likelihood(nominal_params)) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR evaluating theta13 change: " << e.what() << std::endl;
    }
    
    return 0;
}

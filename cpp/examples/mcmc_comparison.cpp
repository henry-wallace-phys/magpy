/**
 * @file mcmc_comparison.cpp
 * @brief Test adaptive Metropolis MCMC and compare performance with Python
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

// Enhanced bounds checking with periodic deltacp, mass ordering flip, and spline systematics
bool check_parameter_bounds(const Eigen::VectorXd& params) {
    // Oscillation parameters (first 6)
    // theta12, theta13, theta23 should be > 0 and < π/2
    if (params(0) <= 0.0 || params(0) >= M_PI/2) return false;
    if (params(1) <= 0.0 || params(1) >= M_PI/2) return false; 
    if (params(2) <= 0.0 || params(2) >= M_PI/2) return false;
    
    // deltacp is periodic - handled in proposal step, no bounds check needed here
    
    // dm21 should be > 0
    if (params(4) <= 0.0) return false;
    
    // dm32 can be negative (inverted hierarchy), check reasonable range
    if (std::abs(params(5)) > 0.01) return false;  // |dm32| < 0.01 eV²
    
    // Spline systematic parameters (parameters 6-10): typically within ±3σ
    for (int i = 6; i < params.size(); ++i) {
        if (std::abs(params(i)) > 3.0) return false;  // ±3σ bounds
    }
    
    return true;
}

// Apply periodic boundary conditions to deltacp
double wrap_deltacp(double deltacp) {
    const double two_pi = 2.0 * M_PI;
    while (deltacp < 0.0) deltacp += two_pi;
    while (deltacp >= two_pi) deltacp -= two_pi;
    return deltacp;
}

// Mass ordering flip: dm32 -> -dm32
Eigen::VectorXd flip_mass_ordering(const Eigen::VectorXd& params) {
    Eigen::VectorXd flipped_params = params;
    flipped_params(5) = -params(5);  // Flip dm32
    return flipped_params;
}

// Gaussian prior for oscillation + spline parameters
double gaussian_log_prior(const Eigen::VectorXd& params) {
    double log_prior = 0.0;
    
    // Oscillation parameter priors (first 6 parameters)
    Eigen::VectorXd osc_nominal(6);
    osc_nominal << 0.3, 0.02, 0.55, 2.19911486, 7.5e-5, 2.5e-3;
    
    Eigen::VectorXd osc_sigmas(6);
    osc_sigmas << 0.1, 0.01, 0.1, 1.0, 2e-5, 1e-3;
    
    for (int i = 0; i < 6; ++i) {
        double diff = params(i) - osc_nominal(i);
        log_prior += -0.5 * (diff * diff) / (osc_sigmas(i) * osc_sigmas(i));
        log_prior += -0.5 * std::log(2.0 * M_PI * osc_sigmas(i) * osc_sigmas(i));
    }
    
    // Spline systematic priors (parameters 6-10): standard normal N(0,1)
    for (int i = 6; i < params.size(); ++i) {
        double syst_val = params(i);
        log_prior += -0.5 * syst_val * syst_val;  // N(0,1) prior
        log_prior += -0.5 * std::log(2.0 * M_PI);
    }
    
    return log_prior;
}

void save_results_to_file(const MCMCResult& result, const std::string& filename) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(8);
    
    // Header
    for (size_t i = 0; i < result.param_names.size(); ++i) {
        file << result.param_names[i];
        if (i < result.param_names.size() - 1) file << ",";
    }
    file << ",log_prob\n";
    
    // Data
    for (int i = 0; i < result.samples.rows(); ++i) {
        for (int j = 0; j < result.samples.cols(); ++j) {
            file << result.samples(i, j) << ",";
        }
        file << result.log_probs(i) << "\n";
    }
}

void print_summary_statistics(const MCMCResult& result) {
    std::cout << "\n📊 Summary Statistics:" << std::endl;
    std::cout << "Chain shape: (" << result.samples.rows() << ", " << result.samples.cols() << ")" << std::endl;
    
    for (int i = 0; i < result.samples.cols(); ++i) {
        Eigen::VectorXd param_samples = result.samples.col(i);
        double mean = param_samples.mean();
        double variance = (param_samples.array() - mean).square().mean();
        double std_dev = std::sqrt(variance);
        
        std::cout << result.param_names[i] << ": " 
                  << std::fixed << std::setprecision(6) << mean 
                  << " ± " << std_dev << std::endl;
    }
}

int main() {
    std::cout << "🎯 C++ Enhanced Adaptive Metropolis MCMC Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Configuration parameters (in a real implementation, read from config file)
    double nominal_pot = 1.47e21;
    double actual_pot = 1.47e21;
    double pot_scale = actual_pot / nominal_pot;
    
    std::cout << "POT scaling factor: " << std::fixed << std::setprecision(3) << pot_scale << std::endl;
    
    // Generate test data
    int n_events = 10000;
    std::cout << "Generating " << n_events << " test MC events..." << std::endl;
    auto mc_events = generate_test_events(n_events, pot_scale, 42);
    auto data_events = generate_test_events(n_events, pot_scale, 42); // Same for pseudo-experiment
    
    // Energy binning (matching Python)
    std::vector<double> energy_bins;
    for (int i = 0; i <= 20; ++i) {
        energy_bins.push_back(0.1 + i * (10.0 - 0.1) / 20);
    }
    
    // Create likelihood
    std::cout << "Setting up likelihood..." << std::endl;
    BinnedPoissonLikelihood likelihood(data_events, mc_events, energy_bins);
    
    // Parameter names (oscillation + spline systematics)
    std::vector<std::string> param_names = {
        "theta12", "theta13", "theta23", "deltacp", "dm21", "dm32",
        "syst_0", "syst_1", "syst_2", "syst_3", "syst_4"  // 5 spline systematics
    };
    
    // Nominal parameters (oscillation + spline systematics)
    Eigen::VectorXd nominal_params(11);  // 6 osc + 5 spline
    nominal_params << 0.3, 0.02, 0.55, 2.19911486, 7.5e-5, 2.5e-3,  // oscillation
                      0.0, 0.0, 0.0, 0.0, 0.0;  // spline systematics (nominal = 0)
    
    // Test likelihood at nominal values
    auto log_posterior_fn = [&](const Eigen::VectorXd& params) -> double {
        double log_likelihood = likelihood(params);
        double log_prior = gaussian_log_prior(params);
        return log_likelihood + log_prior;
    };
    
    double test_logp = log_posterior_fn(nominal_params);
    std::cout << "Log posterior at nominal: " << std::fixed << std::setprecision(6) << test_logp << std::endl;
    
    // Create MCMC sampler
    std::cout << "Initializing Enhanced Adaptive Metropolis sampler..." << std::endl;
    std::cout << "Features enabled:" << std::endl;
    std::cout << "  - Mass ordering flips (50% probability)" << std::endl;
    std::cout << "  - Periodic deltacp boundaries" << std::endl;
    std::cout << "  - POT-scaled event weights" << std::endl;
    
    AdaptiveMetropolis sampler(
        log_posterior_fn,
        11,  // 6 oscillation + 5 spline parameters
        param_names,
        0.1,   // Much larger initial step size for proper mixing
        50,    // More frequent adaptation during warmup
        check_parameter_bounds
    );
    
    // Run MCMC
    std::cout << "Running improved MCMC (50,000 samples)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    

    int n_samples = 50000;  // Longer run to test efficiency
    MCMCResult result = sampler.sample(
        nominal_params,
        n_samples,   // n_samples
        5000,    // n_warmup (10% of total)
        1,       // thin
        42       // seed
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_time = duration.count() / 1000.0;
    
    // Print results
    print_summary_statistics(result);
    
    std::cout << "\n⏱️  Performance:" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
    std::cout << "Time per sample: " << std::fixed << std::setprecision(4) 
              << (total_time * 1000.0) / n_samples << " ms" << std::endl;
    std::cout << "Samples per second: " << std::fixed << std::setprecision(1) 
              << n_samples / total_time << std::endl;
    
    // Save results
    std::cout << "\n💾 Saving enhanced MCMC results to cpp_mcmc_results.csv..." << std::endl;
    save_results_to_file(result, "cpp_mcmc_results.csv");
    
    std::cout << "✅ Enhanced C++ MCMC test completed successfully!" << std::endl;
    std::cout << "\nEnhancements applied:" << std::endl;
    std::cout << "  ✓ Mass ordering flips (50% per step)" << std::endl;
    std::cout << "  ✓ Periodic deltacp boundaries" << std::endl;
    std::cout << "  ✓ POT-scaled event weights (" << pot_scale << "x)" << std::endl;
    
    return 0;
}

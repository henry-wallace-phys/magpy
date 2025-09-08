#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>
#include "exceptions.h"

namespace magpy {

/**
 * High-performance fully vectorized C++ oscillator
 * Uses pointer-based probability assignment for maximum performance
 */
class Oscillator {
public:
    /**
     * Constructor
     * @param L Baseline distance in km
     * @param ye Electron density parameter  
     * @param rho Matter density in g/cm³
     * @param n_newton Number of Newton iterations (not used in simplified version)
     */
    Oscillator(double L, double ye, double rho, int n_newton);

    /**
     * Set energy and oscillation channel arrays
     * Sets up pointer-based probability assignment at initialization
     * @param energies Neutrino energies in GeV
     * @param osc_in Incoming neutrino PDG codes 
     * @param osc_out Outgoing neutrino PDG codes
     */
    void set_energy_osc(const Eigen::VectorXd& energies, 
                       const Eigen::VectorXi& osc_in,
                       const Eigen::VectorXi& osc_out);

    /**
     * Calculate oscillation probabilities - fully vectorized
     * @param osc_params Oscillation parameters [s12sq, s13sq, s23sq, delta_cp, dmsq21, dmsq31]
     * @return Vector of oscillation probabilities
     */
    Eigen::VectorXd calc_probability(const Eigen::VectorXd& osc_params);

private:
    double L_, ye_, rho_;
    int n_newton_;
    bool setup_;

    Eigen::VectorXd energies_;
    Eigen::VectorXi osc_in_;
    Eigen::VectorXi osc_out_;

    // Precomputed probability matrices for all 9 channels
    Eigen::MatrixXd P_matrices_[9];  // 9 possible oscillation channels
    
    // Channel index mapping for pointer-based assignment
    std::vector<int> channel_indices_;
    
    // Probability assignment function pointers (set at initialization)
    std::vector<std::function<void(const Eigen::MatrixXd&, double*, int)>> prob_assigners_;

    /**
     * Setup channel assignment and probability pointers at initialization
     */
    void setup_vectorized_channels();

    /**
     * Compute all 9 oscillation probability matrices in one vectorized operation
     */
    void compute_all_probability_matrices(const Eigen::VectorXd& osc_params);

    /**
     * Core vectorized oscillation calculation using precomputed matrices
     */
    void vectorized_oscillation(const Eigen::VectorXd& osc_params, Eigen::VectorXd& results);
};

} // namespace magpy

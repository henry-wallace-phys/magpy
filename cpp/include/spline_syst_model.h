#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include "spline_handler.h"
#include "mc_event.h"
#include "bin_handler.h"

namespace magpy {

/**
 * Systematic uncertainty information
 */
struct Systematic {
    std::string syst_name;
    std::string spline_name;
    std::pair<double, double> range;
    std::string syst_type; // "norm" or "spline"
    std::vector<int> modes;
    double nominal;
    double error;
    std::string prior = "gaussian";
    bool fixed = false;
};

/**
 * Spline systematic model for event reweighting
 * Equivalent to Python SplineSystematicModel
 */
class SplineSystematicModel {
public:
    /**
     * Constructor
     * @param splines SplineMonolith containing all splines
     * @param systematics Vector of systematic configurations
     */
    SplineSystematicModel(SplineMonolith& splines, const std::vector<Systematic>& systematics);

    /**
     * Setup splines and create index mappings
     * @param bin_handler BinHandler for kinematic binning
     */
    void setup_splines(const BinHandler& bin_handler);

    /**
     * Get spline indices for MC events
     * @param mc_events MC event monolith
     * @param bin_indices Indices of kinematic variables to use for binning
     * @return Matrix mapping events to splines
     */
    Eigen::MatrixXi get_monolith_splines(const MCEventMonolith& mc_events, 
                                        const Eigen::VectorXi& bin_indices);

    /**
     * Get systematic weights for all events
     * @param syst_values Current systematic parameter values
     * @return Vector of combined weights for each event
     */
    Eigen::VectorXd get_weights_only(const Eigen::VectorXd& syst_values);

    /**
     * Reweight MC events with systematic uncertainties
     * @param syst_values Systematic parameter values
     * @param monolith Original MC event weights/data
     * @return Reweighted event data
     */
    Eigen::VectorXd reweight(const Eigen::VectorXd& syst_values, const Eigen::VectorXd& monolith);

private:
    SplineMonolith& spline_monolith_;
    std::vector<Systematic> systematics_;
    
    // Index mappings
    Eigen::MatrixXi index_tensor_;
    Eigen::MatrixXi event_spline_pairs_;
    Eigen::VectorXi valid_events_;
    
    // Normalization systematics: systematic_index -> {mode -> spline_index}
    std::unordered_map<int, std::unordered_map<int, int>> norm_systematics_;
    
    const MCEventMonolith* mc_event_monolith_ = nullptr;
    Eigen::VectorXi bins_;

    void apply_normalization_weights(Eigen::VectorXd& event_weights, 
                                   const Eigen::VectorXd& all_spline_weights);
};

} // namespace magpy

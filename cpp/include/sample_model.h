#pragma once

#include <Eigen/Dense>
#include "oscillator.h"
#include "mc_event.h"
#include "spline_syst_model.h"

namespace magpy {

/**
 * Complete sample model handling MC events, oscillations, and systematics
 * Equivalent to Python SampleModel
 */
class SampleModel {
public:
    /**
     * Constructor
     * @param mc_events MC event monolith
     * @param oscillator Neutrino oscillator
     * @param spline_syst_model Spline systematic model
     */
    SampleModel(const MCEventMonolith& mc_events,
                Oscillator& oscillator, 
                SplineSystematicModel& spline_syst_model);

    /**
     * Set bin variables for analysis
     * @param bin_variables Indices of kinematic variables to use
     */
    void set_bin_variables(const Eigen::VectorXi& bin_variables);

    /**
     * Initialize MC indices for spline mapping
     */
    void initialise_mc_indices();

    /**
     * Reweight events with oscillation and systematic parameters
     * @param osc_params Oscillation parameters
     * @param syst_params Systematic parameters  
     * @return Reweighted event data
     */
    Eigen::VectorXd reweight(const Eigen::VectorXd& osc_params, 
                            const Eigen::VectorXd& syst_params);

    /**
     * Get MC event monolith
     */
    const Eigen::MatrixXd& get_mc_monolith() const { return mc_events_.get_monolith(); }

    /**
     * Get MC indices
     */
    const Eigen::MatrixXi& get_mc_indices() const;

private:
    const MCEventMonolith& mc_events_;
    Oscillator& oscillator_;
    SplineSystematicModel& spline_syst_model_;
    
    Eigen::VectorXi bin_variables_;
    Eigen::MatrixXi mc_indices_;
    bool initialized_ = false;
};

} // namespace magpy

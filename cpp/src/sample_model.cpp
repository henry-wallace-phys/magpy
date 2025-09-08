#include "sample_model.h"

namespace magpy {

SampleModel::SampleModel(const MCEventMonolith& mc_events,
                        Oscillator& oscillator, 
                        SplineSystematicModel& spline_syst_model)
    : mc_events_(mc_events), oscillator_(oscillator), spline_syst_model_(spline_syst_model) {
}

void SampleModel::set_bin_variables(const Eigen::VectorXi& bin_variables) {
    bin_variables_ = bin_variables;
}

void SampleModel::initialise_mc_indices() {
    if (bin_variables_.size() == 0) {
        throw MagpyInvalidObjectError("Bin variables not set. Please set bin variables before initialising MC indices.");
    }
    
    mc_indices_ = spline_syst_model_.get_monolith_splines(mc_events_, bin_variables_);
    initialized_ = true;
}

const Eigen::MatrixXi& SampleModel::get_mc_indices() const {
    if (!initialized_) {
        throw MagpyInvalidObjectError("MC indices not initialized. Call initialise_mc_indices() first.");
    }
    return mc_indices_;
}

Eigen::VectorXd SampleModel::reweight(const Eigen::VectorXd& osc_params, 
                                    const Eigen::VectorXd& syst_params) {
    if (!initialized_) {
        throw MagpyInvalidObjectError("Sample model not initialized. Call initialise_mc_indices() first.");
    }
    
    // Calculate oscillation probabilities
    const auto& monolith = mc_events_.get_monolith();
    Eigen::VectorXd energies = monolith.col(static_cast<int>(MCEventIndices::TRUE_NEUTRINO_ENERGY));
    Eigen::VectorXi start_nu = monolith.col(static_cast<int>(MCEventIndices::START_NU)).cast<int>();
    Eigen::VectorXi end_nu = monolith.col(static_cast<int>(MCEventIndices::END_NU)).cast<int>();
    
    oscillator_.set_energy_osc(energies, start_nu, end_nu);
    Eigen::VectorXd osc_weights = oscillator_.calc_probability(osc_params);
    
    // Get event weights
    Eigen::VectorXd event_weights = monolith.col(static_cast<int>(MCEventIndices::WEIGHT));
    
    // Apply oscillation weights
    Eigen::VectorXd base_weights = event_weights.cwiseProduct(osc_weights);
    
    // Apply systematic reweighting
    return spline_syst_model_.reweight(syst_params, base_weights);
}

} // namespace magpy

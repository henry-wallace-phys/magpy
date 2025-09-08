#pragma once

#include <Eigen/Dense>
#include <vector>

namespace magpy {

/**
 * Neutrino type enumeration matching Python implementation
 */
enum class NuType {
    ELECTRON = 12,
    MUON = 14, 
    TAU = 16
};

/**
 * MC Event indices matching Python MCEventIndices
 */
enum class MCEventIndices {
    TRUE_NEUTRINO_ENERGY = 0,
    TRUE_Q2 = 1,
    RECO_NEUTRINO_ENERGY = 2,
    INTERACTION_MODE = 3,
    START_NU = 4,
    END_NU = 5,
    TARGET = 6,
    WEIGHT = 7,
    NENTRIES = 8,
    DUMMY = -1
};

/**
 * Single Monte Carlo event
 */
struct MCEvent {
    double true_neutrino_energy;
    double true_q2;
    double reco_neutrino_energy;
    int interaction_mode;
    int target;
    int start_nu;
    int end_nu;
    double weight;

    /**
     * Convert to array representation
     */
    Eigen::VectorXd to_array() const {
        Eigen::VectorXd result(8);
        result << true_neutrino_energy, true_q2, reco_neutrino_energy,
                  interaction_mode, start_nu, end_nu, target, weight;
        return result;
    }
};

/**
 * Monolithic MC event storage for vectorized operations
 * Equivalent to Python MCEventMonolith
 */
class MCEventMonolith {
public:
    /**
     * Constructor
     * @param events Vector of individual MC events
     */
    MCEventMonolith(const std::vector<MCEvent>& events);

    /**
     * Get the monolith matrix
     */
    const Eigen::MatrixXd& get_monolith() const { return monolith_; }

    /**
     * Get number of events
     */
    size_t size() const { return monolith_.rows(); }

    /**
     * Get event data for specific column
     */
    Eigen::VectorXd get_column(MCEventIndices index) const {
        return monolith_.col(static_cast<int>(index));
    }

private:
    Eigen::MatrixXd monolith_;
};

} // namespace magpy

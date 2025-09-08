#include "mc_event.h"

namespace magpy {

MCEventMonolith::MCEventMonolith(const std::vector<MCEvent>& events) {
    const size_t n_events = events.size();
    const size_t n_cols = 8; // Number of event properties
    
    monolith_.resize(n_events, n_cols);
    
    for (size_t i = 0; i < n_events; ++i) {
        const auto& event = events[i];
        monolith_(i, static_cast<int>(MCEventIndices::TRUE_NEUTRINO_ENERGY)) = event.true_neutrino_energy;
        monolith_(i, static_cast<int>(MCEventIndices::TRUE_Q2)) = event.true_q2;
        monolith_(i, static_cast<int>(MCEventIndices::RECO_NEUTRINO_ENERGY)) = event.reco_neutrino_energy;
        monolith_(i, static_cast<int>(MCEventIndices::INTERACTION_MODE)) = event.interaction_mode;
        monolith_(i, static_cast<int>(MCEventIndices::START_NU)) = event.start_nu;
        monolith_(i, static_cast<int>(MCEventIndices::END_NU)) = event.end_nu;
        monolith_(i, static_cast<int>(MCEventIndices::TARGET)) = event.target;
        monolith_(i, static_cast<int>(MCEventIndices::WEIGHT)) = event.weight;
    }
}

} // namespace magpy

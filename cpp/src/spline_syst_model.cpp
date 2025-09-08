#include "spline_syst_model.h"
#include <algorithm>
#include <unordered_set>

namespace magpy {

SplineSystematicModel::SplineSystematicModel(SplineMonolith& splines, 
                                           const std::vector<Systematic>& systematics)
    : spline_monolith_(splines), systematics_(systematics) {
}

void SplineSystematicModel::setup_splines(const BinHandler& bin_handler) {
    std::vector<std::vector<int>> out_list;
    
    // Process each systematic
    for (size_t isyst = 0; isyst < systematics_.size(); ++isyst) {
        const auto& syst = systematics_[isyst];
        
        for (int mode : syst.modes) {
            if (syst.syst_type == "spline") {
                // For spline systematics, create entries for each bin
                const auto& bin_indices = bin_handler.get_bin_indices();
                for (int i = 0; i < bin_indices.rows(); ++i) {
                    std::vector<int> entry;
                    entry.push_back(static_cast<int>(isyst));  // systematic index
                    entry.push_back(i);                       // spline index (simplified)
                    entry.push_back(mode);                    // interaction mode
                    
                    // Add bin indices
                    for (int j = 0; j < bin_indices.cols(); ++j) {
                        entry.push_back(bin_indices(i, j));
                    }
                    out_list.push_back(entry);
                }
            } else if (syst.syst_type == "norm") {
                // For normalization systematics
                if (norm_systematics_.find(static_cast<int>(isyst)) == norm_systematics_.end()) {
                    norm_systematics_[static_cast<int>(isyst)] = std::unordered_map<int, int>();
                }
                
                if (norm_systematics_[static_cast<int>(isyst)].find(mode) == 
                    norm_systematics_[static_cast<int>(isyst)].end()) {
                    
                    // Create normalization spline
                    Eigen::VectorXd x(2);
                    Eigen::VectorXd y(2);
                    x << syst.range.first, syst.range.second;
                    y << syst.range.first, syst.range.second;
                    
                    Spline norm_spline(x, y);
                    spline_monolith_.add_spline(norm_spline);
                    
                    int spline_idx = spline_monolith_.size() - 1;
                    norm_systematics_[static_cast<int>(isyst)][mode] = spline_idx;
                }
            }
        }
    }
    
    // Convert to Eigen matrix
    if (!out_list.empty()) {
        const size_t n_rows = out_list.size();
        const size_t n_cols = out_list[0].size();
        index_tensor_.resize(n_rows, n_cols);
        
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                index_tensor_(i, j) = out_list[i][j];
            }
        }
        
        // Setup spline mapping
        Eigen::MatrixXi spline_syst_map(n_rows, 2);
        for (size_t i = 0; i < n_rows; ++i) {
            spline_syst_map(i, 0) = index_tensor_(i, 0);  // systematic index
            spline_syst_map(i, 1) = index_tensor_(i, 1);  // spline index
        }
        spline_monolith_.map_splines_to_syst(spline_syst_map);
    }
}

Eigen::MatrixXi SplineSystematicModel::get_monolith_splines(const MCEventMonolith& mc_events, 
                                                          const Eigen::VectorXi& bin_indices) {
    mc_event_monolith_ = &mc_events;
    
    // Store bin configuration
    bins_.resize(bin_indices.size() + 1);
    bins_[0] = static_cast<int>(MCEventIndices::INTERACTION_MODE);
    for (int i = 0; i < bin_indices.size(); ++i) {
        bins_[i + 1] = bin_indices[i];
    }
    
    // For now, return a simple mapping - this would need more sophisticated logic
    // to match the Python implementation's complex bin matching
    const int n_events = mc_events.size();
    event_spline_pairs_.resize(n_events, 2);
    
    for (int i = 0; i < n_events; ++i) {
        event_spline_pairs_(i, 0) = i;      // event index
        event_spline_pairs_(i, 1) = i % spline_monolith_.size();  // spline index (simplified)
    }
    
    // Mark all events as valid for now
    valid_events_.resize(n_events);
    for (int i = 0; i < n_events; ++i) {
        valid_events_[i] = i;
    }
    
    return event_spline_pairs_;
}

Eigen::VectorXd SplineSystematicModel::get_weights_only(const Eigen::VectorXd& syst_values) {
    // Evaluate all splines
    Eigen::VectorXd all_spline_weights = spline_monolith_(syst_values);
    
    // Get weights for event-spline pairs
    const int n_events = valid_events_.size();
    Eigen::VectorXd event_weights = Eigen::VectorXd::Ones(n_events);
    
    // Apply spline weights
    for (int i = 0; i < event_spline_pairs_.rows(); ++i) {
        int event_idx = event_spline_pairs_(i, 0);
        int spline_idx = event_spline_pairs_(i, 1);
        
        if (event_idx < n_events && spline_idx < all_spline_weights.size()) {
            event_weights[event_idx] *= all_spline_weights[spline_idx];
        }
    }
    
    // Apply normalization weights
    apply_normalization_weights(event_weights, all_spline_weights);
    
    return event_weights;
}

void SplineSystematicModel::apply_normalization_weights(Eigen::VectorXd& event_weights, 
                                                       const Eigen::VectorXd& all_spline_weights) {
    if (norm_systematics_.empty() || !mc_event_monolith_) {
        return;
    }
    
    // Get interaction modes for valid events
    Eigen::VectorXi event_modes(valid_events_.size());
    for (int i = 0; i < valid_events_.size(); ++i) {
        int event_idx = valid_events_[i];
        event_modes[i] = static_cast<int>(mc_event_monolith_->get_monolith()(
            event_idx, static_cast<int>(MCEventIndices::INTERACTION_MODE)));
    }
    
    // Apply normalization for each systematic
    for (const auto& syst_pair : norm_systematics_) {
        for (const auto& mode_pair : syst_pair.second) {
            int mode = mode_pair.first;
            int spline_idx = mode_pair.second;
            
            if (spline_idx < all_spline_weights.size()) {
                double norm_weight = all_spline_weights[spline_idx];
                
                // Apply to all events of this mode
                for (int i = 0; i < event_modes.size(); ++i) {
                    if (event_modes[i] == mode) {
                        event_weights[i] *= norm_weight;
                    }
                }
            }
        }
    }
}

Eigen::VectorXd SplineSystematicModel::reweight(const Eigen::VectorXd& syst_values, 
                                               const Eigen::VectorXd& monolith) {
    Eigen::VectorXd combined_weights = get_weights_only(syst_values);
    
    // Apply weights to monolith data
    Eigen::VectorXd result = monolith;
    for (int i = 0; i < std::min(static_cast<int>(combined_weights.size()), 
                                static_cast<int>(result.size())); ++i) {
        result[i] *= combined_weights[i];
    }
    
    return result;
}

} // namespace magpy

/**
 * @file adaptive_metropolis.cpp
 * @brief Implementation of Adaptive Metropolis-Hastings MCMC sampler
 */

#include "adaptive_metropolis.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>

AdaptiveMetropolis::AdaptiveMetropolis(
    LogProbFunction log_prob_fn,
    int n_params,
    const std::vector<std::string>& param_names,
    double initial_step_size,
    int adaptation_interval,
    BoundsFunction bounds_check
) : log_prob_fn_(log_prob_fn),
    bounds_check_(bounds_check),
    n_params_(n_params),
    param_names_(param_names),
    step_size_(initial_step_size),
    adaptation_interval_(adaptation_interval),
    epsilon_(1e-3),  // Standard epsilon for Haario et al.
    adaptation_start_(std::max(100, 2 * n_params)),
    target_accept_rate_(0.234),
    normal_dist_(0.0, 1.0),
    uniform_dist_(0.0, 1.0) {
    
    // Initialize parameter names if not provided
    if (param_names_.empty()) {
        param_names_.resize(n_params_);
        for (int i = 0; i < n_params_; ++i) {
            param_names_[i] = "param_" + std::to_string(i);
        }
    }
    
    // Initialize covariance matrix as scaled identity (Haario et al.)
    // Use the provided step size as the initial scale
    double scale = initial_step_size * initial_step_size;
    cov_matrix_ = Eigen::MatrixXd::Identity(n_params_, n_params_) * scale;
}

MCMCResult AdaptiveMetropolis::sample(
    const Eigen::VectorXd& initial_params,
    int n_samples,
    int n_warmup,
    int thin,
    unsigned int seed
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    rng_.seed(seed);
    
    int total_iterations = n_warmup + n_samples * thin;
    
    // Storage for all samples (including warmup for adaptation)
    std::vector<Eigen::VectorXd> all_samples;
    std::vector<Eigen::VectorXd> normal_proposals_only;  // Only non-flip samples for covariance
    std::vector<double> all_log_probs;
    all_samples.reserve(total_iterations);
    normal_proposals_only.reserve(total_iterations);
    all_log_probs.reserve(total_iterations);
    
    // Current state
    Eigen::VectorXd current_params = initial_params;
    double current_log_prob = log_prob_fn_(current_params);
    
    // Check initial state
    if (!std::isfinite(current_log_prob)) {
        throw std::runtime_error("Initial log probability is not finite: " + std::to_string(current_log_prob));
    }
    
    if (!check_bounds(current_params)) {
        throw std::runtime_error("Initial parameters are outside bounds");
    }
    
    // Counters
    int n_accepted = 0;
    
    std::cout << "🎯 Starting Adaptive Metropolis: " << n_samples << " samples + " << n_warmup << " warmup" << std::endl;
    std::cout << "Initial log probability: " << std::fixed << std::setprecision(6) << current_log_prob << std::endl;
    
    // Progress tracking
    int progress_interval = std::max(1, total_iterations / 100);
    
    // Main sampling loop
    for (int i = 0; i < total_iterations; ++i) {
        // Progress indicator
        if (i % progress_interval == 0) {
            int progress = (i * 100) / total_iterations;
            std::cout << "\rMCMC sampling: " << progress << "%" << std::flush;
        }
        
        // Generate proposal (track if it's a flip for covariance calculation)
        bool is_flip_proposal;
        Eigen::VectorXd proposed_params = propose_step(current_params, is_flip_proposal);
        
        // Check bounds - if out of bounds, reject immediately
        if (!check_bounds(proposed_params)) {
            // Reject proposal - stay at current state
            // This is the standard MCMC approach for handling bounds
        } else {
            try {
                // Evaluate log probability at proposal
                double proposed_log_prob = log_prob_fn_(proposed_params);
                
                if (std::isfinite(proposed_log_prob)) {
                    // Metropolis acceptance criterion
                    double log_alpha = proposed_log_prob - current_log_prob;
                    double alpha = std::exp(std::min(0.0, log_alpha));
                    
                    // Accept or reject
                    if (uniform_dist_(rng_) < alpha) {
                        current_params = proposed_params;
                        current_log_prob = proposed_log_prob;
                        n_accepted++;
                        
                        // Only store non-flip proposals for covariance adaptation
                        if (!is_flip_proposal) {
                            normal_proposals_only.push_back(current_params);
                        }
                    } else {
                        // Rejected - still store current state if it came from normal proposal
                        if (!is_flip_proposal) {
                            normal_proposals_only.push_back(current_params);
                        }
                    }
                }
            } catch (const std::exception& e) {
                // If evaluation fails, reject proposal
            }
        }
        
        // Store sample
        all_samples.push_back(current_params);
        all_log_probs.push_back(current_log_prob);
        
        // Adaptation during warmup - use only normal proposals for covariance
        if (i < n_warmup && (i + 1) % adaptation_interval_ == 0 && normal_proposals_only.size() > 50) {
            // Build matrix from normal proposals only
            int n_normal = normal_proposals_only.size();
            int start_idx = std::max(0, n_normal - 1000);  // Use last 1000 normal proposals
            
            Eigen::MatrixXd normal_samples(n_normal - start_idx, n_params_);
            for (int j = start_idx; j < n_normal; ++j) {
                normal_samples.row(j - start_idx) = normal_proposals_only[j];
            }
            
            adapt_covariance(normal_samples, i + 1);
            
            // Adapt step size
            double recent_accept_rate = static_cast<double>(n_accepted) / (i + 1);
            adapt_step_size(recent_accept_rate, i + 1);
        }
    }
    
    std::cout << std::endl; // New line after progress
    
    // Extract final samples (post-warmup, thinned)
    Eigen::MatrixXd samples(n_samples, n_params_);
    Eigen::VectorXd log_probs(n_samples);
    
    int sample_idx = 0;
    for (int i = n_warmup; i < total_iterations && sample_idx < n_samples; i += thin) {
        samples.row(sample_idx) = all_samples[i];
        log_probs(sample_idx) = all_log_probs[i];
        sample_idx++;
    }
    
    // Calculate final acceptance rate
    double final_accept_rate = static_cast<double>(n_accepted) / total_iterations;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_time = duration.count() / 1000.0;
    
    std::cout << "✅ Sampling completed!" << std::endl;
    std::cout << "   Acceptance rate: " << std::fixed << std::setprecision(3) << final_accept_rate << std::endl;
    std::cout << "   Final step size: " << std::fixed << std::setprecision(6) << step_size_ << std::endl;
    std::cout << "   Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
    
    return MCMCResult{
        samples,
        log_probs,
        final_accept_rate,
        param_names_,
        cov_matrix_,
        step_size_,
        total_time
    };
}

Eigen::VectorXd AdaptiveMetropolis::propose_step(const Eigen::VectorXd& current_params, bool& is_flip) {
    // Decide whether to do mass ordering flip (50% chance - flat prior on mass ordering)
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    is_flip = uniform(rng_) < 0.5;
    
    if (is_flip && current_params.size() >= 6) {
        // Mass ordering flip: just flip dm32 sign (parameter 5)
        Eigen::VectorXd flipped_params = current_params;
        flipped_params(5) = -current_params(5);  // Flip dm32
        return flipped_params;
    } else {
        // Not actually a flip if we don't have enough parameters
        is_flip = false;
        // Standard Haario et al. multivariate normal proposal
        Eigen::VectorXd random_step(n_params_);
        for (int i = 0; i < n_params_; ++i) {
            random_step(i) = normal_dist_(rng_);
        }
        
        // Use Cholesky decomposition for multivariate normal
        Eigen::LLT<Eigen::MatrixXd> llt(cov_matrix_);
        if (llt.info() == Eigen::Success) {
            random_step = llt.matrixL() * random_step;
        } else {
            // Fallback: use sqrt of diagonal elements
            for (int i = 0; i < n_params_; ++i) {
                random_step(i) *= std::sqrt(std::max(epsilon_, cov_matrix_(i, i)));
            }
        }
        
        Eigen::VectorXd proposed_params = current_params + random_step;
        
        // Apply periodic boundary condition to deltacp (parameter 3) if it exists
        if (proposed_params.size() >= 4) {
            const double two_pi = 2.0 * M_PI;
            while (proposed_params(3) < 0.0) proposed_params(3) += two_pi;
            while (proposed_params(3) >= two_pi) proposed_params(3) -= two_pi;
        }
        
        return proposed_params;
    }
}

void AdaptiveMetropolis::adapt_covariance(const Eigen::MatrixXd& samples, int iteration) {
    if (iteration < adaptation_start_) {
        return;
    }
    
    // Compute empirical covariance
    Eigen::VectorXd sample_mean = samples.colwise().mean();
    Eigen::MatrixXd centered_samples = samples.rowwise() - sample_mean.transpose();
    Eigen::MatrixXd empirical_cov = centered_samples.transpose() * centered_samples / (samples.rows() - 1);
    
    // Adaptive scaling factor (Haario et al. 2001)
    double s_d = 2.38 * 2.38 / n_params_;  // This is the correct scaling
    
    // More aggressive adaptation early on, then stabilize
    double adaptation_weight = 1.0;
    if (iteration < 1000) {
        adaptation_weight = 0.8;  // Faster adaptation early
    } else if (iteration < 5000) {
        adaptation_weight = 0.5;  // Medium adaptation
    } else {
        adaptation_weight = 0.1;  // Conservative adaptation later
    }
    
    // Weighted update of covariance matrix
    Eigen::MatrixXd target_cov = s_d * (empirical_cov + epsilon_ * Eigen::MatrixXd::Identity(n_params_, n_params_));
    cov_matrix_ = (1.0 - adaptation_weight) * cov_matrix_ + adaptation_weight * target_cov;
}

void AdaptiveMetropolis::adapt_step_size(double accept_rate, int iteration) {
    if (iteration < 100) { // Don't adapt too early
        return;
    }
    
    // More aggressive adaptation rate early on, then reduce
    double adaptation_rate;
    if (iteration < 1000) {
        adaptation_rate = 0.2;  // More aggressive early adaptation
    } else if (iteration < 5000) {
        adaptation_rate = 0.1;
    } else {
        adaptation_rate = std::min(0.05, 100.0 / iteration);  // Gentler later
    }
    
    // Dual averaging for step size (more stable than multiplicative)
    double error = accept_rate - target_accept_rate_;
    step_size_ = step_size_ * std::exp(adaptation_rate * error);
    
    // Keep step size reasonable
    step_size_ = std::max(1e-6, std::min(step_size_, 0.5));
    
    // Don't rescale covariance matrix here - let it adapt independently
}

bool AdaptiveMetropolis::check_bounds(const Eigen::VectorXd& params) {
    if (bounds_check_) {
        return bounds_check_(params);
    }
    return true; // No bounds checking if not provided
}

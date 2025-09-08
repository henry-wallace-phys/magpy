/**
 * @file adaptive_metropolis.h
 * @brief Adaptive Metropolis-Hastings MCMC sampler
 */

#ifndef ADAPTIVE_METROPOLIS_H
#define ADAPTIVE_METROPOLIS_H

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <random>
#include <string>

struct MCMCResult {
    Eigen::MatrixXd samples;        // (n_samples, n_params)
    Eigen::VectorXd log_probs;      // (n_samples,)
    double accept_rate;
    std::vector<std::string> param_names;
    Eigen::MatrixXd covariance_matrix;
    double final_step_size;
    double total_time_seconds;
};

class AdaptiveMetropolis {
public:
    using LogProbFunction = std::function<double(const Eigen::VectorXd&)>;
    using BoundsFunction = std::function<bool(const Eigen::VectorXd&)>;

    AdaptiveMetropolis(
        LogProbFunction log_prob_fn,
        int n_params,
        const std::vector<std::string>& param_names = {},
        double initial_step_size = 0.1,
        int adaptation_interval = 100,
        BoundsFunction bounds_check = nullptr
    );

    MCMCResult sample(
        const Eigen::VectorXd& initial_params,
        int n_samples,
        int n_warmup = 1000,
        int thin = 1,
        unsigned int seed = 42
    );

private:
    LogProbFunction log_prob_fn_;
    BoundsFunction bounds_check_;
    int n_params_;
    std::vector<std::string> param_names_;
    double step_size_;
    int adaptation_interval_;
    double epsilon_;
    int adaptation_start_;
    Eigen::MatrixXd cov_matrix_;
    double target_accept_rate_;

    // Random number generation
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

    Eigen::VectorXd propose_step(const Eigen::VectorXd& current_params, bool& is_flip);
    void adapt_covariance(const Eigen::MatrixXd& samples, int iteration);
    void adapt_step_size(double accept_rate, int iteration);
    bool check_bounds(const Eigen::VectorXd& params);
};

#endif // ADAPTIVE_METROPOLIS_H

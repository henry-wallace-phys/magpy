#pragma once

#include <Eigen/Dense>
#include <vector>
#include "exceptions.h"

namespace magpy {

/**
 * Cubic spline interpolation using Eigen
 * Equivalent to Python Spline class
 */
class Spline {
public:
    /**
     * Constructor
     * @param x Knot positions
     * @param y Knot values
     */
    Spline(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

    /**
     * Evaluate spline at given points
     * @param points Points to evaluate at
     * @return Evaluated values
     */
    Eigen::VectorXd evaluate(const Eigen::VectorXd& points) const;

    /**
     * Evaluate spline at single point
     */
    double evaluate(double point) const;

    /**
     * Check if spline is flat (all y values equal)
     */
    bool is_flat() const { return is_flat_; }

    /**
     * Get length (number of spline coefficients)
     */
    size_t length() const { return is_flat_ ? 0 : x_.size(); }

    /**
     * Get knot positions
     */
    const Eigen::VectorXd& get_x() const { return x_; }

    /**
     * Get knot values  
     */
    const Eigen::VectorXd& get_y() const { return y_; }

private:
    Eigen::VectorXd x_, y_;
    Eigen::VectorXd a_, b_, c_, d_; // Spline coefficients
    bool is_flat_;

    void compute_coefficients();
};

/**
 * Monolithic spline handler for vectorized operations
 * Equivalent to Python SplineMonolith
 */
class SplineMonolith {
public:
    /**
     * Constructor
     * @param splines Vector of splines
     */
    SplineMonolith(const std::vector<Spline>& splines);

    /**
     * Add a spline to the monolith
     */
    void add_spline(const Spline& spline);

    /**
     * Map splines to systematics
     * @param spline_syst_map Mapping matrix [syst_index, spline_index]
     */
    void map_splines_to_syst(const Eigen::MatrixXi& spline_syst_map);

    /**
     * Evaluate all splines at given systematic parameter values
     * @param x Systematic parameter values
     * @return Evaluated weights for all splines
     */
    Eigen::VectorXd operator()(const Eigen::VectorXd& x);

    /**
     * Get number of splines
     */
    size_t size() const { return splines_.size(); }

private:
    std::vector<Spline> splines_;
    Eigen::MatrixXi spline_syst_map_;
    bool mapped_;

    // Optimization data structures
    std::vector<std::vector<int>> syst_to_splines_;
    Eigen::VectorXd cached_weights_;
};

} // namespace magpy

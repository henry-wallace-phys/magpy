#include "spline_handler.h"
#include <algorithm>
#include <cmath>

namespace magpy {

Spline::Spline(const Eigen::VectorXd& x, const Eigen::VectorXd& y) : x_(x), y_(y) {
    if (x.size() != y.size()) {
        throw MagpySplineException("x and y vectors must have the same size");
    }
    
    // Check if spline is flat
    is_flat_ = (y.array() == y[0]).all();
    
    if (!is_flat_) {
        compute_coefficients();
    }
}

void Spline::compute_coefficients() {
    const int n = x_.size();
    if (n < 2) {
        throw MagpySplineException("Need at least 2 points for spline interpolation");
    }
    
    // Natural cubic spline computation
    Eigen::VectorXd h(n-1);
    for (int i = 0; i < n-1; ++i) {
        h[i] = x_[i+1] - x_[i];
    }
    
    // Set up tridiagonal system for second derivatives
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    
    // Natural boundary conditions
    A(0, 0) = 1.0;
    A(n-1, n-1) = 1.0;
    
    for (int i = 1; i < n-1; ++i) {
        A(i, i-1) = h[i-1];
        A(i, i) = 2.0 * (h[i-1] + h[i]);
        A(i, i+1) = h[i];
        b[i] = 6.0 * ((y_[i+1] - y_[i]) / h[i] - (y_[i] - y_[i-1]) / h[i-1]);
    }
    
    // Solve for second derivatives
    Eigen::VectorXd c = A.colPivHouseholderQr().solve(b);
    
    // Compute spline coefficients
    a_.resize(n-1);
    b_.resize(n-1);
    c_.resize(n-1);
    d_.resize(n-1);
    
    for (int i = 0; i < n-1; ++i) {
        a_[i] = (c[i+1] - c[i]) / (6.0 * h[i]);
        b_[i] = c[i] / 2.0;
        c_[i] = (y_[i+1] - y_[i]) / h[i] - h[i] * (2.0 * c[i] + c[i+1]) / 6.0;
        d_[i] = y_[i];
    }
}

double Spline::evaluate(double point) const {
    if (is_flat_) {
        return y_[0];
    }
    
    // Find the right interval
    int i = std::lower_bound(x_.data(), x_.data() + x_.size(), point) - x_.data();
    i = std::max(0, std::min(i - 1, static_cast<int>(x_.size()) - 2));
    
    double dx = point - x_[i];
    return a_[i] * dx * dx * dx + b_[i] * dx * dx + c_[i] * dx + d_[i];
}

Eigen::VectorXd Spline::evaluate(const Eigen::VectorXd& points) const {
    Eigen::VectorXd result(points.size());
    
    if (is_flat_) {
        result.setConstant(y_[0]);
        return result;
    }
    
    for (int j = 0; j < points.size(); ++j) {
        result[j] = evaluate(points[j]);
    }
    
    return result;
}

SplineMonolith::SplineMonolith(const std::vector<Spline>& splines) 
    : splines_(splines), mapped_(false) {
    cached_weights_.resize(splines.size());
}

void SplineMonolith::add_spline(const Spline& spline) {
    splines_.push_back(spline);
    cached_weights_.conservativeResize(splines_.size());
}

void SplineMonolith::map_splines_to_syst(const Eigen::MatrixXi& spline_syst_map) {
    spline_syst_map_ = spline_syst_map;
    mapped_ = true;
    
    // Build systematic to splines mapping for efficient evaluation
    if (spline_syst_map.rows() > 0) {
        int max_syst = spline_syst_map.col(0).maxCoeff() + 1;
        syst_to_splines_.resize(max_syst);
        
        for (int i = 0; i < spline_syst_map.rows(); ++i) {
            int syst_idx = spline_syst_map(i, 0);
            int spline_idx = spline_syst_map(i, 1);
            syst_to_splines_[syst_idx].push_back(spline_idx);
        }
    }
}

Eigen::VectorXd SplineMonolith::operator()(const Eigen::VectorXd& x) {
    if (!mapped_) {
        throw MagpySplineException("SplineMonolith not mapped to systematics");
    }
    
    // Evaluate all splines at their corresponding systematic parameter values
    for (size_t syst_idx = 0; syst_idx < syst_to_splines_.size() && syst_idx < static_cast<size_t>(x.size()); ++syst_idx) {
        double param_value = x[syst_idx];
        
        for (int spline_idx : syst_to_splines_[syst_idx]) {
            if (spline_idx < static_cast<int>(splines_.size())) {
                cached_weights_[spline_idx] = splines_[spline_idx].evaluate(param_value);
            }
        }
    }
    
    return cached_weights_;
}

} // namespace magpy

#include "bin_handler.h"
#include <algorithm>
#include <iostream>

namespace magpy {

BinHandler::BinHandler(const std::vector<Eigen::VectorXd>& bin_edges) : bin_edges_(bin_edges) {
    for (const auto& edges : bin_edges_) {
        bin_edge_dims_.push_back(edges.size() - 1);
    }
    generate_bin_tuples();
}

void BinHandler::generate_bin_tuples() {
    // Calculate total number of bins
    size_t total_bins = 1;
    for (int dim : bin_edge_dims_) {
        total_bins *= dim;
    }
    
    // Generate all combinations of bin indices
    std::vector<std::vector<int>> indices;
    indices.reserve(total_bins);
    
    std::vector<int> current_indices(bin_edge_dims_.size(), 0);
    
    // Generate all combinations using nested loops approach
    std::function<void(size_t)> generate = [&](size_t dim) {
        if (dim == bin_edge_dims_.size()) {
            indices.push_back(current_indices);
            return;
        }
        for (int i = 0; i < bin_edge_dims_[dim]; ++i) {
            current_indices[dim] = i;
            generate(dim + 1);
        }
    };
    generate(0);
    
    // Create bin edge matrices [n_bins, 2] for each dimension
    const size_t n_dims = bin_edges_.size();
    bin_edge_matrices_.resize(n_dims);
    bin_edge_indices_.resize(total_bins, n_dims);
    
    for (size_t j = 0; j < n_dims; ++j) {
        bin_edge_matrices_[j].resize(total_bins, 2);
    }
    
    for (size_t i = 0; i < total_bins; ++i) {
        for (size_t j = 0; j < n_dims; ++j) {
            int idx = indices[i][j];
            bin_edge_matrices_[j](i, 0) = bin_edges_[j][idx];
            bin_edge_matrices_[j](i, 1) = bin_edges_[j][idx + 1];
            bin_edge_indices_(i, j) = idx;
        }
    }
}

Eigen::MatrixXi BinHandler::find_bin(const Eigen::MatrixXd& kinematics) const {
    if (kinematics.cols() != static_cast<int>(bin_edges_.size())) {
        throw std::invalid_argument("Kinematic tensor has wrong number of columns");
    }
    
    const int n_events = kinematics.rows();
    const int n_dims = kinematics.cols();
    Eigen::MatrixXi bin_indices(n_events, n_dims);
    
    for (int i = 0; i < n_events; ++i) {
        for (int j = 0; j < n_dims; ++j) {
            double value = kinematics(i, j);
            const Eigen::VectorXd& edges = bin_edges_[j];
            
            // Find bin using binary search
            auto it = std::lower_bound(edges.data(), edges.data() + edges.size() - 1, value);
            int bin_idx = std::max(0, static_cast<int>(it - edges.data()) - 1);
            bin_idx = std::min(bin_idx, static_cast<int>(edges.size()) - 2);
            
            bin_indices(i, j) = bin_idx;
        }
    }
    
    return bin_indices;
}

std::vector<Eigen::MatrixXd> BinHandler::get_bin_from_int(const Eigen::VectorXi& indices) const {
    const int n_events = indices.size();
    const int n_dims = bin_edges_.size();
    
    std::vector<Eigen::MatrixXd> result(n_dims);
    for (int j = 0; j < n_dims; ++j) {
        result[j].resize(n_events, 2);
    }
    
    for (int i = 0; i < n_events; ++i) {
        int idx = indices[i];
        if (idx >= 0 && idx < bin_edge_matrices_[0].rows()) {
            for (int j = 0; j < n_dims; ++j) {
                result[j](i, 0) = bin_edge_matrices_[j](idx, 0);
                result[j](i, 1) = bin_edge_matrices_[j](idx, 1);
            }
        } else {
            // Invalid index, set to -1
            for (int j = 0; j < n_dims; ++j) {
                result[j](i, 0) = -1.0;
                result[j](i, 1) = -1.0;
            }
        }
    }
    
    return result;
}

} // namespace magpy

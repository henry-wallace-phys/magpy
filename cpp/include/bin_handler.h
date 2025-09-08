#pragma once

#include <Eigen/Dense>
#include <vector>

namespace magpy {

/**
 * Bin handler for kinematic binning operations
 * Equivalent to Python BinHandler class with Eigen vectorization
 */
class BinHandler {
public:
    /**
     * Constructor
     * @param bin_edges Vector of bin edge arrays for each dimension
     */
    BinHandler(const std::vector<Eigen::VectorXd>& bin_edges);

    /**
     * Find bin indices for kinematic values
     * @param kinematics Matrix where each row is an event and columns are kinematic variables
     * @return Matrix of bin indices
     */
    Eigen::MatrixXi find_bin(const Eigen::MatrixXd& kinematics) const;

    /**
     * Get bin edges from integer indices
     * @param indices Vector of linear bin indices
     * @return Matrix where each row represents [lower_bound, upper_bound] for each dimension
     */
    std::vector<Eigen::MatrixXd> get_bin_from_int(const Eigen::VectorXi& indices) const;

    /**
     * Get bin edge matrix for dimension
     */
    const Eigen::MatrixXd& get_bin_edge_matrix(int dim) const { return bin_edge_matrices_[dim]; }

    /**
     * Get bin indices
     */
    const Eigen::MatrixXi& get_bin_indices() const { return bin_edge_indices_; }

    /**
     * Get bin edges for specific dimension
     */
    const Eigen::VectorXd& get_bin_edges(int dim) const { return bin_edges_[dim]; }

private:
    std::vector<Eigen::VectorXd> bin_edges_;
    std::vector<int> bin_edge_dims_;
    std::vector<Eigen::MatrixXd> bin_edge_matrices_;  // Store matrices instead of tensor
    Eigen::MatrixXi bin_edge_indices_;

    void generate_bin_tuples();
};

} // namespace magpy

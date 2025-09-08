#pragma once

#include "mc_event.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <Eigen/Dense>

// Forward declarations for ROOT classes
class TFile;
class TTree;

namespace magpy {

// ROOT file reader for MC data
class MCFile {
public:
    // Constructor - opens ROOT file and tree
    MCFile(const std::string& file_name, const std::string& tree_name = "mc_tree");
    
    // Destructor - closes file
    ~MCFile();
    
    // Configure branch mappings
    void set_mc_branch(MCEventIndices index, const std::string& branch_name);
    void set_mc_const(MCEventIndices index, double value);
    
    // Load events into monolith
    void fill_monolith();
    
    // Access the monolith
    const MCEventMonolith& get_monolith() const;
    MCEventMonolith& get_monolith();
    
    // File information
    bool is_open() const { return file_ != nullptr; }
    size_t get_n_entries() const;
    std::vector<std::string> get_branch_names() const;
    
    // Branch information
    bool has_branch(const std::string& branch_name) const;
    
private:
    static const std::string CONST_BRANCH;
    
    // ROOT file and tree
    std::unique_ptr<TFile> file_;
    TTree* tree_;  // Owned by file_, don't delete
    std::string file_name_;
    std::string tree_name_;
    
    // Branch configuration
    std::vector<std::string> branch_names_;    // Maps MCEventIndices to branch names
    std::vector<double> const_values_;         // Constant values for indices
    
    // Data storage
    std::unique_ptr<MCEventMonolith> monolith_;
    bool monolith_filled_;
    
    // Helper methods
    void open_file();
    void close_file();
    double get_const_or_branch_value(size_t entry, MCEventIndices index) const;
    void setup_branch_readers();
    
    // Branch data buffers for reading
    std::map<std::string, std::vector<double>> double_branches_;
    std::map<std::string, std::vector<int>> int_branches_;
    std::map<std::string, std::vector<float>> float_branches_;
};

} // namespace magpy

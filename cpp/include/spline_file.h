#pragma once

#include "spline_handler.h"
#include "bin_handler.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

// Forward declarations for ROOT classes
class TFile;
class TGraph;
class TH3D;

namespace magpy {

// ROOT file reader for spline data
class SplineFile {
public:
    // Constructor - opens ROOT file containing splines
    SplineFile(const std::string& file_name);
    
    // Destructor - closes file
    ~SplineFile();
    
    // Access spline data
    const SplineMonolith& get_monolith() const;
    SplineMonolith& get_monolith();
    
    // Access spline names
    const std::vector<std::string>& get_spline_names() const;
    
    // Access bin handler
    const BinHandler& get_bin_handler() const;
    
    // File information
    bool is_open() const { return file_ != nullptr; }
    size_t get_n_splines() const { return spline_names_.size(); }
    
    // Access individual splines by name
    bool has_spline(const std::string& name) const;
    const Spline& get_spline(const std::string& name) const;
    
private:
    static const std::string BINNING_HIST_STR;
    
    // ROOT file
    std::unique_ptr<TFile> file_;
    std::string file_name_;
    
    // Spline data
    std::unique_ptr<SplineMonolith> monolith_;
    std::vector<Spline> splines_;  // Store individual splines for access
    std::vector<std::string> spline_names_;
    std::map<std::string, size_t> spline_name_to_index_;
    
    // Binning information
    std::unique_ptr<BinHandler> bin_handler_;
    
    // Helper methods
    void open_file();
    void close_file();
    void load_splines();
    void load_binning_histogram();
    
    // ROOT object reading helpers
    std::pair<Eigen::VectorXd, Eigen::VectorXd> read_tgraph(TGraph* graph) const;
    std::vector<Eigen::VectorXd> read_histogram_bins(TH3D* hist) const;
};

} // namespace magpy

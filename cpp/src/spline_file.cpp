#include "spline_file.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifdef HAS_ROOT
#include <TFile.h>
#include <TGraph.h>
#include <TH3D.h>
#include <TKey.h>
#include <TList.h>
#include <TClass.h>
#endif

namespace magpy {

const std::string SplineFile::BINNING_HIST_STR = "dev_tmp.0.0";

// SplineFile implementation
SplineFile::SplineFile(const std::string& file_name)
    : file_(nullptr), file_name_(file_name), monolith_(nullptr), bin_handler_(nullptr) {
    
#ifdef HAS_ROOT
    open_file();
    load_splines();
    load_binning_histogram();
#else
    throw std::runtime_error("ROOT support not compiled in. Please rebuild with ROOT to use spline files.");
#endif
}

SplineFile::~SplineFile() {
    close_file();
}

void SplineFile::open_file() {
#ifdef HAS_ROOT
    std::cout << "Opening spline ROOT file: " << file_name_ << std::endl;
    
    file_ = std::make_unique<TFile>(file_name_.c_str(), "READ");
    
    if (!file_ || file_->IsZombie()) {
        throw std::runtime_error("Failed to open spline ROOT file: " + file_name_);
    }
    
    std::cout << "Successfully opened spline file" << std::endl;
#else
    throw std::runtime_error("ROOT support not available");
#endif
}

void SplineFile::close_file() {
#ifdef HAS_ROOT
    if (file_) {
        file_->Close();
        file_.reset();
        std::cout << "Closed spline ROOT file: " << file_name_ << std::endl;
    }
#endif
}

void SplineFile::load_splines() {
#ifdef HAS_ROOT
    if (!file_) {
        throw std::runtime_error("File not open");
    }
    
    std::cout << "Loading splines from ROOT file..." << std::endl;
    
    std::vector<Spline> splines;
    spline_names_.clear();
    spline_name_to_index_.clear();
    
    // Get list of keys in the file
    TList* keys = file_->GetListOfKeys();
    if (!keys) {
        throw std::runtime_error("Failed to get list of keys from spline file");
    }
    
    size_t spline_count = 0;
    
    // Iterate through all keys in the file
    TIter next(keys);
    TKey* key;
    while ((key = static_cast<TKey*>(next()))) {
        // Check if this is a TGraph
        if (std::string(key->GetClassName()) == "TGraph") {
            TGraph* graph = dynamic_cast<TGraph*>(key->ReadObj());
            if (!graph) {
                std::cout << "Warning: Failed to read TGraph: " << key->GetName() << std::endl;
                continue;
            }
            
            try {
                // Read the graph data
                auto [x_data, y_data] = read_tgraph(graph);
                
                // Create spline
                splines.emplace_back(x_data, y_data);
                
                // Store name and index
                std::string spline_name = key->GetName();
                spline_names_.push_back(spline_name);
                spline_name_to_index_[spline_name] = spline_count;
                spline_count++;
                
                if (spline_count % 100 == 0) {
                    std::cout << "Loaded " << spline_count << " splines..." << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cout << "Warning: Failed to create spline from " << key->GetName() 
                          << ": " << e.what() << std::endl;
                continue;
            }
            
            delete graph;  // Clean up
        }
    }
    
    std::cout << "Successfully loaded " << splines.size() << " splines" << std::endl;
    
    // Store splines for individual access
    splines_ = splines;
    
    // Create monolith
    monolith_ = std::make_unique<SplineMonolith>(std::move(splines));
#else
    throw std::runtime_error("ROOT support not available");
#endif
}

void SplineFile::load_binning_histogram() {
#ifdef HAS_ROOT
    if (!file_) {
        throw std::runtime_error("File not open");
    }
    
    std::cout << "Loading binning histogram: " << BINNING_HIST_STR << std::endl;
    
    // Try to get the binning histogram
    TH3D* hist = dynamic_cast<TH3D*>(file_->Get(BINNING_HIST_STR.c_str()));
    if (!hist) {
        throw std::runtime_error("Binning histogram '" + BINNING_HIST_STR + "' not found in spline file");
    }
    
    try {
        // Read bin edges from histogram
        auto bin_edges = read_histogram_bins(hist);
        
        if (bin_edges.size() != 3) {
            throw std::runtime_error("Expected 3D histogram for binning, got " + std::to_string(bin_edges.size()) + "D");
        }
        
        // Create bin handler
        bin_handler_ = std::make_unique<BinHandler>(bin_edges);
        
        std::cout << "Successfully loaded binning: "
                  << bin_edges[0].size()-1 << " x " 
                  << bin_edges[1].size()-1 << " x " 
                  << bin_edges[2].size()-1 << " bins" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read binning histogram: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("ROOT support not available");
#endif
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> SplineFile::read_tgraph(TGraph* graph) const {
#ifdef HAS_ROOT
    if (!graph) {
        throw std::runtime_error("Null TGraph pointer");
    }
    
    const int n_points = graph->GetN();
    if (n_points <= 0) {
        throw std::runtime_error("TGraph has no points");
    }
    
    Eigen::VectorXd x_data(n_points);
    Eigen::VectorXd y_data(n_points);
    
    // Read points from TGraph
    double* x_array = graph->GetX();
    double* y_array = graph->GetY();
    
    for (int i = 0; i < n_points; ++i) {
        x_data[i] = x_array[i];
        y_data[i] = y_array[i];
    }
    
    return {x_data, y_data};
#else
    throw std::runtime_error("ROOT support not available");
#endif
}

std::vector<Eigen::VectorXd> SplineFile::read_histogram_bins(TH3D* hist) const {
#ifdef HAS_ROOT
    if (!hist) {
        throw std::runtime_error("Null histogram pointer");
    }
    
    std::vector<Eigen::VectorXd> bin_edges;
    
    // X axis
    const int nx = hist->GetNbinsX();
    Eigen::VectorXd x_edges(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        x_edges[i] = hist->GetXaxis()->GetBinLowEdge(i + 1);
    }
    bin_edges.push_back(x_edges);
    
    // Y axis
    const int ny = hist->GetNbinsY();
    Eigen::VectorXd y_edges(ny + 1);
    for (int i = 0; i <= ny; ++i) {
        y_edges[i] = hist->GetYaxis()->GetBinLowEdge(i + 1);
    }
    bin_edges.push_back(y_edges);
    
    // Z axis
    const int nz = hist->GetNbinsZ();
    Eigen::VectorXd z_edges(nz + 1);
    for (int i = 0; i <= nz; ++i) {
        z_edges[i] = hist->GetZaxis()->GetBinLowEdge(i + 1);
    }
    bin_edges.push_back(z_edges);
    
    return bin_edges;
#else
    throw std::runtime_error("ROOT support not available");
#endif
}

const SplineMonolith& SplineFile::get_monolith() const {
    if (!monolith_) {
        throw std::runtime_error("Spline monolith not loaded");
    }
    return *monolith_;
}

SplineMonolith& SplineFile::get_monolith() {
    if (!monolith_) {
        throw std::runtime_error("Spline monolith not loaded");
    }
    return *monolith_;
}

const std::vector<std::string>& SplineFile::get_spline_names() const {
    return spline_names_;
}

const BinHandler& SplineFile::get_bin_handler() const {
    if (!bin_handler_) {
        throw std::runtime_error("Bin handler not loaded");
    }
    return *bin_handler_;
}

bool SplineFile::has_spline(const std::string& name) const {
    return spline_name_to_index_.find(name) != spline_name_to_index_.end();
}

const Spline& SplineFile::get_spline(const std::string& name) const {
    auto it = spline_name_to_index_.find(name);
    if (it == spline_name_to_index_.end()) {
        throw std::runtime_error("Spline '" + name + "' not found");
    }
    return splines_[it->second];
}

} // namespace magpy

#include "mc_file.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>

#ifdef HAS_ROOT
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#endif

namespace magpy {

const std::string MCFile::CONST_BRANCH = "__CONST__";

// MCFile implementation
MCFile::MCFile(const std::string& file_name, const std::string& tree_name)
    : file_(nullptr), tree_(nullptr), file_name_(file_name), tree_name_(tree_name),
      branch_names_(static_cast<int>(MCEventIndices::NENTRIES)),
      const_values_(static_cast<int>(MCEventIndices::NENTRIES)),
      monolith_(nullptr), monolith_filled_(false) {
    
    // Initialize branch names as empty
    std::fill(branch_names_.begin(), branch_names_.end(), "");
    std::fill(const_values_.begin(), const_values_.end(), 0.0);
    
#ifdef HAS_ROOT
    open_file();
#else
    throw std::runtime_error("ROOT support not compiled in. Please rebuild with ROOT or use CSV-based testing.");
#endif
}

MCFile::~MCFile() {
    close_file();
}

void MCFile::open_file() {
#ifdef HAS_ROOT
    std::cout << "Opening ROOT file: " << file_name_ << std::endl;
    
    file_ = std::make_unique<TFile>(file_name_.c_str(), "READ");
    
    if (!file_ || file_->IsZombie()) {
        throw std::runtime_error("Failed to open ROOT file: " + file_name_);
    }
    
    // Get the tree
    tree_ = dynamic_cast<TTree*>(file_->Get(tree_name_.c_str()));
    if (!tree_) {
        throw std::runtime_error("Failed to get TTree '" + tree_name_ + "' from file: " + file_name_);
    }
    
    std::cout << "Successfully opened TTree '" << tree_name_ << "' with " 
              << tree_->GetEntries() << " entries" << std::endl;
#else
    throw std::runtime_error("ROOT support not available");
#endif
}

void MCFile::close_file() {
    if (file_) {
        file_->Close();
        file_.reset();
        tree_ = nullptr;
        std::cout << "Closed ROOT file: " << file_name_ << std::endl;
    }
}

void MCFile::set_mc_branch(MCEventIndices index, const std::string& branch_name) {
    int idx = static_cast<int>(index);
    if (idx >= 0 && idx < static_cast<int>(MCEventIndices::NENTRIES)) {
        branch_names_[idx] = branch_name;
        std::cout << "Set branch mapping: " << idx << " -> " << branch_name << std::endl;
    } else {
        throw std::invalid_argument("Invalid MCEventIndices value");
    }
}

void MCFile::set_mc_const(MCEventIndices index, double value) {
    int idx = static_cast<int>(index);
    if (idx >= 0 && idx < static_cast<int>(MCEventIndices::NENTRIES)) {
        branch_names_[idx] = CONST_BRANCH;
        const_values_[idx] = value;
        std::cout << "Set constant value: " << idx << " -> " << value << std::endl;
    } else {
        throw std::invalid_argument("Invalid MCEventIndices value");
    }
}

size_t MCFile::get_n_entries() const {
    if (!tree_) return 0;
    return static_cast<size_t>(tree_->GetEntries());
}

std::vector<std::string> MCFile::get_branch_names() const {
    std::vector<std::string> names;
    if (!tree_) return names;
    
    TObjArray* branches = tree_->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* branch = dynamic_cast<TBranch*>(branches->At(i));
        if (branch) {
            names.push_back(branch->GetName());
        }
    }
    return names;
}

bool MCFile::has_branch(const std::string& branch_name) const {
    if (!tree_) return false;
    return tree_->GetBranch(branch_name.c_str()) != nullptr;
}

void MCFile::setup_branch_readers() {
    if (!tree_) {
        throw std::runtime_error("Tree not available for branch setup");
    }
    
    // Clear existing buffers
    double_branches_.clear();
    int_branches_.clear();
    float_branches_.clear();
    
    // Set up branch readers for non-constant branches
    for (size_t i = 0; i < branch_names_.size(); ++i) {
        const std::string& branch_name = branch_names_[i];
        
        if (branch_name.empty() || branch_name == CONST_BRANCH) {
            continue;
        }
        
        // Check if branch exists
        if (!has_branch(branch_name)) {
            throw std::runtime_error("Branch '" + branch_name + "' not found in tree");
        }
        
        // Get branch info to determine type
        TBranch* branch = tree_->GetBranch(branch_name.c_str());
        TLeaf* leaf = branch->GetLeaf(branch_name.c_str());
        
        if (!leaf) {
            // If no leaf with same name, get first leaf
            leaf = dynamic_cast<TLeaf*>(branch->GetListOfLeaves()->At(0));
        }
        
        if (!leaf) {
            throw std::runtime_error("Could not get leaf for branch: " + branch_name);
        }
        
        std::string type_name = leaf->GetTypeName();
        
        // Create appropriate buffer based on type
        if (type_name == "Double_t" || type_name == "double") {
            double_branches_[branch_name] = std::vector<double>(1);
            tree_->SetBranchAddress(branch_name.c_str(), &double_branches_[branch_name][0]);
        } else if (type_name == "Float_t" || type_name == "float") {
            float_branches_[branch_name] = std::vector<float>(1);
            tree_->SetBranchAddress(branch_name.c_str(), &float_branches_[branch_name][0]);
        } else if (type_name == "Int_t" || type_name == "int") {
            int_branches_[branch_name] = std::vector<int>(1);
            tree_->SetBranchAddress(branch_name.c_str(), &int_branches_[branch_name][0]);
        } else {
            std::cout << "Warning: Unsupported branch type '" << type_name 
                      << "' for branch '" << branch_name << "', treating as double" << std::endl;
            double_branches_[branch_name] = std::vector<double>(1);
            tree_->SetBranchAddress(branch_name.c_str(), &double_branches_[branch_name][0]);
        }
        
        std::cout << "Set up branch reader for '" << branch_name << "' (type: " << type_name << ")" << std::endl;
    }
}

double MCFile::get_const_or_branch_value(size_t entry, MCEventIndices index) const {
    int idx = static_cast<int>(index);
    const std::string& branch_name = branch_names_[idx];
    
    if (branch_name == CONST_BRANCH) {
        return const_values_[idx];
    }
    
    // Get value from appropriate buffer
    if (double_branches_.count(branch_name)) {
        return double_branches_.at(branch_name)[0];
    } else if (float_branches_.count(branch_name)) {
        return static_cast<double>(float_branches_.at(branch_name)[0]);
    } else if (int_branches_.count(branch_name)) {
        return static_cast<double>(int_branches_.at(branch_name)[0]);
    }
    
    throw std::runtime_error("No data found for branch: " + branch_name);
}

void MCFile::fill_monolith() {
    if (!tree_) {
        throw std::runtime_error("No tree available to fill monolith");
    }
    
    // Check that all branches are configured
    for (size_t i = 0; i < branch_names_.size(); ++i) {
        if (branch_names_[i].empty()) {
            throw std::runtime_error("Not all branches are configured for MCEvent");
        }
    }
    
    // Set up branch readers
    setup_branch_readers();
    
    // Clear existing monolith
    std::vector<MCEvent> events;
    
    const size_t n_entries = get_n_entries();
    std::cout << "Filling MC Event monolith with " << n_entries << " entries..." << std::endl;
    
    // Process events
    for (size_t entry = 0; entry < n_entries; ++entry) {
        if (entry % 10000 == 0) {
            std::cout << "Processing entry " << entry << " / " << n_entries << std::endl;
        }
        
        // Load this entry
        if (tree_->GetEntry(entry) <= 0) {
            std::cout << "Warning: Failed to read entry " << entry << std::endl;
            continue;
        }
        
        try {
            // Extract event data
            MCEvent event;
            event.true_neutrino_energy = get_const_or_branch_value(entry, MCEventIndices::TRUE_NEUTRINO_ENERGY);
            event.true_q2 = get_const_or_branch_value(entry, MCEventIndices::TRUE_Q2);
            event.reco_neutrino_energy = get_const_or_branch_value(entry, MCEventIndices::RECO_NEUTRINO_ENERGY);
            event.interaction_mode = static_cast<int>(get_const_or_branch_value(entry, MCEventIndices::INTERACTION_MODE));
            event.start_nu = static_cast<int>(get_const_or_branch_value(entry, MCEventIndices::START_NU));
            event.end_nu = static_cast<int>(get_const_or_branch_value(entry, MCEventIndices::END_NU));
            event.target = static_cast<int>(get_const_or_branch_value(entry, MCEventIndices::TARGET));
            event.weight = get_const_or_branch_value(entry, MCEventIndices::WEIGHT);
            
            events.push_back(event);
            
        } catch (const std::exception& e) {
            std::cout << "Warning: Error processing entry " << entry << ": " << e.what() << std::endl;
            continue;
        }
    }
    
    // Create monolith from events
    monolith_ = std::make_unique<MCEventMonolith>(events);
    monolith_filled_ = true;
    std::cout << "Successfully filled monolith with " << events.size() << " events" << std::endl;
}

const MCEventMonolith& MCFile::get_monolith() const {
    if (!monolith_filled_ || !monolith_) {
        throw std::runtime_error("Monolith has not been filled yet");
    }
    return *monolith_;
}

MCEventMonolith& MCFile::get_monolith() {
    if (!monolith_filled_ || !monolith_) {
        throw std::runtime_error("Monolith has not been filled yet");
    }
    return *monolith_;
}

} // namespace magpy

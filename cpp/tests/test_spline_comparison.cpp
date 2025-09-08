/**
 * C++ Spline File Comparison Test
 * 
 * Tests SplineFile functionality against Python implementation.
 * Loads ROOT splines and compares metadata and performance.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

#ifdef HAS_ROOT
#include "spline_file.h"
#endif

// JSON parsing 
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <spline_info.json>" << std::endl;
        return 1;
    }

    std::string info_file = argv[1];
    
    try {
        std::cout << "🧪 C++ Spline File Comparison Test" << std::endl;
        
        // Load spline info
        std::ifstream file(info_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open spline info file: " + info_file);
        }
        
        json spline_info;
        file >> spline_info;
        file.close();
        
        std::string spline_file_path = spline_info["spline_file_path"];
        std::vector<std::string> python_spline_names = spline_info["python_spline_names"];
        int python_n_splines = spline_info["python_n_splines"];
        
        std::cout << "Loading splines from: " << spline_file_path << std::endl;
        std::cout << "Python found " << python_n_splines << " splines" << std::endl;
        
#ifdef HAS_ROOT
        // Time the C++ spline loading
        auto start_time = std::chrono::high_resolution_clock::now();
        
        magpy::SplineFile cpp_spline_file(spline_file_path);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto cpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double cpp_time = cpp_duration.count() / 1000000.0;
        
        std::cout << "C++ spline loading completed in " << cpp_time << " seconds" << std::endl;
        
        // Get spline information
        auto spline_names = cpp_spline_file.get_spline_names();
        int cpp_n_splines = cpp_spline_file.get_n_splines();
        
        // Get first 100 spline names for comparison
        std::vector<std::string> cpp_spline_names;
        int names_to_compare = std::min(100, static_cast<int>(spline_names.size()));
        for (int i = 0; i < names_to_compare; ++i) {
            cpp_spline_names.push_back(spline_names[i]);
        }
        
        std::cout << "C++ found " << cpp_n_splines << " splines" << std::endl;
        std::cout << "Comparing first " << names_to_compare << " spline names..." << std::endl;
        
        // Save results
        json results;
        results["n_splines"] = cpp_n_splines;
        results["load_time"] = cpp_time;
        results["spline_names"] = cpp_spline_names;
        results["has_root"] = true;
        
        std::ofstream results_file("cpp_spline_results.json");
        results_file << results.dump(2);
        results_file.close();
        
        std::cout << "✅ Results saved to cpp_spline_results.json" << std::endl;
        
#else
        std::cout << "⚠️  ROOT support not available - skipping spline loading" << std::endl;
        
        // Save minimal results
        json results;
        results["n_splines"] = 0;
        results["load_time"] = 0.0;
        results["spline_names"] = std::vector<std::string>();
        results["has_root"] = false;
        results["error"] = "ROOT support not compiled";
        
        std::ofstream results_file("cpp_spline_results.json");
        results_file << results.dump(2);
        results_file.close();
        
        std::cout << "Results saved (ROOT not available)" << std::endl;
#endif
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}

#include "mc_file.h"
#include "oscillator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

// Simple CSV reader for testing without ROOT
class SimpleCSVFile {
public:
    SimpleCSVFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + filename);
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Skip header
            }
            
            std::stringstream ss(line);
            std::string item;
            std::vector<double> row;
            
            while (std::getline(ss, item, ',')) {
                try {
                    row.push_back(std::stod(item));
                } catch (...) {
                    row.push_back(0.0);
                }
            }
            
            if (row.size() >= 8) {
                magpy::MCEvent event;
                event.true_neutrino_energy = row[0];
                event.true_q2 = row[1];
                event.reco_neutrino_energy = row[2];
                event.interaction_mode = static_cast<int>(row[3]);
                event.start_nu = static_cast<int>(row[4]);
                event.end_nu = static_cast<int>(row[5]);
                event.target = static_cast<int>(row[6]);
                event.weight = row[7];
                
                events_.push_back(event);
            }
        }
        
        std::cout << "Loaded " << events_.size() << " events from CSV" << std::endl;
    }
    
    const std::vector<magpy::MCEvent>& get_events() const { return events_; }
    
private:
    std::vector<magpy::MCEvent> events_;
};

// Function to generate test data
void generate_test_csv(const std::string& filename, size_t n_events) {
    std::ofstream file(filename);
    file << "true_neutrino_energy,true_q2,reco_neutrino_energy,interaction_mode,start_nu,end_nu,target,weight\n";
    
    std::srand(42); // Fixed seed for reproducible data
    
    for (size_t i = 0; i < n_events; ++i) {
        double energy = 0.1 + (10.0 - 0.1) * std::rand() / RAND_MAX;  // 0.1 - 10 GeV
        double q2 = 0.01 + (2.0 - 0.01) * std::rand() / RAND_MAX;     // 0.01 - 2 GeV²
        double reco_energy = energy + 0.1 * (std::rand() / double(RAND_MAX) - 0.5);  // Add some noise
        int mode = 1 + std::rand() % 10;  // Interaction modes 1-10
        int target = (std::rand() % 2) ? 1000060120 : 1000010010;  // Carbon or hydrogen
        
        file << energy << "," << q2 << "," << reco_energy << "," << mode 
             << ",14,14," << target << ",1.0\n";  // muon neutrino -> muon neutrino
    }
    
    std::cout << "Generated " << n_events << " test events in " << filename << std::endl;
}

int main() {
    std::cout << "=== MAGPY C++ MC FILE READING TEST (CSV Mode) ===" << std::endl;
    
    try {
        // Generate test data
        std::string csv_file = "test_mc_events.csv";
        size_t n_events = 50000;
        
        std::cout << "\n--- Generating test data ---" << std::endl;
        generate_test_csv(csv_file, n_events);
        
        std::cout << "\n--- Loading CSV file ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        SimpleCSVFile csv_reader(csv_file);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Loaded CSV in " << duration.count() << " ms" << std::endl;
        
        // Create monolith from events
        std::cout << "\n--- Creating monolith ---" << std::endl;
        magpy::MCEventMonolith monolith(csv_reader.get_events());
        
        std::cout << "Created monolith with " << monolith.size() << " events" << std::endl;
        
        std::cout << "\n--- Data summary ---" << std::endl;
        auto energies = monolith.get_column(magpy::MCEventIndices::TRUE_NEUTRINO_ENERGY);
        auto q2_values = monolith.get_column(magpy::MCEventIndices::TRUE_Q2);
        auto start_nu_dbl = monolith.get_column(magpy::MCEventIndices::START_NU);
        auto end_nu_dbl = monolith.get_column(magpy::MCEventIndices::END_NU);
        
        // Convert to integers
        Eigen::VectorXi start_nu = start_nu_dbl.cast<int>();
        Eigen::VectorXi end_nu = end_nu_dbl.cast<int>();
        
        std::cout << "Energy range: " << energies.minCoeff() << " - " << energies.maxCoeff() << " GeV" << std::endl;
        std::cout << "Q2 range: " << q2_values.minCoeff() << " - " << q2_values.maxCoeff() << " GeV²" << std::endl;
        std::cout << "Start neutrino PDG: " << start_nu[0] << " (all same: " << 
                     (start_nu.array() == start_nu[0]).all() << ")" << std::endl;
        std::cout << "End neutrino PDG: " << end_nu[0] << " (all same: " << 
                     (end_nu.array() == end_nu[0]).all() << ")" << std::endl;
        
        std::cout << "\n--- Testing oscillator integration ---" << std::endl;
        size_t n_test = std::min(size_t(10000), monolith.size());
        
            // Extract data for oscillator
            Eigen::VectorXd test_energies = energies.head(n_test);
            Eigen::VectorXi test_start_nu = start_nu.head(n_test);
            Eigen::VectorXi test_end_nu = end_nu.head(n_test);        // Create oscillator
        magpy::Oscillator osc(1300, 0.5, 3.0, 1000);
        osc.set_energy_osc(test_energies, test_start_nu, test_end_nu);
        
        // Oscillation parameters
        const Eigen::VectorXd osc_params = (Eigen::VectorXd(6) << 
            0.3, 0.02, 0.55, 0.7 * M_PI, 7.5e-5, 2.5e-3).finished();
        
        // Benchmark oscillation calculation
        std::cout << "\n--- Oscillation benchmark ---" << std::endl;
        const int n_iterations = 100;
        
        auto osc_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_iterations; ++i) {
            auto probabilities = osc.calc_probability(osc_params);
        }
        auto osc_end = std::chrono::high_resolution_clock::now();
        
        auto osc_duration = std::chrono::duration_cast<std::chrono::microseconds>(osc_end - osc_start);
        double avg_time_per_call = double(osc_duration.count()) / n_iterations;
        
        std::cout << "Oscillation calculation (" << n_test << " events, " << n_iterations << " iterations):" << std::endl;
        std::cout << "  Average time per call: " << avg_time_per_call << " μs" << std::endl;
        std::cout << "  Time per event: " << avg_time_per_call / n_test << " μs" << std::endl;
        std::cout << "  Events per second: " << (n_test * 1e6) / avg_time_per_call << std::endl;
        
        // Calculate final probabilities
        auto final_probs = osc.calc_probability(osc_params);
        std::cout << "  Probability range: " << final_probs.minCoeff() << " - " << final_probs.maxCoeff() << std::endl;
        std::cout << "  Mean probability: " << final_probs.mean() << std::endl;
        
        std::cout << "\n--- Memory usage ---" << std::endl;
        auto data_matrix = monolith.get_monolith();
        double matrix_mb = (data_matrix.rows() * data_matrix.cols() * sizeof(double)) / (1024.0 * 1024.0);
        std::cout << "Data matrix: " << data_matrix.rows() << " x " << data_matrix.cols() 
                  << " (" << matrix_mb << " MB)" << std::endl;
        
        std::cout << "\n--- Event reweighting simulation ---" << std::endl;
        // Simulate systematic reweighting
        auto weights = monolith.get_column(magpy::MCEventIndices::WEIGHT);
        auto new_weights = weights.array() * final_probs.array();  // Weight by oscillation probability
        
        double original_sum = weights.sum();
        double reweighted_sum = new_weights.sum();
        std::cout << "Original weight sum: " << original_sum << std::endl;
        std::cout << "Reweighted sum: " << reweighted_sum << std::endl;
        std::cout << "Reweighting factor: " << reweighted_sum / original_sum << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 MC file reading test completed successfully!" << std::endl;
    return 0;
}

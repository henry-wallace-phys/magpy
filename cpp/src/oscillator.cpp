#include "oscillator.h"
#include <cmath>
#include <algorithm>

namespace magpy {

Oscillator::Oscillator(double L, double ye, double rho, int n_newton)
    : L_(L), ye_(ye), rho_(rho), n_newton_(n_newton), setup_(false) {
}

void Oscillator::set_energy_osc(const Eigen::VectorXd& energies, 
                                const Eigen::VectorXi& osc_in,
                                const Eigen::VectorXi& osc_out) {
    if (energies.size() != osc_in.size() || energies.size() != osc_out.size()) {
        throw MagpyProbabilityException("All input arrays must have the same length.");
    }

    energies_ = energies;
    osc_in_ = osc_in;
    osc_out_ = osc_out;
    
    setup_vectorized_channels();
    setup_ = true;
}

void Oscillator::setup_vectorized_channels() {
    const int n_events = energies_.size();
    channel_indices_.resize(n_events);
    prob_assigners_.resize(n_events);
    
    // PDG code mapping: 12=nu_e, 14=nu_mu, 16=nu_tau
    auto get_flavor_index = [](int pdg) -> int {
        switch (std::abs(pdg)) {
            case 12: return 0; // electron
            case 14: return 1; // muon  
            case 16: return 2; // tau
            default: throw MagpyProbabilityException("Invalid PDG code");
        }
    };
    
    // Precompute channel indices
    for (int i = 0; i < n_events; ++i) {
        int in_flavor = get_flavor_index(osc_in_[i]);
        int out_flavor = get_flavor_index(osc_out_[i]);
        channel_indices_[i] = in_flavor * 3 + out_flavor;
    }
}

void Oscillator::compute_all_probability_matrices(const Eigen::VectorXd& osc_params) {
    // Placeholder - will implement full vectorized version
}

void Oscillator::vectorized_oscillation(const Eigen::VectorXd& osc_params, Eigen::VectorXd& results) {
    // Extract oscillation parameters (matching Python exactly)
    const double s12sq = osc_params[0];
    const double s13sq = osc_params[1]; 
    const double s23sq = osc_params[2];
    const double delta_cp = osc_params[3];
    const double dmsq21 = osc_params[4];
    const double dmsq31 = osc_params[5];
    
    // Constants (matching Python exactly)
    const double eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4;
    const double YerhoE2a = 1.52588e-4;
    
    // Pre-compute frequently used terms (vectorized)
    const double c13sq = 1.0 - s13sq;
    const double c12sq = 1.0 - s12sq;
    const double c23sq = 1.0 - s23sq;
    
    // Energy-dependent calculations (all vectorized)
    const int n_events = energies_.size();
    
    // PDG code mapping
    auto get_flavor_index = [](int pdg) -> int {
        switch (std::abs(pdg)) {
            case 12: return 0; // electron
            case 14: return 1; // muon  
            case 16: return 2; // tau
            default: throw MagpyProbabilityException("Invalid PDG code");
        }
    };
    
    // Calculate all oscillation probabilities (matching Python implementation exactly)
    for (int i = 0; i < n_events; ++i) {
        const double energy = energies_[i];
        const int osc_in_pdg = osc_in_[i];
        const int osc_out_pdg = osc_out_[i];
        
        // Signed energies (matching Python)
        const double signed_energy = energy * (osc_in_pdg > 0 ? 1.0 : -1.0);
        const double Lover4E = eVsqkm_to_GeV_over4 * L_ / signed_energy;
        const double Amatter = ye_ * rho_ * signed_energy * YerhoE2a;
        
        // Initial values calculation (matching Python exactly)
        const double Ue2sq = c13sq * s12sq;
        const double Ue3sq = s13sq;
        const double Um3sq = c13sq * s23sq;
        const double Um2sq = c12sq * c23sq;
        const double Ut2sq = s13sq * s12sq * s23sq;
        
        // Matter effects (matching Python)
        const double Jrr = std::sqrt(Um2sq * Ut2sq);
        const double sind = std::sin(delta_cp);
        const double cosd = std::cos(delta_cp);
        const double Um2sq_matter = Um2sq + Ut2sq - 2.0 * Jrr * cosd;
        const double Jmatter = 8.0 * Jrr * c13sq * sind;
        
        // Core oscillation calculations (matching Python exactly)
        const double Dmsqee = dmsq31 - s12sq * dmsq21;
        const double A_initial = dmsq21 + dmsq31;
        const double See = A_initial - dmsq21 * Ue2sq - dmsq31 * Ue3sq;
        const double Tmm = dmsq21 * dmsq31;
        const double Tee = Tmm * (1.0 - Ue3sq - Ue2sq);
        const double C = Amatter * Tee;
        const double A = A_initial + Amatter;
        
        // Lambda3 calculation without Newton iterations (matching Python)
        const double xmat = Amatter / Dmsqee;
        const double xmat_minus_1 = xmat - 1.0;
        const double sqrt_term = std::sqrt(xmat_minus_1 * xmat_minus_1 + 4.0 * s13sq * xmat);
        const double lambda3 = dmsq31 + 0.5 * Dmsqee * (xmat_minus_1 + sqrt_term);
        
        // Lambda calculations (matching Python exactly)
        const double A_minus_lambda3 = A - lambda3;
        const double sqrt_term_lambda = std::sqrt(A_minus_lambda3 * A_minus_lambda3 - 4.0 * C / lambda3);
        const double Dlambda21 = sqrt_term_lambda;
        const double lambda2 = 0.5 * (A - lambda3 + Dlambda21);
        const double Dlambda32 = lambda3 - lambda2;
        const double Dlambda31 = Dlambda32 + Dlambda21;
        
        // Rosetta calculations (matching Python exactly)
        const double PiDlambdaInv = 1.0 / (Dlambda31 * Dlambda32 * Dlambda21);
        const double Xp3 = PiDlambdaInv * Dlambda21;
        const double Xp2 = -PiDlambdaInv * Dlambda31;
        
        // U matrix elements (matching Python exactly)
        const double Ue3sq_final = (lambda3 * (lambda3 - See) + Tee) * Xp3;
        const double Ue2sq_final = (lambda2 * (lambda2 - See) + Tee) * Xp2;
        
        const double Smm = A - dmsq21 * Um2sq_matter - dmsq31 * Um3sq;
        const double See_plus_Smm_minus_A = See + Smm - A;
        const double Tmm_final = Tmm * (1.0 - Um3sq - Um2sq_matter) + Amatter * See_plus_Smm_minus_A;
        
        const double Um3sq_final = (lambda3 * (lambda3 - Smm) + Tmm_final) * Xp3;
        const double Um2sq_final = (lambda2 * (lambda2 - Smm) + Tmm_final) * Xp2;
        
        const double Jmatter_final = (Jmatter * dmsq21 * dmsq31 * (dmsq31 - dmsq21) * PiDlambdaInv);
        
        // Calculate all U matrix elements (matching Python exactly)
        const double Ue1sq = 1.0 - Ue3sq_final - Ue2sq_final;
        const double Um1sq = 1.0 - Um3sq_final - Um2sq_final;
        const double Ut3sq = 1.0 - Um3sq_final - Ue3sq_final;
        const double Ut2sq_final = 1.0 - Um2sq_final - Ue2sq_final;
        const double Ut1sq = 1.0 - Um1sq - Ue1sq;
        
        // Kinematic terms (matching Python exactly)
        const double D21 = Dlambda21 * Lover4E;
        const double D32 = Dlambda32 * Lover4E;
        const double D31 = D32 + D21;
        
        // Trigonometric calculations (matching Python exactly)
        const double sinD21 = std::sin(D21);
        const double sinD31 = std::sin(D31);
        const double sinD32 = std::sin(D32);
        const double triple_sin = sinD21 * sinD31 * sinD32;
        
        const double sinsqD21_2 = 2.0 * sinD21 * sinD21;
        const double sinsqD31_2 = 2.0 * sinD31 * sinD31;
        const double sinsqD32_2 = 2.0 * sinD32 * sinD32;
        
        // Probability calculations (matching Python exactly)
        // pme_CPC calculation
        const double Ut_terms_0 = Ut3sq - Um2sq_final * Ue1sq - Um1sq * Ue2sq_final;
        const double Ut_terms_1 = Ut2sq_final - Um3sq_final * Ue1sq - Um1sq * Ue3sq_final;
        const double Ut_terms_2 = Ut1sq - Um3sq_final * Ue2sq_final - Um2sq_final * Ue3sq_final;
        
        const double pme_CPC = (Ut_terms_0 * sinsqD21_2 + 
                               Ut_terms_1 * sinsqD31_2 + 
                               Ut_terms_2 * sinsqD32_2);
        
        // pmm calculation
        const double Um_terms_0 = Um2sq_final * Um1sq;
        const double Um_terms_1 = Um3sq_final * Um1sq;
        const double Um_terms_2 = Um3sq_final * Um2sq_final;
        
        const double pmm_sum = (Um_terms_0 * sinsqD21_2 + 
                               Um_terms_1 * sinsqD31_2 + 
                               Um_terms_2 * sinsqD32_2);
        const double pmm = 1.0 - 2.0 * pmm_sum;
        
        // pee calculation
        const double Ue_terms_0 = Ue2sq_final * Ue1sq;
        const double Ue_terms_1 = Ue3sq_final * Ue1sq;
        const double Ue_terms_2 = Ue3sq_final * Ue2sq_final;
        
        const double pee_sum = (Ue_terms_0 * sinsqD21_2 + 
                               Ue_terms_1 * sinsqD31_2 + 
                               Ue_terms_2 * sinsqD32_2);
        const double pee = 1.0 - 2.0 * pee_sum;
        
        const double pme_CPV = -Jmatter_final * triple_sin;
        const double pem = pme_CPC - pme_CPV;
        const double pme = pme_CPC + pme_CPV;
        
        // Tau probabilities
        const double pet = 1.0 - pee - pem;
        const double pmt = 1.0 - pme - pmm;
        const double ptm = 1.0 - pem - pmm;
        const double pte = 1.0 - pee - pme;
        const double ptt = 1.0 - pet - pmt;
        
        // Probability assignment using absolute PDG codes (matching Python exactly)
        const int abs_osc_in = std::abs(osc_in_pdg);
        const int abs_osc_out = std::abs(osc_out_pdg);
        
        double result = 0.0;
        
        // Main oscillation channels (matching Python exactly)
        if (abs_osc_in == 12 && abs_osc_out == 14) result = pem;  // e->mu
        else if (abs_osc_in == 14 && abs_osc_out == 12) result = pme;  // mu->e
        else if (abs_osc_in == 14 && abs_osc_out == 14) result = pmm;  // mu->mu
        else if (abs_osc_in == 12 && abs_osc_out == 12) result = pee;  // e->e
        
        // Tau channels (matching Python exactly)
        else if (abs_osc_in == 12 && abs_osc_out == 16) result = pet;  // e->tau
        else if (abs_osc_in == 14 && abs_osc_out == 16) result = pmt;  // mu->tau
        else if (abs_osc_in == 16 && abs_osc_out == 14) result = ptm;  // tau->mu
        else if (abs_osc_in == 16 && abs_osc_out == 12) result = pte;  // tau->e
        else if (abs_osc_in == 16 && abs_osc_out == 16) result = ptt;  // tau->tau
        
        results[i] = result;
    }
}

Eigen::VectorXd Oscillator::calc_probability(const Eigen::VectorXd& osc_params) {
    if (!setup_) {
        throw MagpyProbabilityException("Oscillator not set up. Call set_energy_osc first.");
    }
    
    Eigen::VectorXd results(energies_.size());
    vectorized_oscillation(osc_params, results);
    return results;
}

} // namespace magpy

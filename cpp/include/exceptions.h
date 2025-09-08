#pragma once

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

namespace magpy {

/**
 * Exception for probability calculation errors
 */
class MagpyProbabilityException : public std::runtime_error {
public:
    MagpyProbabilityException(const std::string& message) 
        : std::runtime_error("MagpyProbabilityException: " + message) {}
};

/**
 * Exception for invalid object errors  
 */
class MagpyInvalidObjectError : public std::runtime_error {
public:
    MagpyInvalidObjectError(const std::string& message)
        : std::runtime_error("MagpyInvalidObjectError: " + message) {}
};

/**
 * Exception for spline errors
 */
class MagpySplineException : public std::runtime_error {
public:
    MagpySplineException(const std::string& message)
        : std::runtime_error("MagpySplineException: " + message) {}
};

} // namespace magpy

#pragma once

// A non-negative density sampled on a regular grid.
// Used by AHT and FFT as both source and target, by gridOT as the target,
// and by SemiOT as the source.

#include "domain.hpp"

#include <Eigen/Dense>

namespace ot {

struct GridDensity {
    Domain domain{};
    Eigen::MatrixXd values;

    GridDensity() = default;

    explicit GridDensity(Eigen::MatrixXd density, Domain box = {})
        : domain(std::move(box)), values(std::move(density)) {}

    // Uniform density on a rows x cols grid (the implicit source of gridOT).
    static GridDensity uniform(int rows, int cols, Domain box = {}) {
        return GridDensity(Eigen::MatrixXd::Ones(rows, cols), std::move(box));
    }

    double total_mass() const { return values.sum(); }
};

}  // namespace ot

#pragma once

// Discrete atoms: positions plus masses. Used as the target of SemiOT.

#include "domain.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <utility>

namespace ot {

struct WeightedPoints {
    Domain domain{};
    Eigen::MatrixX2d points;
    Eigen::VectorXd masses;

    WeightedPoints() = default;

    WeightedPoints(Eigen::MatrixX2d positions, Eigen::VectorXd weights,
                   Domain box = {})
        : domain(std::move(box)),
          points(std::move(positions)),
          masses(std::move(weights)) {
        if (points.rows() != masses.rows()) {
            throw std::invalid_argument(
                "WeightedPoints: positions and masses size mismatch");
        }
    }

    double total_mass() const { return masses.sum(); }
};

}  // namespace ot

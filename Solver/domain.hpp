#pragma once

// Axis-aligned rectangular domain that anchors both grids and point sets in a
// shared coordinate system. Defaults to the unit square [0,1] x [0,1].

#include <Eigen/Dense>

namespace ot {

struct Domain {
    Eigen::Vector2d min{0.0, 0.0};
    Eigen::Vector2d max{1.0, 1.0};

    Eigen::Vector2d extent() const { return max - min; }

    // Map normalized coordinates in [0,1]^2 to world coordinates.
    Eigen::Vector2d from_unit(const Eigen::Vector2d& u) const {
        return min + u.cwiseProduct(extent());
    }
};

}  // namespace ot

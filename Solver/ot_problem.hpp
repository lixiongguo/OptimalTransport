#pragma once

// Common input/output interface shared by the optimal-transport solvers:
//
//   * Solver/AHT_OT    (namespace aht_ot)   grid density  -> grid density
//   * Solver/FFTOT     (namespace fftot)    grid density  -> grid density
//   * Solver/Sinkhorn  (namespace sinkhorn) grid density  -> grid density
//   * Solver/gridOT    (namespace otmap)    uniform grid  -> grid density
//   * Solver/SemiOT    (namespace MA)       grid density  -> weighted points
//
// The two distribution kinds are independent types:
//
//   * ot::GridDensity     Solver/grid_density.hpp
//   * ot::WeightedPoints  Solver/weighted_points.hpp
//
// A problem stores one of each side as a variant, so every solver still
// implements the same call:
//
//   input  : ot::Problem { source, target }
//   output : ot::Result  { plan, cost }
//
// The header only depends on Eigen so it stays light; each solver provides a
// thin adapter (subclass of ot::Solver) in its own directory that converts to
// and from its native types.

#include "domain.hpp"
#include "grid_density.hpp"
#include "weighted_points.hpp"

#include <Eigen/Dense>

#include <memory>
#include <stdexcept>
#include <variant>

namespace ot {

// Either distribution. The concrete payload lives in GridDensity or
// WeightedPoints; this alias only lets one Problem hold both sides.
using Distribution = std::variant<GridDensity, WeightedPoints>;

inline bool is_grid(const Distribution& distribution) {
    return std::holds_alternative<GridDensity>(distribution);
}

inline bool is_points(const Distribution& distribution) {
    return std::holds_alternative<WeightedPoints>(distribution);
}

inline const GridDensity& as_grid(const Distribution& distribution) {
    if (const auto* grid = std::get_if<GridDensity>(&distribution)) return *grid;
    throw std::invalid_argument("distribution is not a grid density");
}

inline const WeightedPoints& as_points(const Distribution& distribution) {
    if (const auto* points = std::get_if<WeightedPoints>(&distribution)) {
        return *points;
    }
    throw std::invalid_argument("distribution is not weighted points");
}

// Two distributions: this is the common input of every solver.
struct Problem {
    Distribution source;  // mu0
    Distribution target;  // mu1
};

// The common output: a transport plan that maps the source domain onto the
// target domain. `forward` pushes a source point to where its mass lands in the
// target; `inverse` pulls a target point back. Solvers that only produce one
// direction may leave the other unimplemented (throwing).
class TransportPlan {
public:
    virtual ~TransportPlan() = default;

    virtual Eigen::Vector2d forward(const Eigen::Vector2d& x) const = 0;

    virtual Eigen::Vector2d inverse(const Eigen::Vector2d& /*y*/) const {
        throw std::logic_error("TransportPlan::inverse not available");
    }
};

// Result returned by every solver: the plan plus a few common diagnostics.
struct Result {
    std::shared_ptr<TransportPlan> plan;
    double transport_cost = 0.0;   // total (squared) Wasserstein-2 cost, if known
    int iterations = 0;
    bool converged = false;
};

// Abstract solver: takes two distributions, returns a transport plan.
// Each backend (aht_ot / otmap / MA) supplies a subclass adapter.
class Solver {
public:
    virtual ~Solver() = default;
    virtual Result solve(const Problem& problem) = 0;
};

}  // namespace ot

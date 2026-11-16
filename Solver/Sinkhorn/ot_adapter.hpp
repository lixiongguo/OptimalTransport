#pragma once

// Adapter exposing convolutional Sinkhorn through ot::Solver.
// Source and target are grid densities on the same domain and the same grid.

#include "../ot_problem.hpp"
#include "sinkhorn.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sinkhorn {

class SinkhornTransportPlan : public ot::TransportPlan {
public:
    SinkhornTransportPlan(Eigen::MatrixXd map_x, Eigen::MatrixXd map_y, ot::Domain domain)
        : map_x_(std::move(map_x)), map_y_(std::move(map_y)), domain_(domain) {
        rows_ = static_cast<int>(map_x_.rows());
        cols_ = static_cast<int>(map_x_.cols());
        if (rows_ <= 0 || cols_ <= 0 || map_y_.rows() != rows_ || map_y_.cols() != cols_) {
            throw std::invalid_argument("SinkhornTransportPlan: invalid map grid");
        }
    }

    Eigen::Vector2d forward(const Eigen::Vector2d& x) const override {
        return Eigen::Vector2d(sample(map_x_, x), sample(map_y_, x));
    }

private:
    double sample(const Eigen::MatrixXd& grid, const Eigen::Vector2d& x) const {
        const Eigen::Vector2d extent = domain_.extent();
        const double ux = std::clamp((x.x() - domain_.min.x()) / extent.x(), 0.0, 1.0);
        const double uy = std::clamp((x.y() - domain_.min.y()) / extent.y(), 0.0, 1.0);
        const double fc = std::clamp(ux * cols_ - 0.5, 0.0, static_cast<double>(cols_ - 1));
        const double fr = std::clamp(uy * rows_ - 0.5, 0.0, static_cast<double>(rows_ - 1));
        const int r0 = static_cast<int>(std::floor(fr));
        const int c0 = static_cast<int>(std::floor(fc));
        const int r1 = std::min(r0 + 1, rows_ - 1);
        const int c1 = std::min(c0 + 1, cols_ - 1);
        const double tr = fr - r0;
        const double tc = fc - c0;
        const double a = grid(r0, c0) * (1.0 - tc) + grid(r0, c1) * tc;
        const double b = grid(r1, c0) * (1.0 - tc) + grid(r1, c1) * tc;
        return a * (1.0 - tr) + b * tr;
    }

    Eigen::MatrixXd map_x_;
    Eigen::MatrixXd map_y_;
    ot::Domain domain_;
    int rows_ = 0;
    int cols_ = 0;
};

class SinkhornSolverAdapter : public ot::Solver {
public:
    explicit SinkhornSolverAdapter(SinkhornOptions options = {}) : solver_(options) {}

    ot::Result solve(const ot::Problem& problem) override {
        if (!ot::is_grid(problem.source) || !ot::is_grid(problem.target)) {
            throw std::invalid_argument(
                "SinkhornSolver requires grid source and target densities");
        }
        const ot::GridDensity& source = ot::as_grid(problem.source);
        const ot::GridDensity& target = ot::as_grid(problem.target);
        if ((source.domain.min - target.domain.min).norm() > 1e-12 ||
            (source.domain.max - target.domain.max).norm() > 1e-12) {
            throw std::invalid_argument(
                "SinkhornSolver requires source and target on the same domain");
        }

        ot::Domain domain = source.domain;
        const SinkhornResult native =
            solver_.solve(source.values, target.values, domain);

        ot::Result out;
        out.iterations = native.iterations;
        out.converged = native.converged;
        out.transport_cost = native.transport_cost;
        out.plan = std::make_shared<SinkhornTransportPlan>(
            native.map_x, native.map_y, domain);
        return out;
    }

private:
    SinkhornSolver solver_;
};

}  // namespace sinkhorn

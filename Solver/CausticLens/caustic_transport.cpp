#include "CausticLens/caustic_lens.hpp"
#include "SemiOT/ot_adapter.hpp"
#include "otsolver_2dgrid.h"
#include "transport_map.h"
#include "utils/eigen_addons.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace caustic {
namespace {

int semi_site_count(int resolution) {
    return std::clamp(resolution * 2, 200, 500);
}

void sample_sites(const Eigen::MatrixXd& target, int count, Eigen::MatrixX2d& points,
                  Eigen::VectorXd& masses) {
    const int rows = static_cast<int>(target.rows());
    const int cols = static_cast<int>(target.cols());
    // Distinct pixels, drawn with probability proportional to brightness.
    // Coincident sites make a Laguerre cell empty and the Newton solve stops.
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> unit(std::numeric_limits<double>::min(), 1.0);
    std::vector<std::pair<double, int>> keys;
    keys.reserve(static_cast<size_t>(rows * cols));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double value = std::max(target(row, col), 0.0);
            if (value <= 0.0) continue;
            keys.emplace_back(std::log(unit(rng)) / value, row * cols + col);
        }
    }
    if (keys.empty()) {
        throw std::invalid_argument("semi-discrete caustic target has no mass");
    }
    count = std::min(count, static_cast<int>(keys.size()));
    std::nth_element(keys.begin(), keys.begin() + count, keys.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    points.resize(count, 2);
    masses = Eigen::VectorXd::Ones(count);
    for (int site = 0; site < count; ++site) {
        const int index = keys[static_cast<size_t>(site)].second;
        const int row = index / cols;
        const int col = index - row * cols;
        points(site, 0) = (static_cast<double>(col) + 0.5) / cols;
        points(site, 1) = (static_cast<double>(row) + 0.5) / rows;
    }
}

void grid_transport(const Eigen::MatrixXd& target, double max_density_ratio,
                    Eigen::MatrixXd& map_x, Eigen::MatrixXd& map_y, int& iterations,
                    bool& converged) {
    const int n = static_cast<int>(target.rows());
    otmap::GridBasedTransportSolver solver;
    solver.set_verbose_level(0);
    solver.init(n);
    otmap::SolverOptions options;
    options.max_iter = 40;
    options.threshold = 1e-5;
    options.max_ratio = max_density_ratio;
    Eigen::MatrixXd density = target;
    const otmap::TransportMap map = solver.solve(Eigen::vec(density), options);

    std::vector<Eigen::Vector2d> points;
    points.reserve(static_cast<size_t>(n * n));
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            points.emplace_back((col + 0.5) / n, (row + 0.5) / n);
        }
    }
    otmap::apply_inverse_map(map, points, 0);

    map_x.resize(n, n);
    map_y.resize(n, n);
    int finite = 0;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const Eigen::Vector2d& point = points[static_cast<size_t>(row * n + col)];
            if (!std::isfinite(point.x()) || !std::isfinite(point.y())) {
                map_x(row, col) = (col + 0.5) / n;
                map_y(row, col) = (row + 0.5) / n;
                continue;
            }
            map_x(row, col) = std::clamp(point.x(), 0.0, 1.0);
            map_y(row, col) = std::clamp(point.y(), 0.0, 1.0);
            ++finite;
        }
    }
    iterations = options.max_iter;
    converged = finite == n * n;
}

void semi_transport(const Eigen::MatrixXd& target, Eigen::MatrixXd& map_x,
                    Eigen::MatrixXd& map_y, int& iterations, bool& converged) {
    const int n = static_cast<int>(target.rows());
    const int count = semi_site_count(n);
    Eigen::MatrixX2d sites;
    Eigen::VectorXd masses;
    sample_sites(target, count, sites, masses);

    ot::Problem problem;
    // The power diagram is defined on the whole square, so the source grid only
    // has to be fine enough to measure cell masses. The lens itself stays at n.
    const int source_resolution = std::min(n, 64);
    problem.source = ot::GridDensity::uniform(source_resolution, source_resolution);
    problem.target = ot::WeightedPoints(std::move(sites), std::move(masses));
    MA::SemiOTSolver solver(/*eps_g=*/1e-5, /*max_iter=*/30, /*verbose=*/false);
    const ot::Result solved = solver.solve(problem);

    map_x.resize(n, n);
    map_y.resize(n, n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const Eigen::Vector2d hit = solved.plan->forward(
                Eigen::Vector2d((col + 0.5) / n, (row + 0.5) / n));
            map_x(row, col) = std::clamp(hit.x(), 0.0, 1.0);
            map_y(row, col) = std::clamp(hit.y(), 0.0, 1.0);
        }
    }
    if (solved.iterations <= 0) {
        throw std::runtime_error("半离散传输没有得到有效的 power 图");
    }
    iterations = solved.iterations;
    converged = solved.converged;
}

}  // namespace

void compute_transport(const Eigen::MatrixXd& target, Transport transport,
                       double max_density_ratio, Eigen::MatrixXd& map_x,
                       Eigen::MatrixXd& map_y, int& iterations, bool& converged) {
    if (target.rows() != target.cols() || target.rows() < 2) {
        throw std::invalid_argument("caustic transport expects a square density");
    }
    if (transport == Transport::SemiDiscrete) {
        semi_transport(target, map_x, map_y, iterations, converged);
        return;
    }
    grid_transport(target, max_density_ratio, map_x, map_y, iterations, converged);
}

}  // namespace caustic

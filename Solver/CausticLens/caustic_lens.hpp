#pragma once

// Caustic lens design, following fast_caustic_design and Caustic-Design:
//
//   1. Optimal transport moves uniform incoming light onto the target image.
//      Grid transport is OTMap. Semi-discrete transport is the power diagram.
//   2. Inverse Snell gives the surface slope that steers each ray there,
//      and a least-squares integration turns those slopes into a height field.
//
// Light travels in -z. The lens sits near z = 0 over the square [-w/2, w/2]^2
// and the image plane is z = -focal_length. Refractive index is the lens,
// the outside medium is air.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace caustic {

// Grid is the OTMap solver from fast_caustic_design (uniform square onto the
// image). SemiDiscrete is the power-diagram solver from Caustic-Design.
enum class Transport { Grid, SemiDiscrete };

struct Options {
    int resolution = 64;
    double epsilon = 0.02;
    int transport_iterations = 40;
    double focal_length = 1.5;
    double width = 1.0;
    double refractive_index = 1.55;
    int outer_iterations = 4;
    double density_floor = 0.02;  // fraction of the peak kept in dark regions
    double max_density_ratio = 40.0;
    Transport transport = Transport::Grid;
};

struct Result {
    Eigen::MatrixXd target;
    Eigen::MatrixXd height;
    Eigen::MatrixXd caustic;
    Eigen::MatrixXd map_x;
    Eigen::MatrixXd map_y;
    int transport_iterations = 0;
    bool transport_converged = false;
    const char* method = "网格 OT";
    double relief = 0.0;
    double caustic_x = 0.5;
    double caustic_y = 0.5;
};

namespace detail {

inline Eigen::MatrixXd prepare_density(Eigen::MatrixXd image, int resolution, double floor_ratio) {
    if (image.rows() < 2 || image.cols() < 2) {
        throw std::invalid_argument("caustic target image is too small");
    }
    const int side = std::min(static_cast<int>(image.rows()), static_cast<int>(image.cols()));
    const int row0 = (static_cast<int>(image.rows()) - side) / 2;
    const int col0 = (static_cast<int>(image.cols()) - side) / 2;
    Eigen::MatrixXd square = image.block(row0, col0, side, side);
    Eigen::MatrixXd out(resolution, resolution);
    for (int row = 0; row < resolution; ++row) {
        const double src_row =
            (static_cast<double>(row) + 0.5) * side / resolution - 0.5;
        for (int col = 0; col < resolution; ++col) {
            const double src_col =
                (static_cast<double>(col) + 0.5) * side / resolution - 0.5;
            const double r = std::clamp(src_row, 0.0, static_cast<double>(side - 1));
            const double c = std::clamp(src_col, 0.0, static_cast<double>(side - 1));
            const int r0 = static_cast<int>(std::floor(r));
            const int c0 = static_cast<int>(std::floor(c));
            const int r1 = std::min(r0 + 1, side - 1);
            const int c1 = std::min(c0 + 1, side - 1);
            const double tr = r - r0;
            const double tc = c - c0;
            const double a = square(r0, c0) * (1.0 - tc) + square(r0, c1) * tc;
            const double b = square(r1, c0) * (1.0 - tc) + square(r1, c1) * tc;
            out(row, col) = a * (1.0 - tr) + b * tr;
        }
    }
    const double peak = std::max(out.maxCoeff(), 1e-8);
    out = out.cwiseMax(floor_ratio * peak);
    return out;
}

// Recover h from dh/dx and dh/dy. A pure Poisson solve drops constant slopes,
// because their Laplacian is zero, so each grid edge is matched directly.
inline Eigen::MatrixXd integrate_slopes(const Eigen::MatrixXd& gx, const Eigen::MatrixXd& gy,
                                        double dx, double dy) {
    const int rows = static_cast<int>(gx.rows());
    const int cols = static_cast<int>(gx.cols());
    if (gy.rows() != rows || gy.cols() != cols || rows < 2 || cols < 2) {
        throw std::invalid_argument("integrate_slopes: gradient grids do not match");
    }
    const int unknowns = rows * cols;
    const int equations = rows * (cols - 1) + cols * (rows - 1) + 1;
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<size_t>(equations) * 2);
    Eigen::VectorXd rhs(equations);
    int row_index = 0;

    auto add = [&](int equation, int variable, double weight) {
        entries.emplace_back(equation, variable, weight);
    };

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols - 1; ++col) {
            const double slope = 0.5 * (gx(row, col) + gx(row, col + 1));
            add(row_index, row * cols + col, -1.0 / dx);
            add(row_index, row * cols + col + 1, 1.0 / dx);
            rhs(row_index) = slope;
            ++row_index;
        }
    }
    for (int row = 0; row < rows - 1; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double slope = 0.5 * (gy(row, col) + gy(row + 1, col));
            add(row_index, row * cols + col, -1.0 / dy);
            add(row_index, (row + 1) * cols + col, 1.0 / dy);
            rhs(row_index) = slope;
            ++row_index;
        }
    }
    add(row_index, 0, 1.0);
    rhs(row_index) = 0.0;

    Eigen::SparseMatrix<double> system(equations, unknowns);
    system.setFromTriplets(entries.begin(), entries.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(system.transpose() * system);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("caustic height integration failed");
    }
    const Eigen::VectorXd solution = solver.solve(system.transpose() * rhs);
    Eigen::MatrixXd height(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            height(row, col) = solution(row * cols + col);
        }
    }
    height.array() -= height.minCoeff();
    return height;
}

inline void slopes_from_snell(const Eigen::MatrixXd& height, const Eigen::MatrixXd& map_x,
                              const Eigen::MatrixXd& map_y, double width, double focal,
                              double refractive_index, Eigen::MatrixXd& gx,
                              Eigen::MatrixXd& gy) {
    const int rows = static_cast<int>(height.rows());
    const int cols = static_cast<int>(height.cols());
    const double dx = width / static_cast<double>(cols);
    const double dy = width / static_cast<double>(rows);
    gx.resize(rows, cols);
    gy.resize(rows, cols);
    const Eigen::Vector3d incident(0.0, 0.0, -1.0);
    for (int row = 0; row < rows; ++row) {
        const double y = -0.5 * width + (static_cast<double>(row) + 0.5) * dy;
        for (int col = 0; col < cols; ++col) {
            const double x = -0.5 * width + (static_cast<double>(col) + 0.5) * dx;
            const double target_x = -0.5 * width + map_x(row, col) * width;
            const double target_y = -0.5 * width + map_y(row, col) * width;
            Eigen::Vector3d transmitted(target_x - x, target_y - y,
                                        -focal - height(row, col));
            const double length = transmitted.norm();
            if (length < 1e-8) {
                gx(row, col) = 0.0;
                gy(row, col) = 0.0;
                continue;
            }
            transmitted /= length;
            // N_down ∝ n_lens * I - T, the same combination as fresnelMapping.
            // The graph z = h has its upward normal (-hx, -hy, 1).
            Eigen::Vector3d downward = refractive_index * incident - transmitted;
            Eigen::Vector3d upward = -downward;
            if (upward.z() < 1e-4) upward = Eigen::Vector3d(0.0, 0.0, 1.0);
            gx(row, col) = -upward.x() / upward.z();
            gy(row, col) = -upward.y() / upward.z();
        }
    }
}

inline Eigen::Vector3d upward_normal(const Eigen::MatrixXd& height, int row, int col,
                                     double dx, double dy) {
    const int rows = static_cast<int>(height.rows());
    const int cols = static_cast<int>(height.cols());
    const int c0 = std::max(col - 1, 0);
    const int c1 = std::min(col + 1, cols - 1);
    const int r0 = std::max(row - 1, 0);
    const int r1 = std::min(row + 1, rows - 1);
    const double hx = (height(row, c1) - height(row, c0)) / (dx * (c1 - c0));
    const double hy = (height(r1, col) - height(r0, col)) / (dy * (r1 - r0));
    Eigen::Vector3d normal(-hx, -hy, 1.0);
    return normal.normalized();
}

inline Eigen::Vector3d refract_down(const Eigen::Vector3d& normal, double refractive_index) {
    const Eigen::Vector3d incident(0.0, 0.0, -1.0);
    const double ratio = refractive_index;  // lens -> air
    const double cos_i = std::clamp(-normal.dot(incident), 0.0, 1.0);
    const double sin2_t = ratio * ratio * (1.0 - cos_i * cos_i);
    if (sin2_t >= 1.0) return Eigen::Vector3d::Zero();  // total internal reflection
    const double cos_t = std::sqrt(1.0 - sin2_t);
    return (ratio * incident + (ratio * cos_i - cos_t) * normal).normalized();
}

inline Eigen::MatrixXd simulate(const Eigen::MatrixXd& height, double width, double focal,
                                double refractive_index) {
    const int rows = static_cast<int>(height.rows());
    const int cols = static_cast<int>(height.cols());
    const double dx = width / static_cast<double>(cols);
    const double dy = width / static_cast<double>(rows);
    Eigen::MatrixXd image = Eigen::MatrixXd::Zero(rows, cols);
    for (int row = 0; row < rows; ++row) {
        const double y = -0.5 * width + (static_cast<double>(row) + 0.5) * dy;
        for (int col = 0; col < cols; ++col) {
            const double x = -0.5 * width + (static_cast<double>(col) + 0.5) * dx;
            const Eigen::Vector3d direction =
                refract_down(upward_normal(height, row, col, dx, dy), refractive_index);
            if (direction.z() >= -1e-6) continue;
            const double t = (-focal - height(row, col)) / direction.z();
            if (t <= 0.0) continue;
            const double hit_x = x + t * direction.x();
            const double hit_y = y + t * direction.y();
            const double u = (hit_x + 0.5 * width) / width * cols - 0.5;
            const double v = (hit_y + 0.5 * width) / width * rows - 0.5;
            const int c0 = static_cast<int>(std::floor(u));
            const int r0 = static_cast<int>(std::floor(v));
            // A one-pixel footprint leaves grid rings; spread each ray a little.
            constexpr int kRadius = 2;
            constexpr double kSigma2 = 0.85 * 0.85;
            for (int kr = -kRadius; kr <= kRadius; ++kr) {
                for (int kc = -kRadius; kc <= kRadius; ++kc) {
                    const int rr = r0 + kr;
                    const int cc = c0 + kc;
                    if (rr < 0 || cc < 0 || rr >= rows || cc >= cols) continue;
                    const double du = (static_cast<double>(cc) + 0.5) - (u + 0.5);
                    const double dv = (static_cast<double>(rr) + 0.5) - (v + 0.5);
                    image(rr, cc) += std::exp(-(du * du + dv * dv) / (2.0 * kSigma2));
                }
            }
        }
    }
    return image;
}

inline void center_of_mass(const Eigen::MatrixXd& image, double& x, double& y) {
    double mass = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    const int rows = static_cast<int>(image.rows());
    const int cols = static_cast<int>(image.cols());
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double value = std::max(image(row, col), 0.0);
            mass += value;
            sx += value * ((static_cast<double>(col) + 0.5) / cols);
            sy += value * ((static_cast<double>(row) + 0.5) / rows);
        }
    }
    if (mass <= 0.0) {
        x = 0.5;
        y = 0.5;
        return;
    }
    x = sx / mass;
    y = sy / mass;
}

}  // namespace detail

inline Eigen::MatrixXd spot_image(int resolution, double cx, double cy, double sigma) {
    Eigen::MatrixXd image(resolution, resolution);
    for (int row = 0; row < resolution; ++row) {
        for (int col = 0; col < resolution; ++col) {
            const double x = (static_cast<double>(col) + 0.5) / resolution;
            const double y = (static_cast<double>(row) + 0.5) / resolution;
            const double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            image(row, col) = std::exp(-d2 / (2.0 * sigma * sigma));
        }
    }
    return image;
}

inline Eigen::MatrixXd ring_image(int resolution) {
    Eigen::MatrixXd image(resolution, resolution);
    for (int row = 0; row < resolution; ++row) {
        for (int col = 0; col < resolution; ++col) {
            const double x = (static_cast<double>(col) + 0.5) / resolution - 0.5;
            const double y = (static_cast<double>(row) + 0.5) / resolution - 0.5;
            const double radius = std::sqrt(x * x + y * y);
            const double ring = std::exp(-(radius - 0.28) * (radius - 0.28) / (2.0 * 0.035 * 0.035));
            image(row, col) = 0.04 + ring;
        }
    }
    return image;
}

// Where each lens cell sends its light, in the unit square. Defined in
// caustic_transport.cpp: grid OTMap, or the semi-discrete power diagram.
void compute_transport(const Eigen::MatrixXd& target, Transport transport,
                       double max_density_ratio, Eigen::MatrixXd& map_x,
                       Eigen::MatrixXd& map_y, int& iterations, bool& converged);

inline Result design(const Eigen::MatrixXd& target_image, Options options = {}) {
    if (options.resolution < 8) {
        throw std::invalid_argument("caustic resolution must be at least 8");
    }
    if (!(options.focal_length > 0.0) || !(options.width > 0.0) ||
        !(options.refractive_index > 1.0)) {
        throw std::invalid_argument("caustic focal length, width, and index are invalid");
    }

    const int n = options.resolution;
    const Eigen::MatrixXd target =
        detail::prepare_density(target_image, n, options.density_floor);
    Eigen::MatrixXd map_x;
    Eigen::MatrixXd map_y;
    int iterations = 0;
    bool converged = false;
    compute_transport(target, options.transport, options.max_density_ratio, map_x, map_y,
                      iterations, converged);

    Eigen::MatrixXd height = Eigen::MatrixXd::Zero(n, n);
    const double dx = options.width / static_cast<double>(n);
    const double dy = dx;
    for (int iteration = 0; iteration < options.outer_iterations; ++iteration) {
        Eigen::MatrixXd gx;
        Eigen::MatrixXd gy;
        detail::slopes_from_snell(height, map_x, map_y, options.width, options.focal_length,
                                  options.refractive_index, gx, gy);
        Eigen::MatrixXd updated = detail::integrate_slopes(gx, gy, dx, dy);
        if (iteration == 0) height = updated;
        else height = 0.45 * height + 0.55 * updated;
        height.array() -= height.minCoeff();
    }

    Result result;
    result.target = target;
    result.height = height;
    result.map_x = map_x;
    result.map_y = map_y;
    result.caustic = detail::simulate(height, options.width, options.focal_length,
                                      options.refractive_index);
    result.transport_iterations = iterations;
    result.transport_converged = converged;
    result.method = options.transport == Transport::SemiDiscrete ? "半离散" : "网格 OT";
    result.relief = height.maxCoeff();
    detail::center_of_mass(result.caustic, result.caustic_x, result.caustic_y);
    return result;
}

// A constant x-slope must come back out of the integrator. Returns false on failure.
inline bool integration_self_test() {
    const int n = 24;
    Eigen::MatrixXd gx = Eigen::MatrixXd::Constant(n, n, 0.2);
    Eigen::MatrixXd gy = Eigen::MatrixXd::Zero(n, n);
    const double dx = 1.0 / n;
    const Eigen::MatrixXd height = detail::integrate_slopes(gx, gy, dx, dx);
    const double rise = height(n / 2, n - 1) - height(n / 2, 0);
    const double expected = 0.2 * (1.0 - dx);
    return std::abs(rise - expected) < 0.03 * std::max(expected, 1e-6);
}

}  // namespace caustic

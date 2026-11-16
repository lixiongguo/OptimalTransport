#pragma once

// Entropic optimal transport on a pair of grid densities (Cuturi 2013),
// evaluated with a separable Gaussian convolution (Solomon et al. 2015).
//
// Both densities live on the same rectangular grid. The cost is squared
// Euclidean distance, so the Gibbs kernel exp(-||x-y||^2 / ε) factors into
// independent row and column blurs. The barycentric projection of the
// resulting coupling is the transport map.

#include "../domain.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sinkhorn {

struct SinkhornOptions {
    double epsilon = 0.02;        // regularization, in domain-length squared
    int max_iterations = 50;
    double tolerance = 1e-3;      // L1 gap between the source marginal and a
    double min_density = 1e-8;
    int max_kernel_radius = 48;
};

struct SinkhornResult {
    Eigen::MatrixXd map_x;  // barycentric x at cell centers, domain coordinates
    Eigen::MatrixXd map_y;
    int iterations = 0;
    bool converged = false;
    double marginal_error = std::numeric_limits<double>::infinity();
    double transport_cost = 0.0;
};

namespace detail {

inline Eigen::MatrixXd probability(const Eigen::MatrixXd& density, double floor_value) {
    if (density.size() == 0 || !density.allFinite()) {
        throw std::invalid_argument("SinkhornSolver: density is empty or non-finite");
    }
    Eigen::MatrixXd out = density.cwiseMax(floor_value);
    const double mass = out.sum();
    if (!(mass > 0.0)) {
        throw std::invalid_argument("SinkhornSolver: density has zero mass");
    }
    out /= mass;
    return out;
}

inline std::vector<double> gaussian_kernel(double epsilon_pixels, int max_radius, int& radius) {
    if (!(epsilon_pixels > 0.0)) {
        throw std::invalid_argument("SinkhornSolver: epsilon must be positive");
    }
    const double sigma = std::sqrt(epsilon_pixels / 2.0);
    radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    radius = std::min(radius, std::max(1, max_radius));

    std::vector<double> weight(static_cast<size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double value = std::exp(-(static_cast<double>(offset) * offset) / epsilon_pixels);
        weight[static_cast<size_t>(offset + radius)] = value;
        sum += value;
    }
    for (double& value : weight) value /= sum;
    return weight;
}

inline void convolve_horizontal(const Eigen::MatrixXd& in, Eigen::MatrixXd& out,
                                const std::vector<double>& kernel, int radius) {
    const int rows = static_cast<int>(in.rows());
    const int cols = static_cast<int>(in.cols());
    out.resize(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int sample = std::clamp(col + offset, 0, cols - 1);
                sum += kernel[static_cast<size_t>(offset + radius)] * in(row, sample);
            }
            out(row, col) = sum;
        }
    }
}

inline void convolve_vertical(const Eigen::MatrixXd& in, Eigen::MatrixXd& out,
                              const std::vector<double>& kernel, int radius) {
    const int rows = static_cast<int>(in.rows());
    const int cols = static_cast<int>(in.cols());
    out.resize(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int sample = std::clamp(row + offset, 0, rows - 1);
                sum += kernel[static_cast<size_t>(offset + radius)] * in(sample, col);
            }
            out(row, col) = sum;
        }
    }
}

class SeparableBlur {
public:
    SeparableBlur(int rows, int cols, double epsilon, const ot::Domain& domain,
                  int max_radius) {
        const double hx = domain.extent().x() / static_cast<double>(cols);
        const double hy = domain.extent().y() / static_cast<double>(rows);
        kernel_x_ = gaussian_kernel(epsilon / (hx * hx), max_radius, radius_x_);
        kernel_y_ = gaussian_kernel(epsilon / (hy * hy), max_radius, radius_y_);
    }

    Eigen::MatrixXd apply(const Eigen::MatrixXd& image) const {
        Eigen::MatrixXd horizontal;
        Eigen::MatrixXd blurred;
        convolve_horizontal(image, horizontal, kernel_x_, radius_x_);
        convolve_vertical(horizontal, blurred, kernel_y_, radius_y_);
        return blurred;
    }

private:
    std::vector<double> kernel_x_;
    std::vector<double> kernel_y_;
    int radius_x_ = 1;
    int radius_y_ = 1;
};

}  // namespace detail

class SinkhornSolver {
public:
    explicit SinkhornSolver(SinkhornOptions options = {}) : options_(options) {}

    const SinkhornOptions& options() const { return options_; }

    SinkhornResult solve(const Eigen::MatrixXd& source_density,
                         const Eigen::MatrixXd& target_density,
                         ot::Domain domain = {}) const {
        if (source_density.rows() != target_density.rows() ||
            source_density.cols() != target_density.cols()) {
            throw std::invalid_argument(
                "SinkhornSolver requires source and target grids of the same size");
        }
        if (source_density.rows() < 2 || source_density.cols() < 2) {
            throw std::invalid_argument("SinkhornSolver requires at least a 2x2 grid");
        }
        if (domain.extent().x() <= 0.0 || domain.extent().y() <= 0.0) {
            throw std::invalid_argument("SinkhornSolver requires a positive domain");
        }
        if (!(options_.epsilon > 0.0)) {
            throw std::invalid_argument("SinkhornSolver requires positive epsilon");
        }

        const int rows = static_cast<int>(source_density.rows());
        const int cols = static_cast<int>(source_density.cols());
        const Eigen::MatrixXd source =
            detail::probability(source_density, options_.min_density);
        const Eigen::MatrixXd target =
            detail::probability(target_density, options_.min_density);

        const detail::SeparableBlur blur(rows, cols, options_.epsilon, domain,
                                          options_.max_kernel_radius);
        Eigen::MatrixXd scale_u = Eigen::MatrixXd::Ones(rows, cols);
        Eigen::MatrixXd scale_v = Eigen::MatrixXd::Ones(rows, cols);
        double marginal_error = std::numeric_limits<double>::infinity();
        int iterations = 0;
        bool converged = false;

        while (iterations < options_.max_iterations) {
            ++iterations;
            const Eigen::MatrixXd blurred_v = blur.apply(scale_v).cwiseMax(1e-300);
            scale_u = source.cwiseQuotient(blurred_v);
            const Eigen::MatrixXd blurred_u = blur.apply(scale_u).cwiseMax(1e-300);
            scale_v = target.cwiseQuotient(blurred_u);
            const double mean_u = scale_u.mean();
            if (mean_u > 0.0 && std::isfinite(mean_u)) {
                scale_u /= mean_u;
                scale_v *= mean_u;
            }
            marginal_error = (scale_u.cwiseProduct(blur.apply(scale_v)) - source).lpNorm<1>();
            if (marginal_error < options_.tolerance) {
                converged = true;
                break;
            }
        }

        Eigen::MatrixXd coord_x(rows, cols);
        Eigen::MatrixXd coord_y(rows, cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Eigen::Vector2d center = cell_center(row, col, rows, cols, domain);
                coord_x(row, col) = center.x();
                coord_y(row, col) = center.y();
            }
        }

        const Eigen::MatrixXd denom = blur.apply(scale_v).cwiseMax(1e-300);
        SinkhornResult result;
        result.map_x = blur.apply(scale_v.cwiseProduct(coord_x)).cwiseQuotient(denom);
        result.map_y = blur.apply(scale_v.cwiseProduct(coord_y)).cwiseQuotient(denom);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                result.map_x(row, col) =
                    std::clamp(result.map_x(row, col), domain.min.x(), domain.max.x());
                result.map_y(row, col) =
                    std::clamp(result.map_y(row, col), domain.min.y(), domain.max.y());
            }
        }
        result.iterations = iterations;
        result.converged = converged;
        result.marginal_error = marginal_error;
        result.transport_cost = barycentric_cost(source, coord_x, coord_y,
                                                  result.map_x, result.map_y);
        return result;
    }

private:
    static Eigen::Vector2d cell_center(int row, int col, int rows, int cols,
                                       const ot::Domain& domain) {
        const Eigen::Vector2d extent = domain.extent();
        return Eigen::Vector2d(
            domain.min.x() + (static_cast<double>(col) + 0.5) * extent.x() / cols,
            domain.min.y() + (static_cast<double>(row) + 0.5) * extent.y() / rows);
    }

    static double barycentric_cost(const Eigen::MatrixXd& source,
                                   const Eigen::MatrixXd& coord_x,
                                   const Eigen::MatrixXd& coord_y,
                                   const Eigen::MatrixXd& map_x,
                                   const Eigen::MatrixXd& map_y) {
        double cost = 0.0;
        for (int row = 0; row < source.rows(); ++row) {
            for (int col = 0; col < source.cols(); ++col) {
                const double dx = map_x(row, col) - coord_x(row, col);
                const double dy = map_y(row, col) - coord_y(row, col);
                cost += source(row, col) * (dx * dx + dy * dy);
            }
        }
        return cost;
    }

    SinkhornOptions options_;
};

}  // namespace sinkhorn

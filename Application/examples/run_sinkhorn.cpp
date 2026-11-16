// Grid density -> grid density through ot::Solver, backed by Sinkhorn.

#include "Sinkhorn/ot_adapter.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>

static Eigen::MatrixXd gaussian_grid(int rows, int cols, double cx, double cy,
                                     double sigma) {
    Eigen::MatrixXd grid(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double x = (col + 0.5) / cols;
            const double y = (row + 0.5) / rows;
            const double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            grid(row, col) = std::exp(-d2 / (2.0 * sigma * sigma)) + 1e-3;
        }
    }
    return grid;
}

int main() {
    const int rows = 48;
    const int cols = 48;

    ot::Problem problem;
    problem.source = ot::GridDensity(gaussian_grid(rows, cols, 0.32, 0.32, 0.12));
    problem.target = ot::GridDensity(gaussian_grid(rows, cols, 0.70, 0.68, 0.12));

    sinkhorn::SinkhornOptions options;
    options.epsilon = 0.02;
    options.max_iterations = 40;

    sinkhorn::SinkhornSolverAdapter solver(options);
    const ot::Result result = solver.solve(problem);

    const Eigen::Vector2d samples[] = {
        Eigen::Vector2d(0.32, 0.32),
        Eigen::Vector2d(0.50, 0.50),
        Eigen::Vector2d(0.70, 0.68),
    };
    for (const Eigen::Vector2d& sample : samples) {
        const Eigen::Vector2d image = result.plan->forward(sample);
        std::printf("T(%.2f, %.2f) -> (%.4f, %.4f)\n", sample.x(), sample.y(),
                    image.x(), image.y());
    }
    std::printf("iterations = %d ; converged = %d ; W2 cost ~= %.8f\n",
                result.iterations, static_cast<int>(result.converged),
                result.transport_cost);

    const Eigen::Vector2d moved = result.plan->forward(Eigen::Vector2d(0.32, 0.32));
    if (!(moved.x() > 0.50 && moved.y() > 0.50)) {
        std::fprintf(stderr, "Sinkhorn map did not move the source bump toward the target\n");
        return 1;
    }
    return 0;
}

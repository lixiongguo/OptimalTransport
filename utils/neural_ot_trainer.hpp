#pragma once

// 不依赖 PyTorch 的 Neural Lagrangian OT（OT-CFM 变体）最小实现。
//
//   速度场  v_theta(x, t) : 3 -> 32 -> 32 -> 2 的 MLP（SiLU 激活）
//   训练目标: 在 minibatch 上用 log-domain Sinkhorn 求一个熵正则最优耦合
//             (x0, x1)，然后让 v 拟合直线路径的常速度
//             x_t = (1 - t) x0 + t * x1,  u = x1 - x0
//             即 Benamou-Brenier 动能项的 flow-matching 形式。
//   优化器  : 手写反向传播 + Adam（带梯度裁剪）。
//   推理    : 对 dx/dt = v_theta(x, t) 做显式积分，得到连续传输映射 T(x) = X_1(x)。
//
// 全部量化，没有引入任何第三方库，只用 Eigen + 标准库。

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace neural_ot_demo {

inline double sigmoidOf(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double siluOf(double x) { return x * sigmoidOf(x); }
inline double siluGradOf(double x) {
    const double s = sigmoidOf(x);
    return s * (1.0 + x * (1.0 - s));
}

// 一层全连接 + Adam 状态。
struct Dense {
    Eigen::MatrixXd weight;
    Eigen::VectorXd bias;
    Eigen::MatrixXd gw;
    Eigen::VectorXd gb;
    Eigen::MatrixXd mw;
    Eigen::VectorXd mb;
    Eigen::MatrixXd vw;
    Eigen::VectorXd vb;
};

class Mlp {
public:
    void configure(int dimIn, int hidden, int hiddenLayers, int dimOut, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        layers_.clear();
        int last = dimIn;
        for (int k = 0; k < hiddenLayers; ++k) {
            layers_.push_back(makeLayer(last, hidden, uniform, rng));
            last = hidden;
        }
        layers_.push_back(makeLayer(last, dimOut, uniform, rng));
        step_ = 0;
    }

    // input: (in x n)，返回 (out x n)。
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) const {
        preActivation_.clear();
        activation_.clear();
        Eigen::MatrixXd a = input;
        activation_.push_back(a);
        for (size_t i = 0; i < layers_.size(); ++i) {
            Eigen::MatrixXd z = layers_[i].weight * a;
            z.colwise() += layers_[i].bias;
            preActivation_.push_back(z);
            const bool last = (i + 1 == layers_.size());
            a = last ? z : z.unaryExpr([](double v) { return siluOf(v); });
            activation_.push_back(a);
        }
        return a;
    }

    // 单次梯度下降，返回该 batch 的 MSE。
    double trainStep(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, double lr) {
        const Eigen::MatrixXd out = forward(input);
        const double n = static_cast<double>(std::max<Eigen::Index>(1, input.cols()));
        const double loss = (out - target).squaredNorm() / n;
        Eigen::MatrixXd d = (2.0 / n) * (out - target);

        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            const Eigen::MatrixXd& previous = activation_[static_cast<size_t>(i)];
            layers_[static_cast<size_t>(i)].gw = d * previous.transpose();
            layers_[static_cast<size_t>(i)].gb = d.rowwise().sum();
            if (i > 0) {
                const Eigen::MatrixXd dz = layers_[static_cast<size_t>(i)].weight.transpose() * d;
                const Eigen::MatrixXd& zPrevious = preActivation_[static_cast<size_t>(i - 1)];
                const Eigen::MatrixXd slope =
                    zPrevious.unaryExpr([](double v) { return siluGradOf(v); });
                d = (dz.array() * slope.array()).matrix();
            }
        }
        ++step_;
        adam(lr);
        return loss;
    }

    int step() const { return step_; }

private:
    static Dense makeLayer(int in, int out, std::uniform_real_distribution<double>& uniform,
                           std::mt19937& rng) {
        Dense layer;
        const double scale = std::sqrt(1.0 / static_cast<double>(in));
        layer.weight.resize(out, in);
        for (int r = 0; r < out; ++r) {
            for (int c = 0; c < in; ++c) layer.weight(r, c) = scale * uniform(rng);
        }
        layer.bias = Eigen::VectorXd::Zero(out);
        layer.gw = Eigen::MatrixXd::Zero(out, in);
        layer.gb = Eigen::VectorXd::Zero(out);
        layer.mw = Eigen::MatrixXd::Zero(out, in);
        layer.mb = Eigen::VectorXd::Zero(out);
        layer.vw = Eigen::MatrixXd::Zero(out, in);
        layer.vb = Eigen::VectorXd::Zero(out);
        return layer;
    }

    static void clipInPlace(Eigen::MatrixXd& g) { g = g.cwiseMax(-5.0).cwiseMin(5.0); }
    static void clipInPlace(Eigen::VectorXd& g) { g = g.cwiseMax(-5.0).cwiseMin(5.0); }

    void adam(double lr) {
        constexpr double beta1 = 0.9;
        constexpr double beta2 = 0.999;
        constexpr double eps = 1e-8;
        const double c1 = 1.0 - std::pow(beta1, static_cast<double>(step_));
        const double c2 = 1.0 - std::pow(beta2, static_cast<double>(step_));
        for (Dense& layer : layers_) {
            clipInPlace(layer.gw);
            clipInPlace(layer.gb);
            layer.mw = beta1 * layer.mw + (1.0 - beta1) * layer.gw;
            layer.vw = beta2 * layer.vw + (1.0 - beta2) * layer.gw.cwiseProduct(layer.gw);
            const Eigen::MatrixXd mHat = layer.mw / c1;
            const Eigen::MatrixXd vHat = layer.vw / c2;
            layer.weight.array() -= lr * mHat.array() / (vHat.array().sqrt() + eps);

            layer.mb = beta1 * layer.mb + (1.0 - beta1) * layer.gb;
            layer.vb = beta2 * layer.vb + (1.0 - beta2) * layer.gb.cwiseProduct(layer.gb);
            const Eigen::VectorXd mbHat = layer.mb / c1;
            const Eigen::VectorXd vbHat = layer.vb / c2;
            layer.bias.array() -= lr * mbHat.array() / (vbHat.array().sqrt() + eps);
        }
    }

    std::vector<Dense> layers_;
    mutable std::vector<Eigen::MatrixXd> preActivation_;
    mutable std::vector<Eigen::MatrixXd> activation_;
    int step_ = 0;
};

// 两个等权点云之间的 log-domain Sinkhorn 熵正则耦合。
inline Eigen::MatrixXd sinkhornPlan(const std::vector<Eigen::Vector2d>& a,
                                    const std::vector<Eigen::Vector2d>& b, double epsilon,
                                    int iterations) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    Eigen::MatrixXd logK(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            logK(i, j) = -(a[static_cast<size_t>(i)] - b[static_cast<size_t>(j)]).squaredNorm() /
                         epsilon;
        }
    }
    Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(m);
    const double logA = -std::log(static_cast<double>(n));
    const double logB = -std::log(static_cast<double>(m));
    for (int it = 0; it < iterations; ++it) {
        for (int i = 0; i < n; ++i) {
            double best = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < m; ++j) best = std::max(best, logK(i, j) + v(j));
            double sum = 0.0;
            for (int j = 0; j < m; ++j) sum += std::exp(logK(i, j) + v(j) - best);
            u(i) = logA - (best + std::log(std::max(sum, 1e-300)));
        }
        for (int j = 0; j < m; ++j) {
            double best = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < n; ++i) best = std::max(best, logK(i, j) + u(i));
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += std::exp(logK(i, j) + u(i) - best);
            v(j) = logB - (best + std::log(std::max(sum, 1e-300)));
        }
    }
    Eigen::MatrixXd plan(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) plan(i, j) = std::exp(logK(i, j) + u(i) + v(j));
    }
    return plan;
}

// 训练器：持有速度场、固定样本池与它们的最优耦合。
class Trainer {
public:
    void configure(int poolSize, int hidden, int hiddenLayers, unsigned seed) {
        pool_ = poolSize;
        seed_ = seed;
        rng_.seed(seed);
        mlp_.configure(3, hidden, hiddenLayers, 2, seed);
        couplingReady_ = false;
        lastLoss_ = 0.0;
        histLoss_.clear();
    }

    // 重建样本池并重算耦合（目标分布变了、或想换一批样本时调用）。
    void refreshCoupling(const std::vector<Eigen::Vector2d>& source,
                         const std::vector<Eigen::Vector2d>& target, double epsilon, int iterations) {
        source_ = source;
        target_ = target;
        plan_ = sinkhornPlan(source_, target_, epsilon, iterations);
        // 每行的归一化 CDF，便于按耦合抽样。
        cdf_.assign(source_.size(), std::vector<double>(target_.size(), 0.0));
        for (size_t i = 0; i < source_.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < target_.size(); ++j) sum += plan_(static_cast<int>(i), static_cast<int>(j));
            double acc = 0.0;
            for (size_t j = 0; j < target_.size(); ++j) {
                acc += plan_(static_cast<int>(i), static_cast<int>(j)) / std::max(sum, 1e-300);
                cdf_[i][j] = acc;
            }
            cdf_[i].back() = 1.0;
        }
        couplingReady_ = true;
    }

    bool ready() const { return couplingReady_; }

    // 跑 steps 次 Adam 迭代，返回最后一次的 MSE。
    double train(int steps, int batch, double lr) {
        if (!couplingReady_ || source_.empty() || target_.empty()) return 0.0;
        std::uniform_real_distribution<double> pick(0.0, 1.0);
        std::uniform_int_distribution<size_t> pickSource(0, source_.size() - 1);
        const int b = std::max(8, batch);
        Eigen::MatrixXd input(3, b);
        Eigen::MatrixXd targetVel(2, b);
        for (int step = 0; step < steps; ++step) {
            for (int k = 0; k < b; ++k) {
                const size_t i = pickSource(rng_);
                const double r = pick(rng_);
                size_t j = 0;
                while (j + 1 < cdf_[i].size() && cdf_[i][j] < r) ++j;
                const Eigen::Vector2d& x0 = source_[i];
                const Eigen::Vector2d& x1 = target_[j];
                const double t = pick(rng_);
                const Eigen::Vector2d xt = (1.0 - t) * x0 + t * x1;
                input(0, k) = xt.x();
                input(1, k) = xt.y();
                input(2, k) = t;
                targetVel(0, k) = x1.x() - x0.x();
                targetVel(1, k) = x1.y() - x0.y();
            }
            lastLoss_ = mlp_.trainStep(input, targetVel, lr);
        }
        histLoss_.push_back(lastLoss_);
        if (histLoss_.size() > 600) histLoss_.erase(histLoss_.begin());
        return lastLoss_;
    }

    Eigen::MatrixXd velocity(const Eigen::MatrixXd& points, double t) const {
        const Eigen::Index n = points.cols();
        Eigen::MatrixXd input(3, n);
        input.topRows(2) = points;
        input.row(2).setConstant(t);
        return mlp_.forward(input);
    }

    // 对一批点做显式积分，返回各时刻位置（含起点），用于画轨迹。
    std::vector<Eigen::MatrixXd> rollout(const Eigen::MatrixXd& start, int steps) const {
        steps = std::max(1, steps);
        const double dt = 1.0 / static_cast<double>(steps);
        std::vector<Eigen::MatrixXd> path;
        path.reserve(static_cast<size_t>(steps) + 1);
        Eigen::MatrixXd x = start;
        path.push_back(x);
        for (int k = 0; k < steps; ++k) {
            x += dt * velocity(x, static_cast<double>(k) * dt);
            path.push_back(x);
        }
        return path;
    }

    double lastLoss() const { return lastLoss_; }
    const std::vector<double>& lossHistory() const { return histLoss_; }
    int trainedSteps() const { return mlp_.step(); }

private:
    Mlp mlp_;
    std::mt19937 rng_;
    unsigned seed_ = 1;
    int pool_ = 0;
    bool couplingReady_ = false;
    double lastLoss_ = 0.0;
    std::vector<Eigen::Vector2d> source_;
    std::vector<Eigen::Vector2d> target_;
    Eigen::MatrixXd plan_;
    std::vector<std::vector<double>> cdf_;
    std::vector<double> histLoss_;
};

}  // namespace neural_ot_demo

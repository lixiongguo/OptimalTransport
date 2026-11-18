// 训练 + 轨迹 + 耦合信息导出为 data.js，供可视化界面读取。
#include "neural_ot_trainer.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static std::mt19937 rng(20260617u);
static std::normal_distribution<double> gauss(0.0, 1.0);
static std::uniform_int_distribution<int> pick(0, 2);
static const Eigen::Vector2d centers[3] = {{-0.68, 0.48}, {0.68, 0.48}, {0.0, -0.55}};

static Eigen::Vector2d sampleSource(double sd) {
    return Eigen::Vector2d(-0.62 + sd * gauss(rng), -0.46 + sd * gauss(rng));
}
static Eigen::Vector2d sampleTarget(double sd) {
    const Eigen::Vector2d& c = centers[pick(rng)];
    return Eigen::Vector2d(c.x() + sd * gauss(rng), c.y() + sd * gauss(rng));
}

static void writePair(FILE* f, double x, double y) { std::fprintf(f, "[%.3f,%.3f]", x, y); }

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "data.js";
    std::vector<Eigen::Vector2d> src, tgt;
    for (int i = 0; i < 800; ++i) {
        src.push_back(sampleSource(0.30));
        tgt.push_back(sampleTarget(0.12));
    }
    std::vector<Eigen::Vector2d> showSrc, showTgt;
    for (int i = 0; i < 320; ++i) {
        showSrc.push_back(sampleSource(0.18));
        showTgt.push_back(sampleTarget(0.12));
    }

    neural_ot_demo::Trainer trainer;
    trainer.configure(800, 64, 3, 20260617u);
    trainer.refreshCoupling(src, tgt, 0.02, 30);

    std::vector<double> losses;
    for (int c = 0; c < 30; ++c) losses.push_back(trainer.train(50, 64, 2e-3));
    std::printf("训练 %d 步  loss %.4f\n", trainer.trainedSteps(), losses.back());

    Eigen::MatrixXd start(2, static_cast<Eigen::Index>(showSrc.size()));
    for (size_t i = 0; i < showSrc.size(); ++i) {
        start(0, static_cast<Eigen::Index>(i)) = showSrc[i].x();
        start(1, static_cast<Eigen::Index>(i)) = showSrc[i].y();
    }
    const int kSteps = 60;
    const std::vector<Eigen::MatrixXd> path = trainer.rollout(start, kSteps);

    // 用同样的 Sinkhorn 再算一遍耦合，用于把"监督信号"画出来。
    const Eigen::MatrixXd plan = neural_ot_demo::sinkhornPlan(src, tgt, 0.02, 30);
    std::vector<std::vector<double>> row(src.size(), std::vector<double>(tgt.size(), 0.0));
    for (size_t i = 0; i < src.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < tgt.size(); ++j) sum += plan(static_cast<int>(i), static_cast<int>(j));
        double acc = 0.0;
        for (size_t j = 0; j < tgt.size(); ++j) {
            acc += plan(static_cast<int>(i), static_cast<int>(j)) / std::max(sum, 1e-300);
            row[i][j] = acc;
        }
        row[i].back() = 1.0;
    }

    // 每个源点的条件均值 x1 = E[x1 | x0]，即"被要求去的地方"。
    auto conditionalMean = [&](size_t i) {
        Eigen::Vector2d m(0.0, 0.0);
        double prev = 0.0;
        for (size_t j = 0; j < tgt.size(); ++j) {
            const double w = row[i][j] - prev;
            prev = row[i][j];
            m += w * tgt[j];
        }
        return m;
    };

    FILE* f = std::fopen(outPath, "w");
    if (!f) {
        std::fprintf(stderr, "无法写入 %s\n", outPath);
        return 1;
    }
    std::fprintf(f, "window.OT_DATA = {\n");

    std::fprintf(f, "nFrames:%d,nPoints:%d,loss:[", kSteps + 1,
                 static_cast<int>(showSrc.size()));
    for (size_t k = 0; k < losses.size(); ++k) std::fprintf(f, "%s%.4f", k ? "," : "", losses[k]);
    std::fprintf(f, "],\n");

    std::fprintf(f, "centers:[");
    for (int k = 0; k < 3; ++k) {
        std::fprintf(f, "%s", k ? "," : "");
        writePair(f, centers[k].x(), centers[k].y());
    }
    std::fprintf(f, "],\n");

    std::fprintf(f, "targets:[");
    for (size_t i = 0; i < showTgt.size(); ++i) {
        std::fprintf(f, "%s", i ? "," : "");
        writePair(f, showTgt[i].x(), showTgt[i].y());
    }
    std::fprintf(f, "],\n");

    std::fprintf(f, "frames:[");
    for (size_t k = 0; k < path.size(); ++k) {
        std::fprintf(f, "%s[", k ? "," : "");
        for (Eigen::Index i = 0; i < path[k].cols(); ++i) {
            std::fprintf(f, "%s%.3f,%.3f", i ? "," : "", path[k](0, i), path[k](1, i));
        }
        std::fprintf(f, "]");
    }
    std::fprintf(f, "],\n");

    // 稀疏散开的耦合方向箭头（源点 -> 条件均值）。
    std::fprintf(f, "arrows:[");
    bool first = true;
    for (size_t i = 0; i < src.size(); i += 20) {
        const Eigen::Vector2d m = conditionalMean(i);
        std::fprintf(f, "%s[%.3f,%.3f,%.3f,%.3f]", first ? "" : ",", src[i].x(), src[i].y(), m.x(),
                     m.y());
        first = false;
    }
    std::fprintf(f, "],\n");

    // 单点监督信号演示：同一个 x0 在耦合下对应一整片 x1。
    // 取最靠近源分布中心的那个样本，避免展示极端离群点。
    const Eigen::Vector2d srcCenter(-0.62, -0.46);
    size_t i0 = 0;
    for (size_t i = 1; i < src.size(); ++i) {
        if ((src[i] - srcCenter).squaredNorm() < (src[i0] - srcCenter).squaredNorm()) i0 = i;
    }
    std::mt19937 localRng(7u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<double> x1s;
    for (int s = 0; s < 60; ++s) {
        const double r = uni(localRng);
        size_t j = 0;
        while (j + 1 < row[i0].size() && row[i0][j] < r) ++j;
        x1s.push_back(tgt[j].x());
        x1s.push_back(tgt[j].y());
    }
    const Eigen::Vector2d mean = conditionalMean(i0);
    const Eigen::Vector2d& x0 = src[i0];

    std::fprintf(f, "pair:{x0:[%.3f,%.3f],mean:[%.3f,%.3f],x1s:[", x0.x(), x0.y(), mean.x(),
                 mean.y());
    for (size_t k = 0; k < x1s.size(); k += 2) {
        std::fprintf(f, "%s[%.3f,%.3f]", k ? "," : "", x1s[k], x1s[k + 1]);
    }
    std::fprintf(f, "],samples:[");

    const Eigen::Vector2d x1(x1s[0], x1s[1]);
    const Eigen::Vector2d u = x1 - x0;
    for (int s = 0; s <= 20; ++s) {
        const double t = 0.05 * s;
        const Eigen::Vector2d xt = (1.0 - t) * x0 + t * x1;
        Eigen::MatrixXd in(2, 1);
        in.col(0) = xt;
        const Eigen::MatrixXd v = trainer.velocity(in, t);
        std::fprintf(f,
                     "%s{t:%.2f,xt:[%.3f,%.3f],u:[%.3f,%.3f],v:[%.3f,%.3f],err:%.3f}", s ? "," : "",
                     t, xt.x(), xt.y(), u.x(), u.y(), v(0, 0), v(1, 0), (v.col(0) - u).norm());
    }
    std::fprintf(f, "]}");

    std::fprintf(f, "\n};\n");
    std::fclose(f);
    std::printf("已写出 %s\n", outPath);
    return 0;
}

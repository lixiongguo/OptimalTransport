// 本机 ImGui 窗口：Neural OT、图像插值、半离散传输。
// 启动参数 neural | interp | semi 选择打开哪一页，窗口里也可以切换。

#include "ot_problem.hpp"
#include "AHT_OT/ot_adapter.hpp"
#include "FFTOT/ot_adapter.hpp"
#include "Sinkhorn/ot_adapter.hpp"
#include "neural_ot_trainer.hpp"

#include "stb_image.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Rng {
    uint32_t state;
    explicit Rng(uint32_t seed) : state(seed) {}
    float next() {
        uint32_t t = (state += 0x6D2B79F5u);
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + (t ^ (t >> 7)) * (t | 61u);
        return static_cast<float>((t ^ (t >> 14))) / 4294967296.0f;
    }
};

float clamp01(float v) { return std::max(0.f, std::min(1.f, v)); }

float normalSample(Rng& rng) {
    const float u = std::max(1e-6f, rng.next());
    const float v = std::max(1e-6f, rng.next());
    return std::sqrt(-2.f * std::log(u)) * std::cos(2.f * static_cast<float>(M_PI) * v);
}

float smoothstep(float t) { return t * t * (3.f - 2.f * t); }

struct NeuralDemo {
    int count = 320;          // 显示粒子数，也是轨迹缓存的行数
    int kind = 0;             // 目标分布：0 三峰 / 1 圆环 / 2 双月
    float speed = 1.f;
    bool play = true;
    bool trail = true;
    bool field = true;
    bool loss = true;
    bool baseline = false;    // true = 随机配对基线，false = 训练出来的神经流
    bool training = true;
    float phase = 0.f;
    double lastTime = 0.0;

    neural_ot_demo::Trainer trainer;
    std::vector<Eigen::Vector2d> trainSource;   // 训练样本池
    std::vector<Eigen::Vector2d> trainTarget;
    std::vector<Eigen::Vector2d> showSource;    // 显示粒子
    std::vector<Eigen::Vector2d> showTarget;
    std::vector<size_t> baselinePair;           // 基线的随机配对
    std::vector<Eigen::MatrixXd> path;          // 每个显示粒子的积分轨迹
    int framesSinceCache = 0;

    ImVec2 targetPoint(int kindIndex, int i, int n, Rng& rng) const {
        if (kindIndex == 0) {
            const ImVec2 centers[3] = {{-0.68f, 0.48f}, {0.68f, 0.48f}, {0.f, -0.55f}};
            const ImVec2 c = centers[i % 3];
            return {c.x + 0.12f * normalSample(rng), c.y + 0.12f * normalSample(rng)};
        }
        if (kindIndex == 1) {
            const float a = 2.f * static_cast<float>(M_PI) * (static_cast<float>(i) / n + 0.08f * rng.next());
            const float r = 0.68f + 0.08f * normalSample(rng);
            return {r * std::cos(a), r * std::sin(a)};
        }
        const int side = i % 2;
        const float a = static_cast<float>(M_PI) * rng.next();
        const float r = 0.62f + 0.05f * normalSample(rng);
        const float x = r * std::cos(a) + (side ? 0.34f : -0.34f);
        const float y = side ? -r * std::sin(a) + 0.18f : r * std::sin(a) - 0.18f;
        return {x, y};
    }

    void reset() {
        Rng rng(20260617u);
        // 训练样本比显示粒子更宽（σ 0.30 vs 0.18）：速度场在分布边缘也要有监督，
        // 否则外推出来的 v_theta 会把尾巴上的粒子甩飞。
        trainSource.clear();
        trainTarget.clear();
        for (int i = 0; i < 800; ++i) {
            trainSource.emplace_back(-0.62 + 0.30 * normalSample(rng), -0.46 + 0.30 * normalSample(rng));
            const ImVec2 q = targetPoint(kind, i, 800, rng);
            trainTarget.emplace_back(q.x, q.y);
        }
        showSource.clear();
        showTarget.clear();
        for (int i = 0; i < count; ++i) {
            showSource.emplace_back(-0.62 + 0.18 * normalSample(rng), -0.46 + 0.18 * normalSample(rng));
            const ImVec2 q = targetPoint(kind, i, count, rng);
            showTarget.emplace_back(q.x, q.y);
        }
        baselinePair.resize(showTarget.size());
        for (size_t i = 0; i < baselinePair.size(); ++i) baselinePair[i] = i;
        std::mt19937 shuffleRng(4217u);
        std::shuffle(baselinePair.begin(), baselinePair.end(), shuffleRng);

        trainer.configure(static_cast<int>(trainSource.size()), 64, 3, 20260617u);
        trainer.refreshCoupling(trainSource, trainTarget, 0.02, 30);
        phase = 0.f;
        rebuildPath();
    }

    // 用当前速度场把显示粒子重新积分一遍（只在训练推进后做）。
    void rebuildPath() {
        if (showSource.empty()) return;
        Eigen::MatrixXd start(2, static_cast<Eigen::Index>(showSource.size()));
        for (size_t i = 0; i < showSource.size(); ++i) {
            start(0, static_cast<Eigen::Index>(i)) = showSource[i].x();
            start(1, static_cast<Eigen::Index>(i)) = showSource[i].y();
        }
        path = trainer.rollout(start, 12);
        framesSinceCache = 0;
    }

    void tick() {
        const double now = glfwGetTime();
        const float dt = static_cast<float>(std::min(0.05, now - lastTime));
        lastTime = now;
        if (play) phase = std::fmod(phase + dt * speed * 0.18f, 1.f);
        if (training && !baseline && !showSource.empty()) {
            trainer.train(6, 64, 2e-3);
            if (++framesSinceCache >= 4) rebuildPath();
        }
    }

    ImVec2 particleAt(size_t i, float t) const {
        const Eigen::Index index = static_cast<Eigen::Index>(i);
        if (baseline || path.size() < 2) {
            const float s = smoothstep(t);
            const Eigen::Vector2d& a = showSource[i];
            const Eigen::Vector2d& b = showTarget[baselinePair[i]];
            return ImVec2(static_cast<float>(a.x() * (1.0 - s) + b.x() * s),
                          static_cast<float>(a.y() * (1.0 - s) + b.y() * s));
        }
        const int last = static_cast<int>(path.size()) - 1;
        const float f = t * static_cast<float>(last);
        const int k = std::min(last - 1, std::max(0, static_cast<int>(std::floor(f))));
        const float frac = std::clamp(f - static_cast<float>(k), 0.f, 1.f);
        const Eigen::MatrixXd& a = path[static_cast<size_t>(k)];
        const Eigen::MatrixXd& b = path[static_cast<size_t>(k + 1)];
        return ImVec2(static_cast<float>(a(0, index) * (1.0 - frac) + b(0, index) * frac),
                      static_cast<float>(a(1, index) * (1.0 - frac) + b(1, index) * frac));
    }

    float lossValue() const { return static_cast<float>(trainer.lastLoss()); }

    void draw(ImDrawList* draw, ImVec2 origin, ImVec2 size) const {
        const float t = 0.5f - 0.5f * std::cos(2.f * static_cast<float>(M_PI) * phase);
        draw->AddRectFilledMultiColor(origin, ImVec2(origin.x + size.x, origin.y + size.y),
                                      IM_COL32(7, 11, 20, 255), IM_COL32(7, 21, 26, 255),
                                      IM_COL32(11, 20, 48, 255), IM_COL32(7, 11, 20, 255));
        const float step = std::min(size.x, size.y) * 0.072f;
        const ImU32 grid = IM_COL32(122, 162, 255, 20);
        if (step > 4.f) {
            for (float x = std::fmod(size.x * 0.5f, step); x < size.x; x += step) {
                draw->AddLine(ImVec2(origin.x + x, origin.y), ImVec2(origin.x + x, origin.y + size.y), grid);
            }
            for (float y = std::fmod(size.y * 0.5f, step); y < size.y; y += step) {
                draw->AddLine(ImVec2(origin.x, origin.y + y), ImVec2(origin.x + size.x, origin.y + y), grid);
            }
        }
        const float scale = std::min(size.x, size.y) * 0.36f;
        const auto world = [&](ImVec2 p) {
            return ImVec2(origin.x + size.x * 0.5f + p.x * scale, origin.y + size.y * 0.54f - p.y * scale);
        };

        for (size_t i = 0; i < showTarget.size(); i += 2) {
            const ImVec2 q = world(ImVec2(static_cast<float>(showTarget[i].x()),
                                          static_cast<float>(showTarget[i].y())));
            draw->AddRectFilled(ImVec2(q.x - 1.5f, q.y - 1.5f), ImVec2(q.x + 1.5f, q.y + 1.5f),
                                IM_COL32(255, 111, 145, 50));
        }

        // 真实速度场：直接问训练出来的 v_theta，任意位置都能取到值。
        if (field && !baseline && trainer.trainedSteps() > 0) {
            const float xs[7] = {-0.9f, -0.6f, -0.3f, 0.f, 0.3f, 0.6f, 0.9f};
            const float ys[6] = {-0.75f, -0.45f, -0.15f, 0.15f, 0.45f, 0.75f};
            Eigen::MatrixXd probe(2, 42);
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 7; ++col) {
                    probe(0, row * 7 + col) = xs[col];
                    probe(1, row * 7 + col) = ys[row];
                }
            }
            const Eigen::MatrixXd flow = trainer.velocity(probe, t);
            int k = 0;
            for (float y : ys) {
                for (float x : xs) {
                    const float vx = static_cast<float>(flow(0, k));
                    const float vy = static_cast<float>(flow(1, k));
                    ++k;
                    const float len = std::max(1e-6f, std::hypot(vx, vy));
                    constexpr float arrow = 0.065f;
                    const ImVec2 a = world(ImVec2(x, y));
                    const ImVec2 b = world(ImVec2(x + vx / len * arrow, y + vy / len * arrow));
                    draw->AddLine(a, b, IM_COL32(255, 209, 102, 92), 1.f);
                    const float ang = std::atan2(b.y - a.y, b.x - a.x);
                    const ImVec2 p1(b.x - 5.f * std::cos(ang - 0.45f), b.y - 5.f * std::sin(ang - 0.45f));
                    const ImVec2 p2(b.x - 5.f * std::cos(ang + 0.45f), b.y - 5.f * std::sin(ang + 0.45f));
                    draw->AddTriangleFilled(b, p1, p2, IM_COL32(255, 209, 102, 143));
                }
            }
        }

        if (trail) {
            for (size_t i = 0; i < showSource.size(); i += 4) {
                ImVec2 prev = world(particleAt(i, 0.f));
                for (int k = 1; k <= 10; ++k) {
                    const ImVec2 q = world(particleAt(i, t * static_cast<float>(k) / 10.f));
                    draw->AddLine(prev, q, IM_COL32(96, 150, 255, 46), 1.f);
                    prev = q;
                }
            }
        }

        for (size_t i = 0; i < showSource.size(); ++i) {
            const ImVec2 q = world(particleAt(i, t));
            const float shift = static_cast<float>((i * 2654435761u) % 1000u) / 1000.f;
            const int r = static_cast<int>(90 + 80 * shift);
            const int g = static_cast<int>(160 + 60 * shift);
            draw->AddRectFilled(ImVec2(q.x - 2.2f, q.y - 2.2f), ImVec2(q.x + 2.2f, q.y + 2.2f),
                                IM_COL32(126, 174, 255, 70));
            draw->AddRectFilled(ImVec2(q.x - 1.2f, q.y - 1.2f), ImVec2(q.x + 1.2f, q.y + 1.2f),
                                IM_COL32(r, g, 255, 230));
        }

        const ImVec2 panel(origin.x + 12.f, origin.y + 12.f);
        draw->AddRectFilled(panel, ImVec2(panel.x + 250.f, panel.y + 108.f), IM_COL32(8, 13, 24, 200), 12.f);
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 8.f), IM_COL32(244, 247, 255, 255),
                      baseline ? "随机配对（基线）" : "Neural OT（训练中）");
        char line[96];
        std::snprintf(line, sizeof(line), "flow time t      %.2f", t);
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 30.f), IM_COL32(255, 214, 107, 255), line);
        std::snprintf(line, sizeof(line), "train loss       %.4f", lossValue());
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 48.f), IM_COL32(255, 214, 107, 255), line);
        std::snprintf(line, sizeof(line), "adam steps       %d", trainer.trainedSteps());
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 66.f), IM_COL32(255, 214, 107, 255), line);
        std::snprintf(line, sizeof(line), "particles        %d", static_cast<int>(showSource.size()));
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 84.f), IM_COL32(255, 214, 107, 255), line);

        draw->AddRectFilled(ImVec2(origin.x + 12.f, origin.y + size.y - 36.f),
                            ImVec2(origin.x + 430.f, origin.y + size.y - 10.f), IM_COL32(8, 13, 24, 200), 10.f);
        draw->AddCircleFilled(ImVec2(origin.x + 28.f, origin.y + size.y - 23.f), 4.f, IM_COL32(66, 230, 164, 255));
        draw->AddText(ImVec2(origin.x + 38.f, origin.y + size.y - 30.f), IM_COL32(174, 192, 235, 255), "源");
        draw->AddCircleFilled(ImVec2(origin.x + 78.f, origin.y + size.y - 23.f), 4.f, IM_COL32(114, 165, 255, 255));
        draw->AddText(ImVec2(origin.x + 88.f, origin.y + size.y - 30.f), IM_COL32(174, 192, 235, 255), "粒子");
        draw->AddCircleFilled(ImVec2(origin.x + 140.f, origin.y + size.y - 23.f), 4.f, IM_COL32(255, 111, 145, 255));
        draw->AddText(ImVec2(origin.x + 150.f, origin.y + size.y - 30.f), IM_COL32(174, 192, 235, 255), "目标");
        draw->AddCircleFilled(ImVec2(origin.x + 210.f, origin.y + size.y - 23.f), 4.f, IM_COL32(255, 209, 102, 255));
        draw->AddText(ImVec2(origin.x + 220.f, origin.y + size.y - 30.f), IM_COL32(174, 192, 235, 255), "速度场 v_θ");

        if (loss) {
            const float w = std::min(360.f, size.x * 0.42f);
            const float h = 110.f;
            const ImVec2 box(origin.x + size.x - w - 14.f, origin.y + size.y - h - 48.f);
            draw->AddRectFilled(box, ImVec2(box.x + w, box.y + h), IM_COL32(8, 13, 24, 184), 12.f);
            draw->AddRect(box, ImVec2(box.x + w, box.y + h), IM_COL32(122, 162, 255, 51), 12.f);
            draw->AddText(ImVec2(box.x + 14.f, box.y + 8.f), IM_COL32(220, 230, 255, 255), "train loss (log)");
            const ImVec2 plot(box.x + 18.f, box.y + 36.f);
            const float pw = w - 34.f;
            const float ph = h - 52.f;
            const std::vector<double>& history = trainer.lossHistory();
            if (history.size() >= 2) {
                double lo = history[0];
                double hi = history[0];
                for (double value : history) {
                    lo = std::min(lo, value);
                    hi = std::max(hi, value);
                }
                lo = std::max(lo, 1e-6);
                hi = std::max(hi, lo * 1.000001);
                ImVec2 prev;
                for (size_t k = 0; k < history.size(); ++k) {
                    const float x = static_cast<float>(k) / static_cast<float>(history.size() - 1);
                    const double norm = (std::log(std::max(history[k], 1e-6)) - std::log(lo)) /
                                        (std::log(hi) - std::log(lo));
                    const ImVec2 q(plot.x + pw * x, plot.y + ph * (1.f - static_cast<float>(norm)));
                    if (k > 0) draw->AddLine(prev, q, IM_COL32(114, 165, 255, 255), 2.f);
                    prev = q;
                }
            } else {
                draw->AddText(ImVec2(plot.x, plot.y + ph * 0.4f), IM_COL32(140, 158, 196, 255), "开始训练后出现");
            }
        }
    }
};

struct Texture {
    GLuint id = 0;
    int width = 0;
    int height = 0;

    ~Texture() { reset(); }
    Texture() = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    void reset() {
        if (id != 0) {
            glDeleteTextures(1, &id);
            id = 0;
        }
    }

    // 数据按行自上而下传入：ImGui::Image 的 uv(0,0) 就是贴图第一行，
    // 再翻转一次反而会把整张图上下颠倒。
    void upload(const std::vector<uint8_t>& rgba, int w, int h) {
        if (w <= 0 || h <= 0 || static_cast<int>(rgba.size()) < w * h * 4) return;
        if (id == 0) glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        width = w;
        height = h;
    }

    void show(float maxSide) const {
        if (id == 0) {
            ImGui::TextUnformatted("尚无图像");
            return;
        }
        float widthPx = maxSide;
        float heightPx = maxSide;
        if (width > 0 && height > 0 && width != height) {
            if (width > height) heightPx = maxSide * static_cast<float>(height) / static_cast<float>(width);
            else widthPx = maxSide * static_cast<float>(width) / static_cast<float>(height);
        }
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(id)), ImVec2(widthPx, heightPx));
    }
};

struct Rgb {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;
};

std::string findData(const char* name) {
    const std::filesystem::path relative[] = {
        std::filesystem::path("data") / name,
        std::filesystem::path("OptimalTransport") / "data" / name,
    };
    for (const auto& path : relative) {
        std::error_code error;
        if (std::filesystem::exists(path, error)) return path.string();
    }
    return (std::filesystem::path("data") / name).string();
}

bool loadImage(const char* name, Rgb& image, std::string& error) {
    const std::string path = findData(name);
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (pixels == nullptr || w <= 0 || h <= 0) {
        error = "读不到 " + path;
        return false;
    }
    image.width = w;
    image.height = h;
    image.rgba.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

Rgb resample(const Rgb& source, int outWidth, int outHeight) {
    Rgb out;
    out.width = outWidth;
    out.height = outHeight;
    out.rgba.assign(static_cast<size_t>(outWidth) * outHeight * 4, 255);
    if (source.width <= 0 || source.height <= 0 || outWidth <= 0 || outHeight <= 0) return out;
    for (int row = 0; row < outHeight; ++row) {
        const double v = (row + 0.5) * source.height / outHeight - 0.5;
        const int y0 = std::clamp(static_cast<int>(std::floor(v)), 0, source.height - 1);
        const int y1 = std::min(y0 + 1, source.height - 1);
        const double fy = std::clamp(v - y0, 0.0, 1.0);
        for (int col = 0; col < outWidth; ++col) {
            const double u = (col + 0.5) * source.width / outWidth - 0.5;
            const int x0 = std::clamp(static_cast<int>(std::floor(u)), 0, source.width - 1);
            const int x1 = std::min(x0 + 1, source.width - 1);
            const double fx = std::clamp(u - x0, 0.0, 1.0);
            uint8_t* dst = &out.rgba[(static_cast<size_t>(row) * outWidth + col) * 4];
            for (int channel = 0; channel < 4; ++channel) {
                const auto at = [&](int x, int y) {
                    return source.rgba[(static_cast<size_t>(y) * source.width + x) * 4 + channel];
                };
                const double value = (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy) +
                                     (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy;
                dst[channel] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
            }
        }
    }
    return out;
}

Rgb resample(const Rgb& source, int n) { return resample(source, n, n); }

Rgb resampleFit(const Rgb& source, int maxEdge) {
    if (source.width <= 0 || source.height <= 0) return source;
    const double scale = static_cast<double>(maxEdge) / std::max(source.width, source.height);
    const int width = std::max(8, static_cast<int>(std::lround(source.width * scale)));
    const int height = std::max(8, static_cast<int>(std::lround(source.height * scale)));
    return resample(source, width, height);
}

Eigen::MatrixXd grayOf(const Rgb& image) {
    Eigen::MatrixXd gray(image.height, image.width);
    for (int row = 0; row < image.height; ++row) {
        for (int col = 0; col < image.width; ++col) {
            const uint8_t* p = &image.rgba[(static_cast<size_t>(row) * image.width + col) * 4];
            const double luma = (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / 255.0;
            gray(row, col) = std::max(luma, 1e-3);
        }
    }
    return gray;
}

std::unique_ptr<ot::Solver> makeSolver(int algorithm) {
    if (algorithm == 0) {
        sinkhorn::SinkhornOptions options;
        options.epsilon = 0.02;
        options.max_iterations = 30;
        return std::make_unique<sinkhorn::SinkhornSolverAdapter>(options);
    }
    if (algorithm == 1) {
        fftot::FFTOTOptions options;
        options.max_iterations = 40;
        options.tolerance = 1e-4;
        return std::make_unique<fftot::FFTOTSolverAdapter>(options);
    }
    aht_ot::AHTOptions options;
    options.max_iterations = 16;
    options.square_edge_len = 4;
    return std::make_unique<aht_ot::AhtOTSolver>(options);
}

const char* solverName(int algorithm) {
    if (algorithm == 0) return "Sinkhorn";
    if (algorithm == 1) return "FFT";
    return "AHT";
}

double sampleChannel(const Rgb& image, int channel, double row, double col) {
    const double r = std::clamp(row, 0.0, static_cast<double>(image.height - 1));
    const double c = std::clamp(col, 0.0, static_cast<double>(image.width - 1));
    const int r0 = static_cast<int>(std::floor(r));
    const int c0 = static_cast<int>(std::floor(c));
    const int r1 = std::min(r0 + 1, image.height - 1);
    const int c1 = std::min(c0 + 1, image.width - 1);
    const double tr = r - r0;
    const double tc = c - c0;
    const auto at = [&](int y, int x) {
        return image.rgba[(static_cast<size_t>(y) * image.width + x) * 4 + channel];
    };
    const double a = at(r0, c0) * (1.0 - tc) + at(r0, c1) * tc;
    const double b = at(r1, c0) * (1.0 - tc) + at(r1, c1) * tc;
    return a * (1.0 - tr) + b * tr;
}

struct InterpPair {
    const char* label;
    const char* sourceFile;
    const char* targetFile;
};

constexpr InterpPair kInterpPairs[] = {
    {"火焰", "flame_1.bmp", "flame_2.bmp"},
    {"云", "cloud_1.bmp", "cloud_2.bmp"},
    {"海水", "water_1.jpg", "water_2.jpg"},
};
constexpr int kInterpPairCount = static_cast<int>(sizeof(kInterpPairs) / sizeof(kInterpPairs[0]));

Rgb morphFrame(const Rgb& source, const Rgb& target, const ot::TransportPlan& plan, int frame, int frames) {
    const int width = source.width;
    const int height = source.height;
    Rgb out = source;
    const double towardIdentity = static_cast<double>(frame - 1) / static_cast<double>(frames);
    const double targetWeight = static_cast<double>(frame) / static_cast<double>(frames);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const Eigen::Vector2d from((col + 0.5) / width, (row + 0.5) / height);
            const Eigen::Vector2d to = plan.forward(from);
            const double mappedCol = std::clamp(to.x() * width - 0.5, 0.0, static_cast<double>(width - 1));
            const double mappedRow = std::clamp(to.y() * height - 0.5, 0.0, static_cast<double>(height - 1));
            const double sampleCol = towardIdentity * col + (1.0 - towardIdentity) * mappedCol;
            const double sampleRow = towardIdentity * row + (1.0 - towardIdentity) * mappedRow;
            uint8_t* dst = &out.rgba[(static_cast<size_t>(row) * width + col) * 4];
            for (int channel = 0; channel < 3; ++channel) {
                const double color = (1.0 - targetWeight) * dst[channel] +
                                     targetWeight * sampleChannel(target, channel, sampleRow, sampleCol);
                dst[channel] = static_cast<uint8_t>(std::lround(std::clamp(color, 0.0, 255.0)));
            }
            dst[3] = 255;
        }
    }
    return out;
}

// 一个算法的结果：映射 + 当前帧纹理。三个算法各占一份，并排展示。
struct InterpAlgorithm {
    const char* name = "";
    std::string status = "未计算";
    bool ok = false;
    std::shared_ptr<ot::TransportPlan> plan;
    Texture tex;
};

struct InterpDemo {
    int pair = 0;
    int frame = 4;
    int frames = 8;
    bool have = false;
    std::string status = "点计算";
    Rgb source;
    Rgb target;
    Texture sourceTex;
    Texture targetTex;
    std::array<InterpAlgorithm, 3> results;  // 0=Sinkhorn 1=FFT 2=AHT

    InterpDemo() {
        results[0].name = "Sinkhorn";
        results[1].name = "FFT";
        results[2].name = "AHT（自研）";
    }

    // 一次把三个算法都算一遍，结果留在 results 里并排对比。
    bool compute() {
        const int index = std::clamp(pair, 0, kInterpPairCount - 1);
        Rgb a;
        Rgb b;
        std::string error;
        if (!loadImage(kInterpPairs[index].sourceFile, a, error) ||
            !loadImage(kInterpPairs[index].targetFile, b, error)) {
            status = error;
            have = false;
            return false;
        }
        source = resampleFit(a, 96);
        target = resample(b, source.width, source.height);
        sourceTex.upload(source.rgba, source.width, source.height);
        targetTex.upload(target.rgba, target.width, target.height);

        const ot::GridDensity sourceDensity(grayOf(source));
        const ot::GridDensity targetDensity(grayOf(target));

        std::string summary;
        int solved = 0;
        for (int algorithm = 0; algorithm < static_cast<int>(results.size()); ++algorithm) {
            InterpAlgorithm& slot = results[algorithm];
            slot.ok = false;
            slot.plan.reset();
            const auto started = std::chrono::steady_clock::now();
            try {
                auto solver = makeSolver(algorithm);
                ot::Problem problem;
                problem.source = sourceDensity;
                problem.target = targetDensity;
                const ot::Result result = solver->solve(problem);
                if (!result.plan) {
                    slot.status = "传输映射为空";
                    continue;
                }
                slot.plan = result.plan;
                slot.ok = true;
                ++solved;
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
                char timing[32];
                std::snprintf(timing, sizeof(timing), "%.1f ms", ms);
                if (result.iterations > 0) {
                    slot.status = std::to_string(result.iterations) + " 步  " + timing;
                } else {
                    slot.status = std::string("初始映射后收敛  ") + timing;
                }
                summary += std::string(slot.name) + " " + timing + "   ";
            } catch (const std::exception& ex) {
                slot.status = ex.what();
            }
        }
        have = solved > 0;
        status = solved > 0 ? summary : "全部算法失败";
        refreshFrames();
        return have;
    }

    void refreshFrames() {
        const int shown = std::clamp(frame, 1, frames);
        for (InterpAlgorithm& slot : results) {
            if (!slot.ok || !slot.plan) continue;
            const Rgb out = morphFrame(source, target, *slot.plan, shown, frames);
            slot.tex.upload(out.rgba, out.width, out.height);
        }
    }
};

void labelPass(const std::vector<double>& source, int size, const std::vector<double>& siteX,
               const std::vector<double>& siteY, const std::vector<double>& weights,
               std::vector<int>& label, std::vector<double>& mass, std::vector<double>& gray,
               std::vector<double>& counts) {
    const int n = static_cast<int>(siteX.size());
    label.assign(static_cast<size_t>(size) * size, -1);
    mass.assign(static_cast<size_t>(n), 0.0);
    gray.assign(static_cast<size_t>(n), 0.0);
    counts.assign(static_cast<size_t>(n), 0.0);
    for (int row = 0; row < size; ++row) {
        const double y = (row + 0.5) / size;
        for (int col = 0; col < size; ++col) {
            const double x = (col + 0.5) / size;
            int best = 0;
            double bestPower = 1e300;
            for (int i = 0; i < n; ++i) {
                const double dx = x - siteX[static_cast<size_t>(i)];
                const double dy = y - siteY[static_cast<size_t>(i)];
                const double power = dx * dx + dy * dy - weights[static_cast<size_t>(i)];
                if (power < bestPower) {
                    bestPower = power;
                    best = i;
                }
            }
            const size_t index = static_cast<size_t>(row) * size + col;
            label[index] = best;
            const double value = source[index];
            mass[static_cast<size_t>(best)] += value;
            gray[static_cast<size_t>(best)] += value;
            counts[static_cast<size_t>(best)] += 1.0;
        }
    }
}

void sampleSites(const std::vector<double>& source, int size, int count, std::mt19937& rng,
                 std::vector<double>& siteX, std::vector<double>& siteY) {
    siteX.resize(static_cast<size_t>(count));
    siteY.resize(static_cast<size_t>(count));
    std::vector<double> cdf(source.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < source.size(); ++i) {
        sum += std::max(source[i], 1e-3);
        cdf[i] = sum;
    }
    std::uniform_real_distribution<double> pick(0.0, sum);
    std::uniform_real_distribution<double> jitter(-0.35, 0.35);
    const double spacing = 0.6 / size;
    for (int i = 0; i < count; ++i) {
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), pick(rng));
        int index = static_cast<int>(it - cdf.begin());
        if (index >= size * size) index = size * size - 1;
        double x = ((index % size) + 0.5 + jitter(rng)) / size;
        double y = ((index / size) + 0.5 + jitter(rng)) / size;
        for (int j = 0; j < i; ++j) {
            if (std::fabs(x - siteX[static_cast<size_t>(j)]) < spacing &&
                std::fabs(y - siteY[static_cast<size_t>(j)]) < spacing) {
                x += spacing * (1.0 + (i % 3));
                y += spacing * (1.0 + (i % 5));
            }
        }
        siteX[static_cast<size_t>(i)] = std::clamp(x, 0.005, 0.995);
        siteY[static_cast<size_t>(i)] = std::clamp(y, 0.005, 0.995);
    }
}

void separateSites(std::vector<double>& siteX, std::vector<double>& siteY, int size) {
    const double spacing = 1.2 / size;
    for (size_t i = 0; i < siteX.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (std::fabs(siteX[i] - siteX[j]) < spacing && std::fabs(siteY[i] - siteY[j]) < spacing) {
                siteX[i] = std::clamp(siteX[i] + spacing, 0.005, 0.995);
                siteY[i] = std::clamp(siteY[i] + spacing * 0.7, 0.005, 0.995);
            }
        }
    }
}

void lloydSites(const std::vector<double>& source, int size, std::vector<double>& siteX,
                std::vector<double>& siteY, int iterations) {
    const int n = static_cast<int>(siteX.size());
    std::vector<double> weights(static_cast<size_t>(n), 0.0);
    std::vector<int> label;
    std::vector<double> mass;
    std::vector<double> gray;
    std::vector<double> counts;
    for (int it = 0; it < iterations; ++it) {
        labelPass(source, size, siteX, siteY, weights, label, mass, gray, counts);
        std::vector<double> sumX(static_cast<size_t>(n), 0.0);
        std::vector<double> sumY(static_cast<size_t>(n), 0.0);
        for (int row = 0; row < size; ++row) {
            const double y = (row + 0.5) / size;
            for (int col = 0; col < size; ++col) {
                const size_t index = static_cast<size_t>(row) * size + col;
                const int cell = label[index];
                if (cell < 0) continue;
                const double weight = source[index];
                sumX[static_cast<size_t>(cell)] += weight * (col + 0.5) / size;
                sumY[static_cast<size_t>(cell)] += weight * y;
            }
        }
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            if (mass[static_cast<size_t>(i)] <= 1e-9) continue;
            siteX[static_cast<size_t>(i)] = std::clamp(sumX[static_cast<size_t>(i)] / mass[static_cast<size_t>(i)], 0.005, 0.995);
            siteY[static_cast<size_t>(i)] = std::clamp(sumY[static_cast<size_t>(i)] / mass[static_cast<size_t>(i)], 0.005, 0.995);
            moved = true;
        }
        if (!moved) break;
    }
}

std::vector<double> densityVector(const Rgb& image) {
    const Eigen::MatrixXd gray = grayOf(image);
    std::vector<double> out(static_cast<size_t>(image.width * image.height));
    for (int row = 0; row < image.height; ++row) {
        for (int col = 0; col < image.width; ++col) out[static_cast<size_t>(row) * image.width + col] = gray(row, col);
    }
    return out;
}

Rgb cellsImage(const std::vector<int>& label, const std::vector<double>& cellGray, int size) {
    Rgb image;
    image.width = size;
    image.height = size;
    image.rgba.assign(static_cast<size_t>(size) * size * 4, 255);
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            const int cell = label[static_cast<size_t>(row) * size + col];
            const double value = cell >= 0 ? cellGray[static_cast<size_t>(cell)] : 0.0;
            const auto byte = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0, 1.0) * 255.0));
            uint8_t* p = &image.rgba[(static_cast<size_t>(row) * size + col) * 4];
            p[0] = p[1] = p[2] = byte;
        }
    }
    return image;
}

// 原图 + 单元边界：保留像素纹理，只把幂图的分块边界描出来，便于和插值帧对照。
Rgb boundaryImage(const std::vector<double>& density, const std::vector<int>& label, int size) {
    Rgb image;
    image.width = size;
    image.height = size;
    image.rgba.assign(static_cast<size_t>(size) * size * 4, 255);
    const int neighbours[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            const size_t index = static_cast<size_t>(row) * size + col;
            const int cell = label[index];
            bool edge = false;
            for (const int* neighbour : neighbours) {
                const int rr = row + neighbour[0];
                const int cc = col + neighbour[1];
                if (rr < 0 || cc < 0 || rr >= size || cc >= size) continue;
                if (label[static_cast<size_t>(rr) * size + cc] != cell) {
                    edge = true;
                    break;
                }
            }
            uint8_t* pixel = &image.rgba[index * 4];
            if (edge) {
                pixel[0] = 255;
                pixel[1] = 168;
                pixel[2] = 32;
            } else {
                const auto byte = static_cast<uint8_t>(
                    std::lround(std::clamp(density[index], 0.0, 1.0) * 255.0));
                pixel[0] = pixel[1] = pixel[2] = byte;
            }
        }
    }
    return image;
}

// 在图上画一个站点标记（菱形），坐标是归一化的 [0,1]。
void drawSiteMarker(Rgb& image, const Eigen::Vector2d& at, int radius) {
    if (image.width <= 0 || image.height <= 0) return;
    const int col = std::clamp(static_cast<int>(std::lround(at.x() * image.width - 0.5)), 0,
                               image.width - 1);
    const int row = std::clamp(static_cast<int>(std::lround(at.y() * image.height - 0.5)), 0,
                               image.height - 1);
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (std::abs(dx) + std::abs(dy) > radius) continue;
            const int rr = row + dy;
            const int cc = col + dx;
            if (rr < 0 || cc < 0 || rr >= image.height || cc >= image.width) continue;
            uint8_t* pixel = &image.rgba[(static_cast<size_t>(rr) * image.width + cc) * 4];
            const bool edge = std::abs(dx) + std::abs(dy) == radius;
            pixel[0] = edge ? 90 : 255;
            pixel[1] = edge ? 48 : 168;
            pixel[2] = edge ? 12 : 32;
        }
    }
}

// 幂图划分下每个单元的加权重心与质量。
void cellCentroids(const std::vector<double>& density, const std::vector<int>& label, int size, int count,
                   std::vector<Eigen::Vector2d>& centroid, std::vector<double>& mass) {
    centroid.assign(static_cast<size_t>(count), Eigen::Vector2d(0.5, 0.5));
    mass.assign(static_cast<size_t>(count), 0.0);
    std::vector<double> sumX(static_cast<size_t>(count), 0.0);
    std::vector<double> sumY(static_cast<size_t>(count), 0.0);
    for (int row = 0; row < size; ++row) {
        const double y = (row + 0.5) / size;
        for (int col = 0; col < size; ++col) {
            const size_t index = static_cast<size_t>(row) * size + col;
            const int cell = label[index];
            if (cell < 0) continue;
            const double weight = density[index];
            sumX[static_cast<size_t>(cell)] += weight * (col + 0.5) / size;
            sumY[static_cast<size_t>(cell)] += weight * y;
            mass[static_cast<size_t>(cell)] += weight;
        }
    }
    for (int i = 0; i < count; ++i) {
        if (mass[static_cast<size_t>(i)] > 1e-12) {
            centroid[static_cast<size_t>(i)] =
                Eigen::Vector2d(sumX[static_cast<size_t>(i)] / mass[static_cast<size_t>(i)],
                                sumY[static_cast<size_t>(i)] / mass[static_cast<size_t>(i)]);
        }
    }
}

// 半离散（幂图）传输映射：整块单元搬到它接收质量的重心，单元内部再按质量比缩放，
// 这样插值帧是连续的位移插值，而不是一格一格跳。
class PowerCellPlan : public ot::TransportPlan {
public:
    PowerCellPlan(int size, const std::vector<int>& label,
                  std::vector<Eigen::Vector2d> sourceCentroid,
                  std::vector<Eigen::Vector2d> targetCentroid,
                  std::vector<double> scale,
                  std::vector<Eigen::Vector2d> sites)
        : size_(size),
          label_(label),
          sourceCentroid_(std::move(sourceCentroid)),
          targetCentroid_(std::move(targetCentroid)),
          scale_(std::move(scale)),
          sites_(std::move(sites)) {}

    // 站点原位与落点，用来在插值帧上把橙点一起推着走。
    const std::vector<Eigen::Vector2d>& sites() const { return sites_; }
    const std::vector<Eigen::Vector2d>& targets() const { return targetCentroid_; }

    Eigen::Vector2d forward(const Eigen::Vector2d& x) const override {
        const int col = std::clamp(static_cast<int>(x.x() * size_), 0, size_ - 1);
        const int row = std::clamp(static_cast<int>(x.y() * size_), 0, size_ - 1);
        const int cell = label_[static_cast<size_t>(row) * size_ + col];
        if (cell < 0) return x;
        const size_t index = static_cast<size_t>(cell);
        const Eigen::Vector2d mapped =
            targetCentroid_[index] + (x - sourceCentroid_[index]) * scale_[index];
        return Eigen::Vector2d(std::clamp(mapped.x(), 0.0, 1.0), std::clamp(mapped.y(), 0.0, 1.0));
    }

private:
    int size_ = 0;
    std::vector<int> label_;
    std::vector<Eigen::Vector2d> sourceCentroid_;
    std::vector<Eigen::Vector2d> targetCentroid_;
    std::vector<double> scale_;
    std::vector<Eigen::Vector2d> sites_;
};

// 把站点按 alpha 从原位推到落点，画在图上；非幂图映射（没有站点）就什么都不画。
void drawMovingSites(Rgb& image, const ot::TransportPlan& plan, double alpha) {
    const auto* cells = dynamic_cast<const PowerCellPlan*>(&plan);
    if (cells == nullptr) return;
    const int radius = std::max(1, image.width / 48);
    const std::vector<Eigen::Vector2d>& start = cells->sites();
    const std::vector<Eigen::Vector2d>& end = cells->targets();
    for (size_t i = 0; i < start.size() && i < end.size(); ++i) {
        const Eigen::Vector2d at = start[i] * (1.0 - alpha) + end[i] * alpha;
        drawSiteMarker(image, at, radius);
    }
}

bool powerDiagram(const std::vector<double>& own, const std::vector<double>& other, int size, int siteCount,
                  Rgb& image, int& iterations, std::string& error,
                  std::shared_ptr<ot::TransportPlan>* planOut = nullptr,
                  Rgb* outlineOut = nullptr) {
    const int n = std::max(2, siteCount);
    std::mt19937 rng(20240517u);
    std::vector<double> siteX;
    std::vector<double> siteY;
    sampleSites(own, size, n, rng, siteX, siteY);
    lloydSites(own, size, siteX, siteY, 3);
    separateSites(siteX, siteY, size);
    std::vector<double> weights(static_cast<size_t>(n), 0.0);

    std::vector<int> label;
    std::vector<double> mass;
    std::vector<double> gray;
    std::vector<double> counts;
    std::vector<double> zero(static_cast<size_t>(n), 0.0);
    labelPass(other, size, siteX, siteY, zero, label, mass, gray, counts);
    double otherSum = 0.0;
    for (double value : mass) otherSum += value;
    std::vector<double> target(static_cast<size_t>(n), 1.0 / n);
    if (otherSum > 0.0) {
        const double mean = otherSum / n;
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            target[static_cast<size_t>(i)] = std::clamp(mass[static_cast<size_t>(i)], 0.4 * mean, 2.5 * mean);
            total += target[static_cast<size_t>(i)];
        }
        for (int i = 0; i < n; ++i) target[static_cast<size_t>(i)] /= std::max(total, 1e-12);
    }

    const auto started = std::chrono::steady_clock::now();
    double learningRate = 0.5;
    double previousLoss = 1e300;
    iterations = 0;
    while (iterations < 80) {
        labelPass(own, size, siteX, siteY, weights, label, mass, gray, counts);
        double sum = 0.0;
        for (double value : mass) sum += value;
        const double norm = sum > 0.0 ? 1.0 / sum : 0.0;
        double loss = 0.0;
        for (int i = 0; i < n; ++i) {
            const double gap = target[static_cast<size_t>(i)] - mass[static_cast<size_t>(i)] * norm;
            loss += gap * gap;
        }
        ++iterations;
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (loss < 1e-9 || elapsed > 0.8) break;
        if (loss > previousLoss) learningRate *= 0.85;
        previousLoss = loss;
        for (int i = 0; i < n; ++i) {
            weights[static_cast<size_t>(i)] += learningRate * (target[static_cast<size_t>(i)] - mass[static_cast<size_t>(i)] * norm);
        }
    }
    labelPass(own, size, siteX, siteY, weights, label, mass, gray, counts);
    std::vector<double> cellGray(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        if (counts[static_cast<size_t>(i)] > 0.0) {
            cellGray[static_cast<size_t>(i)] =
                std::clamp(gray[static_cast<size_t>(i)] / counts[static_cast<size_t>(i)], 0.0, 1.0);
        }
    }
    image = cellsImage(label, cellGray, size);
    // 边界版要在 label 被目标图的划分覆盖之前算。
    if (outlineOut != nullptr) *outlineOut = boundaryImage(own, label, size);
    if (planOut != nullptr) {
        std::vector<Eigen::Vector2d> sourceCentroid;
        std::vector<Eigen::Vector2d> targetCentroid;
        std::vector<double> sourceMass;
        std::vector<double> targetMass;
        // 先用源的划分算重心，再拿同一套站点/权重去划目标，得到每个单元的落点。
        cellCentroids(own, label, size, n, sourceCentroid, sourceMass);
        labelPass(other, size, siteX, siteY, weights, label, mass, gray, counts);
        cellCentroids(other, label, size, n, targetCentroid, targetMass);
        std::vector<double> scale(static_cast<size_t>(n), 1.0);
        for (int i = 0; i < n; ++i) {
            if (sourceMass[static_cast<size_t>(i)] > 1e-12 && targetMass[static_cast<size_t>(i)] > 1e-12) {
                scale[static_cast<size_t>(i)] = std::clamp(
                    std::sqrt(targetMass[static_cast<size_t>(i)] / sourceMass[static_cast<size_t>(i)]),
                    0.4, 2.5);
            }
        }
        std::vector<Eigen::Vector2d> sites(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            sites[static_cast<size_t>(i)] =
                Eigen::Vector2d(siteX[static_cast<size_t>(i)], siteY[static_cast<size_t>(i)]);
        }
        *planOut = std::make_shared<PowerCellPlan>(size, label, std::move(sourceCentroid),
                                                   std::move(targetCentroid), std::move(scale),
                                                   std::move(sites));
    }
    for (int i = 0; i < n; ++i) {
        const int col = std::clamp(static_cast<int>(std::lround(siteX[static_cast<size_t>(i)] * size - 0.5)), 0, size - 1);
        const int row = std::clamp(static_cast<int>(std::lround(siteY[static_cast<size_t>(i)] * size - 0.5)), 0, size - 1);
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (std::abs(dx) + std::abs(dy) > 2) continue;
                const int rr = row + dy;
                const int cc = col + dx;
                if (rr < 0 || cc < 0 || rr >= size || cc >= size) continue;
                uint8_t* pixel = &image.rgba[(static_cast<size_t>(rr) * size + cc) * 4];
                const bool edge = std::abs(dx) + std::abs(dy) == 2;
                pixel[0] = edge ? 90 : 255;
                pixel[1] = edge ? 48 : 168;
                pixel[2] = edge ? 12 : 32;
            }
        }
    }
    error.clear();
    return true;
}

Eigen::MatrixXd pushDensity(const Eigen::MatrixXd& density, const ot::TransportPlan& plan) {
    const int size = static_cast<int>(density.rows());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(size, size);
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            const Eigen::Vector2d from((col + 0.5) / size, (row + 0.5) / size);
            const Eigen::Vector2d to = plan.forward(from);
            const double u = to.x() * size - 0.5;
            const double v = to.y() * size - 0.5;
            const int c0 = static_cast<int>(std::floor(u));
            const int r0 = static_cast<int>(std::floor(v));
            const double tc = u - c0;
            const double tr = v - r0;
            for (int kr = 0; kr <= 1; ++kr) {
                for (int kc = 0; kc <= 1; ++kc) {
                    const int rr = r0 + kr;
                    const int cc = c0 + kc;
                    if (rr < 0 || cc < 0 || rr >= size || cc >= size) continue;
                    const double weight = (kr == 0 ? 1.0 - tr : tr) * (kc == 0 ? 1.0 - tc : tc);
                    out(rr, cc) += density(row, col) * weight;
                }
            }
        }
    }
    return out;
}

Rgb fieldImage(const Eigen::MatrixXd& field) {
    const int size = static_cast<int>(field.rows());
    const double peak = std::max(field.maxCoeff(), 1e-12);
    Rgb image;
    image.width = size;
    image.height = size;
    image.rgba.assign(static_cast<size_t>(size) * size * 4, 255);
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            const double value = std::clamp(field(row, col) / peak, 0.0, 1.0);
            const auto byte = static_cast<uint8_t>(std::lround(value * 255.0));
            uint8_t* p = &image.rgba[(static_cast<size_t>(row) * size + col) * 4];
            p[0] = p[1] = p[2] = byte;
        }
    }
    return image;
}

struct SemiDemo {
    int algorithm = 0;
    int sites = 140;
    int frame = 4;
    int frames = 8;
    std::string status;
    // 沿传输映射做位移插值，得到和「图像插值」页一样的中间帧。
    Rgb morphSource;
    Rgb morphTarget;
    std::shared_ptr<ot::TransportPlan> morphPlan;
    // 分区图两种画法：块均值（默认）／ 原图 + 单元边界。
    bool outline = false;
    Rgb leftMean;
    Rgb leftOutline;
    Rgb rightMean;
    Rgb rightOutline;
    Texture sourceInput;
    Texture targetInput;
    Texture left;
    Texture right;
    Texture morph;

    void uploadCells() {
        const Rgb& leftImage = outline ? leftOutline : leftMean;
        const Rgb& rightImage = outline ? rightOutline : rightMean;
        if (leftImage.width > 0) left.upload(leftImage.rgba, leftImage.width, leftImage.height);
        if (rightImage.width > 0) right.upload(rightImage.rgba, rightImage.width, rightImage.height);
    }

    void refreshFrame() {
        if (!morphPlan || morphSource.width <= 0 || morphTarget.width <= 0) return;
        const int shown = std::clamp(frame, 1, frames);
        Rgb out = morphFrame(morphSource, morphTarget, *morphPlan, shown, frames);
        // 站点跟着一起走：从源块的站点位置平滑移到目标在同一块里的质量重心。
        drawMovingSites(out, *morphPlan, static_cast<double>(shown) / static_cast<double>(frames));
        morph.upload(out.rgba, out.width, out.height);
    }

    bool compute() {
        Rgb cameraman;
        Rgb boat;
        std::string error;
        if (!loadImage("cameraman.png", cameraman, error) || !loadImage("boat.png", boat, error)) {
            status = error;
            return false;
        }
        sourceInput.upload(cameraman.rgba, cameraman.width, cameraman.height);
        targetInput.upload(boat.rgba, boat.width, boat.height);
        morphPlan.reset();
        const auto started = std::chrono::steady_clock::now();
        try {
            if (algorithm == 0) {
                constexpr int kSize = 96;
                const Rgb leftImage = resample(cameraman, kSize);
                const Rgb rightImage = resample(boat, kSize);
                const std::vector<double> leftDensity = densityVector(leftImage);
                const std::vector<double> rightDensity = densityVector(rightImage);
                Rgb leftCells;
                Rgb rightCells;
                int leftSteps = 0;
                int rightSteps = 0;
                powerDiagram(leftDensity, rightDensity, kSize, sites, leftCells, leftSteps, error, &morphPlan,
                             &leftOutline);
                powerDiagram(rightDensity, leftDensity, kSize, sites, rightCells, rightSteps, error, nullptr,
                             &rightOutline);
                leftMean = leftCells;
                rightMean = rightCells;
                uploadCells();
                morphSource = leftImage;
                morphTarget = rightImage;
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
                status = "半离散  " + std::to_string(sites) + " 站点  " + std::to_string(leftSteps) + "/" +
                         std::to_string(rightSteps) + " 步  " + std::to_string(static_cast<int>(ms)) + " ms";
                refreshFrame();
                return true;
            }
            constexpr int kSize = 48;
            const Rgb leftImage = resample(cameraman, kSize);
            const Rgb rightImage = resample(boat, kSize);
            const Eigen::MatrixXd source = grayOf(leftImage);
            const Eigen::MatrixXd target = grayOf(rightImage);
            auto forwardSolver = makeSolver(algorithm - 1);
            auto backwardSolver = makeSolver(algorithm - 1);
            ot::Problem forward;
            forward.source = ot::GridDensity(source);
            forward.target = ot::GridDensity(target);
            const ot::Result toTarget = forwardSolver->solve(forward);
            ot::Problem backward;
            backward.source = ot::GridDensity(target);
            backward.target = ot::GridDensity(source);
            const ot::Result toSource = backwardSolver->solve(backward);
            if (!toTarget.plan || !toSource.plan) {
                status = "传输映射为空";
                return false;
            }
            const Rgb pushed = fieldImage(pushDensity(source, *toTarget.plan));
            const Rgb pulled = fieldImage(pushDensity(target, *toSource.plan));
            left.upload(pushed.rgba, pushed.width, pushed.height);
            right.upload(pulled.rgba, pulled.width, pulled.height);
            morphSource = leftImage;
            morphTarget = rightImage;
            morphPlan = toTarget.plan;
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
            status = std::string(solverName(algorithm - 1)) + "  " + std::to_string(toTarget.iterations) +
                     " 步  " + std::to_string(static_cast<int>(ms)) + " ms";
            refreshFrame();
            return true;
        } catch (const std::exception& ex) {
            status = ex.what();
            return false;
        }
    }
};

void loadCjkFont() {
    ImGuiIO& io = ImGui::GetIO();
    const char* paths[] = {
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    };
    ImFontConfig config;
    config.PixelSnapH = true;
    static const ImWchar ranges[] = {
        0x0020, 0x00FF, 0x0370, 0x03FF, 0x2000, 0x206F, 0x2200, 0x22FF,
        0x3000, 0x30FF, 0x4E00, 0x9FFF, 0xFF00, 0xFFEF, 0,
    };
    for (const char* path : paths) {
        std::error_code error;
        if (!std::filesystem::exists(path, error)) continue;
        if (io.Fonts->AddFontFromFileTTF(path, 18.f, &config, ranges)) return;
    }
    io.Fonts->AddFontDefault();
}

int runCheck() {
    Eigen::MatrixXd source = Eigen::MatrixXd::Ones(16, 16);
    Eigen::MatrixXd target = Eigen::MatrixXd::Ones(16, 16);
    source(4, 4) = 8.0;
    target(11, 10) = 8.0;
    sinkhorn::SinkhornOptions options;
    options.epsilon = 0.02;
    options.max_iterations = 12;
    sinkhorn::SinkhornSolverAdapter solver(options);
    ot::Problem problem;
    problem.source = ot::GridDensity(source);
    problem.target = ot::GridDensity(target);
    const ot::Result result = solver.solve(problem);
    if (!result.plan) {
        std::cerr << "ot_examples --check: 没有传输映射\n";
        return 1;
    }
    const Eigen::Vector2d hit = result.plan->forward(Eigen::Vector2d(0.28, 0.28));
    if (!std::isfinite(hit.x()) || !std::isfinite(hit.y())) {
        std::cerr << "ot_examples --check: 映射不是有限数\n";
        return 1;
    }
    fftot::FFTOTOptions fftOptions;
    fftOptions.max_iterations = 8;
    fftOptions.tolerance = 1e-3;
    const ot::Result fft = fftot::FFTOTSolverAdapter(fftOptions).solve(problem);
    aht_ot::AHTOptions ahtOptions;
    ahtOptions.max_iterations = 2;
    ahtOptions.square_edge_len = 4;
    const ot::Result aht = aht_ot::AhtOTSolver(ahtOptions).solve(problem);
    if (!fft.plan || !aht.plan) {
        std::cerr << "ot_examples --check: FFT 或 AHT 没有传输映射\n";
        return 1;
    }
    std::cout << "ot_examples --check  Sinkhorn " << result.iterations << " 步  FFT " << fft.iterations
              << " 步  AHT 映射 (" << hit.x() << ", " << hit.y() << ")\n";
    return 0;
}

const char* pageTitle(int page) {
    if (page == 1) return "OT 图像插值";
    if (page == 2) return "半离散传输";
    return "Neural OT";
}

}  // namespace

int main(int argc, char** argv) {
    int page = 0;
    if (argc > 1) {
        const std::string arg = argv[1];
        if (arg == "--check") return runCheck();
        if (arg == "interp") page = 1;
        else if (arg == "semi") page = 2;
        else page = 0;
    }

    if (!glfwInit()) {
        std::cerr << "GLFW 初始化失败\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow* window = glfwCreateWindow(1180, 760, pageTitle(page), nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "窗口创建失败\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    loadCjkFont();

    NeuralDemo neural;
    neural.lastTime = glfwGetTime();
    neural.reset();
    InterpDemo interp;
    SemiDemo semi;
    bool interpReady = false;
    bool semiReady = false;
    bool semiGuide = true;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImVec2 display = ImGui::GetIO().DisplaySize;
        const float guideWidth = 340.f;
        const bool showGuide = page == 2 && semiGuide;
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(showGuide ? std::max(640.f, display.x - guideWidth) : display.x, display.y));
        ImGui::Begin("最优传输", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (ImGui::Combo("例子", &page, "Neural OT\0图像插值\0半离散传输\0")) {
            glfwSetWindowTitle(window, pageTitle(page));
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%s", pageTitle(page));

        if (page == 0) {
            if (ImGui::Button(neural.play ? "暂停" : "播放")) neural.play = !neural.play;
            ImGui::SameLine();
            if (ImGui::Button("重训")) neural.reset();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(140);
            if (ImGui::Combo("目标分布", &neural.kind, "三峰混合\0圆环\0双月形\0")) neural.reset();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(160);
            if (ImGui::SliderInt("粒子数", &neural.count, 120, 640)) neural.reset();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("流动时间", &neural.speed, 0.25f, 2.5f, "%.2f");
            ImGui::SameLine();
            ImGui::Checkbox("训练", &neural.training);
            ImGui::SameLine();
            ImGui::Checkbox("随机配对基线", &neural.baseline);
            ImGui::SameLine();
            ImGui::Checkbox("轨迹", &neural.trail);
            ImGui::SameLine();
            ImGui::Checkbox("速度场", &neural.field);
            ImGui::SameLine();
            ImGui::Checkbox("Loss 曲线", &neural.loss);
            neural.tick();
            ImGui::BeginChild("neural-canvas", ImVec2(0, 0), false, ImGuiWindowFlags_NoScrollbar);
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const ImVec2 size = ImGui::GetContentRegionAvail();
            neural.draw(ImGui::GetWindowDrawList(), origin, size);
            ImGui::EndChild();
        } else if (page == 1) {
            if (!interpReady) {
                interp.compute();
                interpReady = true;
            }
            ImGui::SetNextItemWidth(180);
            if (ImGui::Combo("图像", &interp.pair, "火焰\0云\0海水\0")) interp.compute();
            ImGui::SameLine();
            if (ImGui::Button("计算全部算法")) interp.compute();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderInt("帧", &interp.frame, 1, interp.frames)) interp.refreshFrames();
            ImGui::TextUnformatted(interp.status.c_str());

            const char* pairLabel = kInterpPairs[std::clamp(interp.pair, 0, kInterpPairCount - 1)].label;
            const float small = 140.f;
            ImGui::BeginGroup();
            ImGui::Text("源 · %s", pairLabel);
            interp.sourceTex.show(small);
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::Text("目标 · %s", pairLabel);
            interp.targetTex.show(small);
            ImGui::EndGroup();

            ImGui::Spacing();
            const float side =
                std::max(96.f, std::min(300.f, (ImGui::GetContentRegionAvail().x - 56.f) / 3.f));
            for (int algorithm = 0; algorithm < static_cast<int>(interp.results.size()); ++algorithm) {
                InterpAlgorithm& slot = interp.results[algorithm];
                if (algorithm > 0) ImGui::SameLine();
                ImGui::BeginGroup();
                ImGui::TextUnformatted(slot.name);
                ImGui::TextDisabled("%s", slot.status.c_str());
                slot.tex.show(side);
                ImGui::EndGroup();
            }
        } else {
            if (!semiReady) {
                semi.compute();
                semiReady = true;
            }
            const bool power = semi.algorithm == 0;
            ImGui::SetNextItemWidth(170);
            if (ImGui::Combo("算法", &semi.algorithm, "半离散 Power 图\0Sinkhorn\0FFT\0AHT\0")) semi.compute();
            if (power) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(180);
                ImGui::SliderInt("站点数", &semi.sites, 40, 240);
                if (ImGui::IsItemDeactivatedAfterEdit()) semi.compute();
                ImGui::SameLine();
                if (ImGui::Checkbox("原图 + 边界", &semi.outline)) semi.uploadCells();
            }
            ImGui::SameLine();
            if (ImGui::Button("重算")) semi.compute();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(180);
            if (ImGui::SliderInt("插值帧", &semi.frame, 1, semi.frames)) semi.refreshFrame();
            if (!semiGuide) {
                ImGui::SameLine();
                if (ImGui::Button("说明")) semiGuide = true;
            }
            if (!semi.status.empty()) ImGui::TextDisabled("%s", semi.status.c_str());

            const float gap = 12.f;
            const float caption = ImGui::GetTextLineHeightWithSpacing();
            const float availW = ImGui::GetContentRegionAvail().x;
            const float availH = ImGui::GetContentRegionAvail().y;
            const float side = std::max(110.f, std::min((availW - gap * 2.f) / 3.f,
                                                        (availH - caption * 4.f - gap) * 0.5f));
            auto captioned = [&](const char* title, Texture& texture) {
                ImGui::BeginGroup();
                ImGui::TextUnformatted(title);
                texture.show(side);
                ImGui::EndGroup();
            };
            ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.f, 1.f), "输入");
            captioned(power ? "源密度 μ · Cameraman" : "网格密度 · Cameraman", semi.sourceInput);
            ImGui::SameLine(0.f, gap);
            captioned(power ? "目标质量 ν · Boat" : "网格密度 · Boat", semi.targetInput);
            ImGui::TextColored(ImVec4(1.f, 0.72f, 0.45f, 1.f), "输出");
            const std::string morphTitle =
                std::string(power ? "插值帧 + 站点 " : "插值帧 ") +
                std::to_string(std::clamp(semi.frame, 1, semi.frames)) + "/" +
                std::to_string(semi.frames);
            captioned(morphTitle.c_str(), semi.morph);
            ImGui::SameLine(0.f, gap);
            captioned(power ? (semi.outline ? "μ 的单元边界（叠在原图上）" : "μ 的传输单元，橙点是站点")
                            : "Cameraman 推到 Boat 的密度",
                      semi.left);
            ImGui::SameLine(0.f, gap);
            captioned(power ? (semi.outline ? "方向对调的单元边界" : "方向对调：Boat 切给站点")
                            : "Boat 推到 Cameraman 的密度",
                      semi.right);
        }
        ImGui::End();

        if (showGuide) {
            ImGui::SetNextWindowPos(ImVec2(display.x - guideWidth, 0.f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(guideWidth, display.y), ImGuiCond_Always);
            ImGui::Begin("说明", &semiGuide,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + ImGui::GetContentRegionAvail().x);
            if (semi.algorithm == 0) {
                ImGui::TextColored(ImVec4(1.f, 0.86f, 0.45f, 1.f), "半离散在算什么");
                ImGui::TextWrapped("源是连续的灰度，亮度就是要送出的质量。目标只有有限个站点，每个站点只领取一份质量。计算把源平面切成块，让每块的质量刚好送给一个站点。");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.f, 1.f), "输入");
                ImGui::TextWrapped("左上 Cameraman：源密度 μ。越亮，那里要送出的质量越多。");
                ImGui::TextWrapped("右上 Boat：不接收像素，只决定每个站点该分到多少质量。站点数越少，目标越粗。");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.55f, 0.95f, 0.7f, 1.f), "计算");
                ImGui::TextWrapped("按源图亮度放站点，亮处更密。像素归给「距离平方 − 权重」最小的站点。质量不够就加大权重，这块会变大。");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.f, 0.72f, 0.45f, 1.f), "输出");
                ImGui::TextWrapped("下排左一是插值帧 + 站点：拖动「插值帧」滑块，橙点从源块的站点位置平滑移到目标在同一块里的质量重心，块的灰度同时渐变，能直接看到质量往哪走。");
                ImGui::TextWrapped("下排中间是 μ 的分块：每块只留一个均值灰度，所以看着是马赛克。橙点接收这块的全部质量。勾上「原图 + 边界」改成把分块边界描在原图上，方便和左边的插值帧对照。右边把源和目标对调。");
            } else {
                ImGui::TextColored(ImVec4(1.f, 0.86f, 0.45f, 1.f), "这一档不是半离散");
                ImGui::TextWrapped("Sinkhorn、FFT、AHT 把两张图都留在网格上，没有站点。它们求映射 T，把源的每个像素搬到目标域，使搬完的分布接近另一张图。");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.f, 1.f), "输入");
                ImGui::TextWrapped("上排两张都是网格密度。左图当源时，右图就是目标；下面再反向算一次。");
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.f, 0.72f, 0.45f, 1.f), "输出");
                ImGui::TextWrapped("下排左一是插值帧：用正向映射做位移插值，拖动「插值帧」滑块看 Cameraman 逐步变成 Boat。");
                ImGui::TextWrapped("下排中间是推前密度。越亮，表示越多源质量被搬到这里。右边是反向推的结果。");
            }
            ImGui::PopTextWrapPos();
            ImGui::End();
        }

        ImGui::Render();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.05f, 0.07f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

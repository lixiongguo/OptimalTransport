// 本机 ImGui 窗口：Neural OT、图像插值、半离散传输。
// 启动参数 neural | interp | semi 选择打开哪一页，窗口里也可以切换。

#include "ot_problem.hpp"
#include "AHT_OT/ot_adapter.hpp"
#include "FFTOT/ot_adapter.hpp"
#include "Sinkhorn/ot_adapter.hpp"

#include "stb_image.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <Eigen/Dense>

#include <algorithm>
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

struct Particle {
    ImVec2 source;
    ImVec2 target;
    ImVec2 bend;
    float colorShift = 0.f;
};

struct NeuralDemo {
    std::vector<Particle> particles;
    int count = 1200;
    int kind = 0;
    float speed = 1.f;
    bool play = true;
    bool trail = true;
    bool field = true;
    bool loss = true;
    float phase = 0.f;
    double lastTime = 0.0;

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
        particles.clear();
        particles.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            Particle p;
            p.source = {-0.62f + 0.18f * normalSample(rng), -0.46f + 0.18f * normalSample(rng)};
            p.target = targetPoint(kind, i, count, rng);
            p.bend = {0.18f * normalSample(rng) + 0.12f * std::sin(i * 12.989f),
                      0.18f * normalSample(rng) + 0.12f * std::cos(i * 7.313f)};
            p.colorShift = rng.next();
            particles.push_back(p);
        }
        phase = 0.f;
    }

    static ImVec2 position(const Particle& p, float t) {
        const float s = smoothstep(t);
        const float arc = std::sin(static_cast<float>(M_PI) * t);
        return {p.source.x * (1.f - s) + p.target.x * s + p.bend.x * arc,
                p.source.y * (1.f - s) + p.target.y * s + p.bend.y * arc};
    }

    static ImVec2 velocity(const Particle& p, float t) {
        constexpr float eps = 0.002f;
        const ImVec2 a = position(p, std::max(0.f, std::min(1.f, t - eps)));
        const ImVec2 b = position(p, std::max(0.f, std::min(1.f, t + eps)));
        return {(b.x - a.x) / (2.f * eps), (b.y - a.y) / (2.f * eps)};
    }

    void tick() {
        const double now = glfwGetTime();
        const float dt = static_cast<float>(std::min(0.05, now - lastTime));
        lastTime = now;
        if (play) phase = std::fmod(phase + dt * speed * 0.18f, 1.f);
    }

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

        for (size_t i = 0; i < particles.size(); i += 2) {
            const ImVec2 q = world(particles[i].target);
            draw->AddRectFilled(ImVec2(q.x - 1.5f, q.y - 1.5f), ImVec2(q.x + 1.5f, q.y + 1.5f),
                                IM_COL32(255, 111, 145, 50));
        }
        if (field && !particles.empty()) {
            const float xs[7] = {-0.9f, -0.6f, -0.3f, 0.f, 0.3f, 0.6f, 0.9f};
            const float ys[6] = {-0.75f, -0.45f, -0.15f, 0.15f, 0.45f, 0.75f};
            for (float x : xs) {
                for (float y : ys) {
                    const Particle* nearest = &particles[0];
                    float best = 1e30f;
                    for (size_t i = 0; i < particles.size(); i += 25) {
                        const ImVec2 q = position(particles[i], t);
                        const float d = (q.x - x) * (q.x - x) + (q.y - y) * (q.y - y);
                        if (d < best) {
                            best = d;
                            nearest = &particles[i];
                        }
                    }
                    const ImVec2 v = velocity(*nearest, t);
                    const float len = std::max(1e-6f, std::hypot(v.x, v.y));
                    constexpr float arrow = 0.065f;
                    const ImVec2 a = world(ImVec2(x, y));
                    const ImVec2 b = world(ImVec2(x + v.x / len * arrow, y + v.y / len * arrow));
                    draw->AddLine(a, b, IM_COL32(255, 209, 102, 92), 1.f);
                    const float ang = std::atan2(b.y - a.y, b.x - a.x);
                    const ImVec2 p1(b.x - 5.f * std::cos(ang - 0.45f), b.y - 5.f * std::sin(ang - 0.45f));
                    const ImVec2 p2(b.x - 5.f * std::cos(ang + 0.45f), b.y - 5.f * std::sin(ang + 0.45f));
                    draw->AddTriangleFilled(b, p1, p2, IM_COL32(255, 209, 102, 143));
                }
            }
        }
        if (trail) {
            for (size_t i = 0; i < particles.size(); i += 6) {
                const Particle& p = particles[i];
                const int alpha = static_cast<int>(255.f * (0.055f + 0.08f * p.colorShift));
                ImVec2 prev = world(position(p, 0.f));
                for (int k = 1; k <= 10; ++k) {
                    const ImVec2 q = world(position(p, t * static_cast<float>(k) / 10.f));
                    draw->AddLine(prev, q, IM_COL32(96, 150, 255, alpha), 1.f);
                    prev = q;
                }
            }
        }
        for (const Particle& p : particles) {
            const ImVec2 q = world(position(p, t));
            const int r = static_cast<int>(90 + 80 * p.colorShift);
            const int g = static_cast<int>(160 + 60 * p.colorShift);
            draw->AddRectFilled(ImVec2(q.x - 2.2f, q.y - 2.2f), ImVec2(q.x + 2.2f, q.y + 2.2f),
                                IM_COL32(126, 174, 255, 70));
            draw->AddRectFilled(ImVec2(q.x - 1.2f, q.y - 1.2f), ImVec2(q.x + 1.2f, q.y + 1.2f),
                                IM_COL32(r, g, 255, 230));
        }

        float kinetic = 0.f;
        float match = 0.f;
        int samples = 0;
        for (size_t i = 0; i < particles.size(); i += 8) {
            const Particle& p = particles[i];
            const ImVec2 v = velocity(p, t);
            const ImVec2 q = position(p, t);
            kinetic += 0.5f * (v.x * v.x + v.y * v.y);
            match += (q.x - p.target.x) * (q.x - p.target.x) + (q.y - p.target.y) * (q.y - p.target.y);
            ++samples;
        }
        if (samples > 0) {
            kinetic /= static_cast<float>(samples);
            match /= static_cast<float>(samples);
        }

        const ImVec2 panel(origin.x + 12.f, origin.y + 12.f);
        draw->AddRectFilled(panel, ImVec2(panel.x + 230.f, panel.y + 108.f), IM_COL32(8, 13, 24, 200), 12.f);
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 8.f), IM_COL32(244, 247, 255, 255), "训练态势");
        char line[96];
        std::snprintf(line, sizeof(line), "flow time t        %.2f", t);
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 30.f), IM_COL32(255, 214, 107, 255), line);
        std::snprintf(line, sizeof(line), "kinetic energy   %.3f", kinetic);
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 48.f), IM_COL32(255, 214, 107, 255), line);
        std::snprintf(line, sizeof(line), "terminal match   %.3f", match);
        draw->AddText(ImVec2(panel.x + 12.f, panel.y + 66.f), IM_COL32(255, 214, 107, 255), line);
        std::snprintf(line, sizeof(line), "particles          %d", static_cast<int>(particles.size()));
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
        draw->AddText(ImVec2(origin.x + 220.f, origin.y + size.y - 30.f), IM_COL32(174, 192, 235, 255), "速度场");

        if (loss) {
            const float w = std::min(360.f, size.x * 0.42f);
            const float h = 110.f;
            const ImVec2 box(origin.x + size.x - w - 14.f, origin.y + size.y - h - 48.f);
            draw->AddRectFilled(box, ImVec2(box.x + w, box.y + h), IM_COL32(8, 13, 24, 184), 12.f);
            draw->AddRect(box, ImVec2(box.x + w, box.y + h), IM_COL32(122, 162, 255, 51), 12.f);
            draw->AddText(ImVec2(box.x + 14.f, box.y + 8.f), IM_COL32(220, 230, 255, 255), "training loss (示意)");
            const ImVec2 plot(box.x + 18.f, box.y + 36.f);
            const float pw = w - 34.f;
            const float ph = h - 52.f;
            auto curve = [&](ImU32 color, auto fn) {
                ImVec2 prev;
                for (int k = 0; k <= 80; ++k) {
                    const float i = static_cast<float>(k) / 80.f;
                    const float v = clamp01(fn(i));
                    const ImVec2 q(plot.x + pw * i, plot.y + ph * (1.f - v));
                    if (k > 0) draw->AddLine(prev, q, color, 2.f);
                    prev = q;
                }
            };
            curve(IM_COL32(114, 165, 255, 255), [](float i) {
                return std::exp(-4.f * i) * (1.f + 0.08f * std::sin(34.f * i));
            });
            curve(IM_COL32(255, 209, 102, 255), [](float i) {
                return 0.35f * std::exp(-2.2f * i) + 0.055f;
            });
            const float px = plot.x + pw * t;
            draw->AddLine(ImVec2(px, plot.y - 2.f), ImVec2(px, plot.y + ph), IM_COL32(255, 255, 255, 100));
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

    void upload(const std::vector<uint8_t>& rgba, int w, int h) {
        if (w <= 0 || h <= 0 || static_cast<int>(rgba.size()) < w * h * 4) return;
        std::vector<uint8_t> flipped(static_cast<size_t>(w * h * 4));
        for (int row = 0; row < h; ++row) {
            std::memcpy(&flipped[static_cast<size_t>(h - 1 - row) * w * 4],
                        &rgba[static_cast<size_t>(row) * w * 4], static_cast<size_t>(w * 4));
        }
        if (id == 0) glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, flipped.data());
        width = w;
        height = h;
    }

    void show(float side) const {
        if (id == 0) {
            ImGui::TextUnformatted("尚无图像");
            return;
        }
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(id)), ImVec2(side, side));
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

Rgb resample(const Rgb& source, int n) {
    Rgb out;
    out.width = n;
    out.height = n;
    out.rgba.assign(static_cast<size_t>(n) * n * 4, 255);
    if (source.width <= 0 || source.height <= 0) return out;
    for (int row = 0; row < n; ++row) {
        const double v = (row + 0.5) * source.height / n - 0.5;
        const int y0 = std::clamp(static_cast<int>(std::floor(v)), 0, source.height - 1);
        const int y1 = std::min(y0 + 1, source.height - 1);
        const double fy = std::clamp(v - y0, 0.0, 1.0);
        for (int col = 0; col < n; ++col) {
            const double u = (col + 0.5) * source.width / n - 0.5;
            const int x0 = std::clamp(static_cast<int>(std::floor(u)), 0, source.width - 1);
            const int x1 = std::min(x0 + 1, source.width - 1);
            const double fx = std::clamp(u - x0, 0.0, 1.0);
            uint8_t* dst = &out.rgba[(static_cast<size_t>(row) * n + col) * 4];
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

Rgb morphFrame(const Rgb& source, const Rgb& target, const ot::TransportPlan& plan, int frame, int frames) {
    const int n = source.width;
    Rgb out = source;
    const double towardIdentity = static_cast<double>(frame - 1) / static_cast<double>(frames);
    const double targetWeight = static_cast<double>(frame) / static_cast<double>(frames);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const Eigen::Vector2d from((col + 0.5) / n, (row + 0.5) / n);
            const Eigen::Vector2d to = plan.forward(from);
            const double mappedCol = std::clamp(to.x() * n - 0.5, 0.0, static_cast<double>(n - 1));
            const double mappedRow = std::clamp(to.y() * n - 0.5, 0.0, static_cast<double>(n - 1));
            const double sampleCol = towardIdentity * col + (1.0 - towardIdentity) * mappedCol;
            const double sampleRow = towardIdentity * row + (1.0 - towardIdentity) * mappedRow;
            uint8_t* dst = &out.rgba[(static_cast<size_t>(row) * n + col) * 4];
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

struct InterpDemo {
    int pair = 0;
    int algorithm = 0;
    int frame = 4;
    int frames = 8;
    bool have = false;
    std::string status = "点计算";
    Rgb source;
    Rgb target;
    Rgb current;
    Texture sourceTex;
    Texture targetTex;
    Texture frameTex;
    std::shared_ptr<ot::TransportPlan> plan;

    bool compute() {
        const char* names[2][2] = {{"flame_1.bmp", "flame_2.bmp"}, {"cameraman.png", "boat.png"}};
        Rgb a;
        Rgb b;
        std::string error;
        if (!loadImage(names[pair][0], a, error) || !loadImage(names[pair][1], b, error)) {
            status = error;
            have = false;
            return false;
        }
        constexpr int kSize = 64;
        source = resample(a, kSize);
        target = resample(b, kSize);
        const auto started = std::chrono::steady_clock::now();
        try {
            auto solver = makeSolver(algorithm);
            ot::Problem problem;
            problem.source = ot::GridDensity(grayOf(source));
            problem.target = ot::GridDensity(grayOf(target));
            const ot::Result result = solver->solve(problem);
            if (!result.plan) {
                status = "传输映射为空";
                have = false;
                return false;
            }
            plan = result.plan;
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
            status = std::string(solverName(algorithm)) + "  " + std::to_string(result.iterations) +
                     " 步  " + std::to_string(static_cast<int>(ms)) + " ms";
            have = true;
            refreshFrame();
            sourceTex.upload(source.rgba, source.width, source.height);
            targetTex.upload(target.rgba, target.width, target.height);
            return true;
        } catch (const std::exception& ex) {
            status = ex.what();
            have = false;
            return false;
        }
    }

    void refreshFrame() {
        if (!have || !plan) return;
        const int shown = std::clamp(frame, 1, frames);
        current = morphFrame(source, target, *plan, shown, frames);
        frameTex.upload(current.rgba, current.width, current.height);
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

bool powerDiagram(const std::vector<double>& own, const std::vector<double>& other, int size, int siteCount,
                  Rgb& image, int& iterations, std::string& error) {
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
    std::string status = "点计算";
    Texture left;
    Texture right;

    bool compute() {
        Rgb cameraman;
        Rgb boat;
        std::string error;
        if (!loadImage("cameraman.png", cameraman, error) || !loadImage("boat.png", boat, error)) {
            status = error;
            return false;
        }
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
                powerDiagram(leftDensity, rightDensity, kSize, sites, leftCells, leftSteps, error);
                powerDiagram(rightDensity, leftDensity, kSize, sites, rightCells, rightSteps, error);
                left.upload(leftCells.rgba, leftCells.width, leftCells.height);
                right.upload(rightCells.rgba, rightCells.width, rightCells.height);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
                status = "半离散  " + std::to_string(sites) + " 站点  " + std::to_string(leftSteps) + "/" +
                         std::to_string(rightSteps) + " 步  " + std::to_string(static_cast<int>(ms)) + " ms";
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
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
            status = std::string(solverName(algorithm - 1)) + "  " + std::to_string(toTarget.iterations) +
                     " 步  " + std::to_string(static_cast<int>(ms)) + " ms";
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

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
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
            if (ImGui::Button("重新采样")) neural.reset();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(140);
            if (ImGui::Combo("目标分布", &neural.kind, "三峰混合\0圆环\0双月形\0")) neural.reset();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(160);
            if (ImGui::SliderInt("粒子数", &neural.count, 300, 1600)) neural.reset();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(140);
            ImGui::SliderFloat("流动时间", &neural.speed, 0.25f, 2.5f, "%.2f");
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
            ImGui::Combo("图像", &interp.pair, "火焰\0Cameraman / Boat\0");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(140);
            ImGui::Combo("算法", &interp.algorithm, "Sinkhorn\0FFT\0AHT\0");
            ImGui::SameLine();
            if (ImGui::Button("计算")) interp.compute();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderInt("帧", &interp.frame, 1, interp.frames)) interp.refreshFrame();
            ImGui::TextUnformatted(interp.status.c_str());
            const float side = std::min(320.f, (ImGui::GetContentRegionAvail().x - 40.f) / 3.f);
            ImGui::BeginGroup();
            ImGui::TextUnformatted("源");
            interp.sourceTex.show(side);
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::TextUnformatted("当前帧");
            interp.frameTex.show(side);
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::TextUnformatted("目标");
            interp.targetTex.show(side);
            ImGui::EndGroup();
        } else {
            if (!semiReady) {
                semi.compute();
                semiReady = true;
            }
            ImGui::SetNextItemWidth(160);
            ImGui::Combo("算法", &semi.algorithm, "半离散\0Sinkhorn\0FFT\0AHT\0");
            ImGui::SameLine();
            if (semi.algorithm == 0) {
                ImGui::SetNextItemWidth(180);
                ImGui::SliderInt("站点数", &semi.sites, 40, 240);
                ImGui::SameLine();
            }
            if (ImGui::Button("计算")) semi.compute();
            ImGui::TextUnformatted(semi.status.c_str());
            const float side = std::min(420.f, (ImGui::GetContentRegionAvail().x - 24.f) / 2.f);
            ImGui::BeginGroup();
            ImGui::TextUnformatted("Cameraman");
            semi.left.show(side);
            ImGui::EndGroup();
            ImGui::SameLine();
            ImGui::BeginGroup();
            ImGui::TextUnformatted("Boat");
            semi.right.show(side);
            ImGui::EndGroup();
        }
        ImGui::End();

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

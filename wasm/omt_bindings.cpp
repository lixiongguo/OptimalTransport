// ot-interpolation.html 的 WASM 内核。
//
// 网页侧把两张 RGBA 图像写进堆里，这里转成灰度图交给 AHT_OT_solver 解传输映射，
// 再用 create_morph_sequence_rgb 铺出插值序列，最后统一以 RGBA 交回网页。
// 导出符号与 ot-interpolation.html 里的 ccall 名字一一对应。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "aht_ot.hpp"

namespace {

// 最近一次求解的结果，按 RGBA 逐帧连续存放，供 get_omt_result_data 交出。
std::vector<uint8_t> g_result;
int g_width = 0;
int g_height = 0;
int g_frames = 0;
double g_last_ms = 0.0;

inline void rgba_to_gray(const uint8_t* src, int w, int h, aht_ot::ImageMat& out) {
    out.resize(h, w);
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const uint8_t* p = src + (static_cast<size_t>(r) * w + c) * 4;
            out(r, c) = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        }
    }
}

inline void rgba_to_rgb(const uint8_t* src, int w, int h, aht_ot::RGBImage& out) {
    for (int k = 0; k < 3; ++k) out[k].resize(h, w);
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const uint8_t* p = src + (static_cast<size_t>(r) * w + c) * 4;
            out[0](r, c) = static_cast<double>(p[0]);
            out[1](r, c) = static_cast<double>(p[1]);
            out[2](r, c) = static_cast<double>(p[2]);
        }
    }
}

inline uint8_t clamp_byte(double v) {
    const double r = std::round(v);
    if (r < 0.0) return 0;
    if (r > 255.0) return 255;
    return static_cast<uint8_t>(r);
}

}  // namespace

extern "C" {

// 返回 0 表示成功，非 0 为错误码（网页据此报错）。
int solve_omt_series(const uint8_t* data0, int len0, int w0, int h0,
                     const uint8_t* data1, int len1, int w1, int h1,
                     int channels, int square_edge_len, int num_frames,
                     int eq_method) {
    g_width = 0;
    g_height = 0;
    g_frames = 0;
    g_result.clear();

    if (channels != 4) return 1;
    if (w0 != w1 || h0 != h1) return 2;
    if (w0 <= 0 || h0 <= 0) return 3;
    if (len0 < w0 * h0 * channels || len1 < w1 * h1 * channels) return 4;
    if (num_frames <= 0) return 5;

    // 块大小必须能整除，否则密度网格与像素坐标对不上。
    const int block = std::max(1, square_edge_len);
    if (block >= w0 || block >= h0) return 6;

    const auto started = std::chrono::steady_clock::now();

    try {
        aht_ot::ImageMat gray0;
        aht_ot::ImageMat gray1;
        rgba_to_gray(data0, w0, h0, gray0);
        rgba_to_gray(data1, w1, h1, gray1);

        aht_ot::RGBImage rgb0;
        aht_ot::RGBImage rgb1;
        rgba_to_rgb(data0, w0, h0, rgb0);
        rgba_to_rgb(data1, w1, h1, rgb1);

        aht_ot::AHTOptions opt;
        opt.square_edge_len = block;
        opt.p_type = aht_ot::PType::WithComparison;
        opt.flow = aht_ot::FlowType::NonLocal;
        opt.equalization = (eq_method == 1) ? aht_ot::Equalization::HistEq
                                            : aht_ot::Equalization::MassRatio;
        // 网页是交互式的，迭代上限压到比命令行默认值更低一点。
        opt.max_iterations = 24;

        const aht_ot::AHTResult result = aht_ot::solve(gray0, gray1, opt);

        // warp 会裁掉 block + 4 的边，输出尺寸由它决定。
        const int crop = block + 4;
        const int out_w = std::max(1, w0 - crop);
        const int out_h = std::max(1, h0 - crop);

        const std::vector<aht_ot::RGBImage> frames = aht_ot::create_morph_sequence_rgb(
            result.optimal_map, rgb0, rgb1, block, num_frames);

        g_width = out_w;
        g_height = out_h;
        g_frames = static_cast<int>(frames.size());
        g_result.assign(static_cast<size_t>(g_frames) * out_w * out_h * 4, 0);

        for (int f = 0; f < g_frames; ++f) {
            const aht_ot::RGBImage& frame = frames[static_cast<size_t>(f)];
            uint8_t* dst = g_result.data() + static_cast<size_t>(f) * out_w * out_h * 4;
            for (int r = 0; r < out_h; ++r) {
                for (int c = 0; c < out_w; ++c) {
                    uint8_t* p = dst + (static_cast<size_t>(r) * out_w + c) * 4;
                    p[0] = clamp_byte(frame[0](r, c));
                    p[1] = clamp_byte(frame[1](r, c));
                    p[2] = clamp_byte(frame[2](r, c));
                    p[3] = 255;
                }
            }
        }
    } catch (const std::exception&) {
        g_result.clear();
        g_width = 0;
        g_height = 0;
        g_frames = 0;
        return 7;
    }

    const auto ended = std::chrono::steady_clock::now();
    g_last_ms = std::chrono::duration<double, std::milli>(ended - started).count();
    return 0;
}

int get_omt_frame_width() { return g_width; }

int get_omt_frame_height() { return g_height; }

int get_omt_frame_count() { return g_frames; }

int get_omt_result_size() { return static_cast<int>(g_result.size()); }

uint8_t* get_omt_result_data() { return g_result.empty() ? nullptr : g_result.data(); }

double get_omt_last_time_ms() { return g_last_ms; }

void omt_dispose() {
    std::vector<uint8_t>().swap(g_result);
    g_width = 0;
    g_height = 0;
    g_frames = 0;
    g_last_ms = 0.0;
}

}  // extern "C"

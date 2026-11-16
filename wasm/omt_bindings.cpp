// ot-interpolation.html 的 WASM 内核。
//
// 网页侧把两张 RGBA 图像写进堆里，这里转成灰度图交给 Solver/AHT_OT 解传输映射，
// 再用 create_morph_sequence_rgb 铺出插值序列，最后统一以 RGBA 交回网页。
// 导出符号与 ot-interpolation.html 里的 ccall 名字一一对应。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "aht_ot.hpp"
#include "FFTOT/fft_ot.hpp"
#include "Sinkhorn/sinkhorn.hpp"

namespace {

// 最近一次求解的结果，按 RGBA 逐帧连续存放，供 get_omt_result_data 交出。
std::vector<uint8_t> g_result;
int g_width = 0;
int g_height = 0;
int g_frames = 0;
double g_last_ms = 0.0;
int g_iterations = 0;

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

inline double clamp_unit(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

inline double sample_bilinear(const Eigen::MatrixXd& image, double row, double col) {
    const int rows = static_cast<int>(image.rows());
    const int cols = static_cast<int>(image.cols());
    row = clamp_unit(row, 0.0, static_cast<double>(rows - 1));
    col = clamp_unit(col, 0.0, static_cast<double>(cols - 1));
    const int r0 = static_cast<int>(std::floor(row));
    const int c0 = static_cast<int>(std::floor(col));
    const int r1 = std::min(r0 + 1, rows - 1);
    const int c1 = std::min(c0 + 1, cols - 1);
    const double tr = row - r0;
    const double tc = col - c0;
    const double a = image(r0, c0) * (1.0 - tc) + image(r0, c1) * tc;
    const double b = image(r1, c0) * (1.0 - tc) + image(r1, c1) * tc;
    return a * (1.0 - tr) + b * tr;
}

inline Eigen::MatrixXd pool_average(const Eigen::MatrixXd& image, int factor) {
    if (factor <= 1) return image;
    const int rows = static_cast<int>(image.rows()) / factor;
    const int cols = static_cast<int>(image.cols()) / factor;
    if (rows < 2 || cols < 2) return image;
    Eigen::MatrixXd out(rows, cols);
    const double inv = 1.0 / static_cast<double>(factor * factor);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (int i = 0; i < factor; ++i) {
                for (int j = 0; j < factor; ++j) {
                    sum += image(row * factor + i, col * factor + j);
                }
            }
            out(row, col) = sum * inv;
        }
    }
    return out;
}

inline int downsample_factor(int height, int width, int max_side) {
    int factor = 1;
    while (height / factor > max_side || width / factor > max_side) ++factor;
    return factor;
}

inline void domain_map_to_pixels(const Eigen::MatrixXd& map_x, const Eigen::MatrixXd& map_y,
                                 int height, int width, Eigen::MatrixXd& pixel_row,
                                 Eigen::MatrixXd& pixel_col) {
    const int coarse_rows = static_cast<int>(map_x.rows());
    const int coarse_cols = static_cast<int>(map_x.cols());
    pixel_row.resize(height, width);
    pixel_col.resize(height, width);
    for (int row = 0; row < height; ++row) {
        const double src_row = (static_cast<double>(row) + 0.5) * coarse_rows / height - 0.5;
        for (int col = 0; col < width; ++col) {
            const double src_col = (static_cast<double>(col) + 0.5) * coarse_cols / width - 0.5;
            const double x = sample_bilinear(map_x, src_row, src_col);
            const double y = sample_bilinear(map_y, src_row, src_col);
            pixel_col(row, col) = clamp_unit(x * width - 0.5, 0.0, static_cast<double>(width - 1));
            pixel_row(row, col) = clamp_unit(y * height - 0.5, 0.0, static_cast<double>(height - 1));
        }
    }
}

inline std::vector<aht_ot::RGBImage> morph_along_map(const Eigen::MatrixXd& pixel_row,
                                                     const Eigen::MatrixXd& pixel_col,
                                                     const aht_ot::RGBImage& img0,
                                                     const aht_ot::RGBImage& img1,
                                                     int num_frames) {
    const int height = static_cast<int>(img0[0].rows());
    const int width = static_cast<int>(img0[0].cols());
    std::vector<aht_ot::RGBImage> frames;
    frames.reserve(static_cast<size_t>(num_frames));
    for (int frame = num_frames; frame >= 1; --frame) {
        const double toward_identity =
            static_cast<double>(frame - 1) / static_cast<double>(num_frames);
        const double target_weight = static_cast<double>(frame) / static_cast<double>(num_frames);
        aht_ot::RGBImage out;
        for (int channel = 0; channel < 3; ++channel) out[channel].setZero(height, width);
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                const double sample_row = toward_identity * row +
                                          (1.0 - toward_identity) * pixel_row(row, col);
                const double sample_col = toward_identity * col +
                                          (1.0 - toward_identity) * pixel_col(row, col);
                for (int channel = 0; channel < 3; ++channel) {
                    const double target = sample_bilinear(img1[channel], sample_row, sample_col);
                    out[channel](row, col) =
                        (1.0 - target_weight) * img0[channel](row, col) + target_weight * target;
                }
            }
        }
        frames.push_back(std::move(out));
    }
    return frames;
}

inline uint8_t clamp_byte(double v) {
    const double r = std::round(v);
    if (r < 0.0) return 0;
    if (r > 255.0) return 255;
    return static_cast<uint8_t>(r);
}

inline void store_frames(const std::vector<aht_ot::RGBImage>& frames, int width, int height) {
    g_width = width;
    g_height = height;
    g_frames = static_cast<int>(frames.size());
    g_result.assign(static_cast<size_t>(g_frames) * width * height * 4, 0);
    for (int frame = 0; frame < g_frames; ++frame) {
        const aht_ot::RGBImage& image = frames[static_cast<size_t>(frame)];
        uint8_t* dst = g_result.data() + static_cast<size_t>(frame) * width * height * 4;
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                uint8_t* pixel = dst + (static_cast<size_t>(row) * width + col) * 4;
                pixel[0] = clamp_byte(image[0](row, col));
                pixel[1] = clamp_byte(image[1](row, col));
                pixel[2] = clamp_byte(image[2](row, col));
                pixel[3] = 255;
            }
        }
    }
}

}  // namespace

extern "C" {

// 返回 0 表示成功，非 0 为错误码（网页据此报错）。
int solve_omt_series(const uint8_t* data0, int len0, int w0, int h0,
                     const uint8_t* data1, int len1, int w1, int h1,
                     int channels, int square_edge_len, int num_frames,
                     int eq_method, int solver_id, int epsilon_milli) {
    g_width = 0;
    g_height = 0;
    g_frames = 0;
    g_iterations = 0;
    g_result.clear();

    if (channels != 4) return 1;
    if (w0 != w1 || h0 != h1) return 2;
    if (w0 <= 0 || h0 <= 0) return 3;
    if (len0 < w0 * h0 * channels || len1 < w1 * h1 * channels) return 4;
    if (num_frames <= 0) return 5;

    // AHT 的块必须落在图像内部，否则密度网格和像素对不上。
    const int block = std::max(1, square_edge_len);
    if (solver_id == 0 && (block >= w0 || block >= h0)) return 6;

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

        std::vector<aht_ot::RGBImage> frames;
        int out_w = w0;
        int out_h = h0;

        if (solver_id == 0) {
            aht_ot::AHTOptions opt;
            opt.square_edge_len = block;
            opt.p_type = aht_ot::PType::WithComparison;
            opt.flow = aht_ot::FlowType::NonLocal;
            opt.equalization = (eq_method == 1) ? aht_ot::Equalization::HistEq
                                                : aht_ot::Equalization::MassRatio;
            opt.max_iterations = 24;
            const aht_ot::AHTResult result = aht_ot::solve(gray0, gray1, opt);
            frames = aht_ot::create_morph_sequence_rgb(result.optimal_map, rgb0, rgb1, block,
                                                       num_frames);
            const int crop = block + 4;
            out_w = std::max(1, w0 - crop);
            out_h = std::max(1, h0 - crop);
        } else if (solver_id == 1 || solver_id == 2) {
            const int factor = downsample_factor(h0, w0, 192);
            const Eigen::MatrixXd source = pool_average(gray0, factor);
            const Eigen::MatrixXd target = pool_average(gray1, factor);
            Eigen::MatrixXd map_x;
            Eigen::MatrixXd map_y;
            if (solver_id == 1) {
                sinkhorn::SinkhornOptions opt;
                opt.epsilon = std::clamp(epsilon_milli / 1000.0, 1e-4, 0.2);
                opt.max_iterations = 30;
                const sinkhorn::SinkhornResult result =
                    sinkhorn::SinkhornSolver(opt).solve(source, target);
                g_iterations = result.iterations;
                map_x = result.map_x;
                map_y = result.map_y;
            } else {
                fftot::FFTOTOptions opt;
                opt.max_iterations = 25;
                opt.tolerance = 1e-4;
                const fftot::FFTOTResult result =
                    fftot::FFTOTSolver(opt).solve(source, target);
                g_iterations = result.iterations;
                map_x = result.map_x;
                map_y = result.map_y;
            }
            Eigen::MatrixXd pixel_row;
            Eigen::MatrixXd pixel_col;
            domain_map_to_pixels(map_x, map_y, h0, w0, pixel_row, pixel_col);
            frames = morph_along_map(pixel_row, pixel_col, rgb0, rgb1, num_frames);
        } else {
            return 8;
        }

        store_frames(frames, out_w, out_h);
    } catch (const std::exception&) {
        g_result.clear();
        g_width = 0;
        g_height = 0;
        g_frames = 0;
        g_iterations = 0;
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

int get_omt_iterations() { return g_iterations; }

void omt_dispose() {
    std::vector<uint8_t>().swap(g_result);
    g_width = 0;
    g_height = 0;
    g_frames = 0;
    g_iterations = 0;
    g_last_ms = 0.0;
}

}  // extern "C"

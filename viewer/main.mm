// 半离散最优传输（Power 图）本机窗口。
//
// 站点按源图像密度采样，每个站点要托运的质量来自另一张图像；
// 权重迭代让 Power 单元的源质量为给定值，单元面积用像素级 argmin 统计，
// 因此每次迭代只做一遍 S²×N 的比较，几十毫秒就能出一张图。
#import <Cocoa/Cocoa.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kImageSize = 256;  // 读图分辨率，与网页版一致
constexpr int kGridSize = 192;   // 解算分辨率（每像素归到一个站点）

struct Density {
    std::vector<double> values;  // row * 256 + col，第 0 行是图像底部
    double total = 1.0;
};

struct View {
    bool ready = false;
    std::vector<int> label;         // 每个像素所属站点
    std::vector<double> cellGray;   // 每个站点像素的平均灰度
    std::vector<double> siteX;
    std::vector<double> siteY;
    std::vector<double> weights;
    int iterations = 0;
    int emptyCells = 0;
    double residual = 0.0;
    double seconds = 0.0;
    std::string note;
};

Density loadDensity(const std::string& path) {
    Density out;
    NSString* file = [NSString stringWithUTF8String:path.c_str()];
    NSImage* image = [[NSImage alloc] initWithContentsOfFile:file];
    if (!image) return out;

    const int S = kImageSize;
    NSBitmapImageRep* rep = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL
                      pixelsWide:S
                      pixelsHigh:S
                   bitsPerSample:8
                 samplesPerPixel:4
                        hasAlpha:YES
                        isPlanar:NO
                  colorSpaceName:NSCalibratedRGBColorSpace
                     bitmapFormat:0
                      bytesPerRow:S * 4
                     bitsPerPixel:32];
    if (!rep) return out;

    NSGraphicsContext* context = [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
    [NSGraphicsContext saveGraphicsState];
    [NSGraphicsContext setCurrentContext:context];
    [[NSColor blackColor] setFill];
    NSRectFill(NSMakeRect(0, 0, S, S));
    [image drawInRect:NSMakeRect(0, 0, S, S)
             fromRect:NSZeroRect
            operation:NSCompositingOperationSourceOver
             fraction:1.0];
    [NSGraphicsContext restoreGraphicsState];

    out.values.assign(static_cast<size_t>(S) * S, 0.0);
    const unsigned char* data = rep.bitmapData;
    const NSInteger stride = rep.bytesPerRow;
    double total = 0.0;
    for (int r = 0; r < S; ++r) {
        const unsigned char* row = data + r * stride;
        for (int c = 0; c < S; ++c) {
            const double lum =
                (0.299 * row[c * 4] + 0.587 * row[c * 4 + 1] + 0.114 * row[c * 4 + 2]) / 255.0;
            out.values[static_cast<size_t>(S - 1 - r) * S + c] = lum;  // 位图第 0 行是顶部
            total += lum;
        }
    }
    out.total = total / static_cast<double>(S * S);
    return out;
}

// 把 256² 的图重采样到解算网格。
std::vector<double> resample(const Density& density, int size) {
    std::vector<double> out(static_cast<size_t>(size) * size, 0.0);
    if (density.values.empty()) return out;
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            const double u = (c + 0.5) * kImageSize / size - 0.5;
            const double v = (r + 0.5) * kImageSize / size - 0.5;
            const int x0 = std::clamp(static_cast<int>(std::floor(u)), 0, kImageSize - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(v)), 0, kImageSize - 1);
            const int x1 = std::min(x0 + 1, kImageSize - 1);
            const int y1 = std::min(y0 + 1, kImageSize - 1);
            const double fx = std::clamp(u - x0, 0.0, 1.0);
            const double fy = std::clamp(v - y0, 0.0, 1.0);
            const auto& src = density.values;
            const double a = src[static_cast<size_t>(y0) * kImageSize + x0];
            const double b = src[static_cast<size_t>(y0) * kImageSize + x1];
            const double c0 = src[static_cast<size_t>(y1) * kImageSize + x0];
            const double d = src[static_cast<size_t>(y1) * kImageSize + x1];
            out[static_cast<size_t>(r) * size + c] =
                (a * (1 - fx) + b * fx) * (1 - fy) + (c0 * (1 - fx) + d * fx) * fy;
        }
    }
    return out;
}

// 像素级 Power 图：每个像素归到 |x - x_i|² - w_i 最小的站点。
void labelPass(const std::vector<double>& source, int size, const std::vector<double>& siteX,
               const std::vector<double>& siteY, const std::vector<double>& weights,
               std::vector<int>& label, std::vector<double>& mass, std::vector<double>& gray,
               std::vector<double>& counts) {
    const int n = static_cast<int>(siteX.size());
    label.assign(static_cast<size_t>(size) * size, -1);
    mass.assign(static_cast<size_t>(n), 0.0);
    gray.assign(static_cast<size_t>(n), 0.0);
    counts.assign(static_cast<size_t>(n), 0.0);

    for (int r = 0; r < size; ++r) {
        const double y = (r + 0.5) / size;
        for (int c = 0; c < size; ++c) {
            const double x = (c + 0.5) / size;
            int best = 0;
            double bestPower = 1e300;
            for (int i = 0; i < n; ++i) {
                const double dx = x - siteX[i];
                const double dy = y - siteY[i];
                const double power = dx * dx + dy * dy - weights[i];
                if (power < bestPower) {
                    bestPower = power;
                    best = i;
                }
            }
            const size_t idx = static_cast<size_t>(r) * size + c;
            label[idx] = best;
            const double value = source[idx];
            mass[best] += value;
            gray[best] += value;
            counts[best] += 1.0;
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
            if (std::fabs(x - siteX[j]) < spacing && std::fabs(y - siteY[j]) < spacing) {
                x += spacing * (1.0 + (i % 3));
                y += spacing * (1.0 + (i % 5));
            }
        }
        siteX[i] = std::clamp(x, 0.005, 0.995);
        siteY[i] = std::clamp(y, 0.005, 0.995);
    }
}

// 把挨得太近的站点推开，避免出现空的 Power 单元（质量恒为 0）。
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

// 密度加权 Lloyd：站点移动到单元内按源密度的质心，让分布更均匀。
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
        for (int r = 0; r < size; ++r) {
            const double y = (r + 0.5) / size;
            for (int c = 0; c < size; ++c) {
                const size_t idx = static_cast<size_t>(r) * size + c;
                const int cell = label[idx];
                if (cell < 0) continue;
                const double w = source[idx];
                sumX[cell] += w * (c + 0.5) / size;
                sumY[cell] += w * y;
            }
        }
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            if (mass[i] <= 1e-9) continue;
            siteX[i] = std::clamp(sumX[i] / mass[i], 0.005, 0.995);
            siteY[i] = std::clamp(sumY[i] / mass[i], 0.005, 0.995);
            moved = true;
        }
        if (!moved) break;
    }
}

bool computeView(const Density& ownDensity, const Density& otherDensity, int siteCount, View& view,
                 std::string& error, bool reset) {
    if (ownDensity.values.empty() || otherDensity.values.empty()) {
        error = "图像没有读进来";
        return false;
    }
    const int size = kGridSize;
    const int n = std::max(2, siteCount);
    const std::vector<double> own = resample(ownDensity, size);
    std::vector<double> other = resample(otherDensity, size);

    const auto started = std::chrono::steady_clock::now();

    if (reset || static_cast<int>(view.siteX.size()) != n) {
        std::mt19937 rng(20240517u);
        sampleSites(own, size, n, rng, view.siteX, view.siteY);
        lloydSites(own, size, view.siteX, view.siteY, 3);
        separateSites(view.siteX, view.siteY, size);
        view.weights.assign(static_cast<size_t>(n), 0.0);
        view.iterations = 0;
    }

    std::vector<int> label;
    std::vector<double> mass;
    std::vector<double> gray;
    std::vector<double> counts;
    std::vector<double> zero(static_cast<size_t>(n), 0.0);

    // 目标质量：另一张图在站点 Voronoi 单元上的积分，归一化到 1。
    labelPass(other, size, view.siteX, view.siteY, zero, label, mass, gray, counts);
    double otherSum = 0.0;
    for (double value : mass) otherSum += value;
    std::vector<double> target(static_cast<size_t>(n), 1.0 / n);
    if (otherSum > 0.0) {
        const double mean = otherSum / n;
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            // 限制在 0.4~2.5 倍均值，保证问题可行、权重不会跑飞。
            target[static_cast<size_t>(i)] =
                std::clamp(mass[static_cast<size_t>(i)], 0.4 * mean, 2.5 * mean);
            total += target[static_cast<size_t>(i)];
        }
        for (int i = 0; i < n; ++i) target[static_cast<size_t>(i)] /= std::max(total, 1e-12);
    }

    // 权重优化：梯度为 (目标质量 - 当前质量)，损失变大就衰减学习率。
    double learningRate = 0.5;
    double previousLoss = 1e300;
    double worst = 1e9;
    int iteration = 0;
    const int maxIterations = 120;
    const double budget = 1.2;  // 秒
    while (iteration < maxIterations) {
        labelPass(own, size, view.siteX, view.siteY, view.weights, label, mass, gray, counts);
        double sum = 0.0;
        for (double value : mass) sum += value;
        const double norm = sum > 0.0 ? 1.0 / sum : 0.0;

        double loss = 0.0;
        worst = 0.0;
        for (int i = 0; i < n; ++i) {
            const size_t k = static_cast<size_t>(i);
            const double current = mass[k] * norm;
            const double gap = target[k] - current;
            loss += gap * gap;
            if (counts[k] <= 0.0) worst = 1.0;
            else worst = std::max(worst, std::fabs(gap) / target[k]);
        }
        ++iteration;
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (loss < 1e-9 || elapsed > budget) break;
        if (loss > previousLoss) learningRate *= 0.85;
        previousLoss = loss;

        for (int i = 0; i < n; ++i) {
            const size_t k = static_cast<size_t>(i);
            view.weights[k] += learningRate * (target[k] - mass[k] * norm);
        }
    }

    // 最终一遍：单元灰度取自身图像在该单元内的平均值。
    labelPass(own, size, view.siteX, view.siteY, view.weights, label, mass, gray, counts);
    view.cellGray.assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        if (counts[static_cast<size_t>(i)] > 0.0) {
            view.cellGray[static_cast<size_t>(i)] =
                std::clamp(gray[static_cast<size_t>(i)] / counts[static_cast<size_t>(i)], 0.0, 1.0);
        }
    }
    // 单元平均灰度按最大值拉到满量程，和网页版观感一致。
    double maxGray = 0.0;
    for (int i = 0; i < n; ++i) maxGray = std::max(maxGray, view.cellGray[static_cast<size_t>(i)]);
    if (maxGray > 1e-9) {
        for (int i = 0; i < n; ++i) view.cellGray[static_cast<size_t>(i)] /= maxGray;
    }
    int emptyCells = 0;
    for (int i = 0; i < n; ++i) {
        if (counts[static_cast<size_t>(i)] <= 0.0) ++emptyCells;
    }
    view.label = std::move(label);
    view.iterations += iteration;
    view.residual = worst;
    view.emptyCells = emptyCells;
    view.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    view.ready = n > 0;

    char buffer[192];
    std::snprintf(buffer, sizeof(buffer), "%d 站点 · %d 轮 · %.2f 秒 · 质量误差 %.0f%% · 空单元 %d", n,
                  view.iterations, view.seconds, worst * 100.0, emptyCells);
    view.note = buffer;
    return true;
}

CGImageRef createGrayImage(const Density& density) {
    const int S = kImageSize;
    std::vector<unsigned char> rgba(static_cast<size_t>(S) * S * 4, 255);
    for (int r = 0; r < S; ++r) {
        for (int c = 0; c < S; ++c) {
            const double v = std::clamp(density.values[static_cast<size_t>(S - 1 - r) * S + c], 0.0, 1.0);
            const auto g = static_cast<unsigned char>(std::lround(v * 255.0));
            const size_t i = (static_cast<size_t>(r) * S + c) * 4;
            rgba[i] = g;
            rgba[i + 1] = g;
            rgba[i + 2] = g;
            rgba[i + 3] = 255;
        }
    }
    CFDataRef data = CFDataCreate(nullptr, rgba.data(), static_cast<CFIndex>(rgba.size()));
    CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGImageRef image = CGImageCreate(S, S, 8, 32, static_cast<size_t>(S) * 4, space,
                                     kCGBitmapByteOrderDefault | kCGImageAlphaLast, provider, nullptr,
                                     false, kCGRenderingIntentDefault);
    CGColorSpaceRelease(space);
    CGDataProviderRelease(provider);
    CFRelease(data);
    return image;
}

// 把每个 Power 单元填成它自己图像的灰度平均值。
CGImageRef createCellImage(const View& view) {
    if (!view.ready || view.label.empty()) return nullptr;
    const int S = kGridSize;
    std::vector<unsigned char> rgba(static_cast<size_t>(S) * S * 4, 255);
    for (int r = 0; r < S; ++r) {
        for (int c = 0; c < S; ++c) {
            const size_t idx = static_cast<size_t>(r) * S + c;
            const int cell = view.label[idx];
            const double v = cell >= 0 ? view.cellGray[static_cast<size_t>(cell)] : 0.0;
            const auto g = static_cast<unsigned char>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
            const size_t i = idx * 4;
            rgba[i] = g;
            rgba[i + 1] = g;
            rgba[i + 2] = g;
            rgba[i + 3] = 255;
        }
    }
    CFDataRef data = CFDataCreate(nullptr, rgba.data(), static_cast<CFIndex>(rgba.size()));
    CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGImageRef image = CGImageCreate(S, S, 8, 32, static_cast<size_t>(S) * 4, space,
                                     kCGBitmapByteOrderDefault | kCGImageAlphaLast, provider, nullptr,
                                     false, kCGRenderingIntentDefault);
    CGColorSpaceRelease(space);
    CGDataProviderRelease(provider);
    CFRelease(data);
    return image;
}

struct Experiment {
    Density source;
    Density target;
    std::shared_ptr<View> sourceView;
    std::shared_ptr<View> targetView;
    std::mutex mutex;
};

bool loadExperiment(Experiment& experiment, std::string& error) {
    experiment.source = loadDensity("assets/cameraman.png");
    experiment.target = loadDensity("assets/boat.png");
    if (experiment.source.values.empty() || experiment.target.values.empty()) {
        experiment.source = loadDensity("../assets/cameraman.png");
        experiment.target = loadDensity("../assets/boat.png");
    }
    if (experiment.source.values.empty() || experiment.target.values.empty()) {
        error = "找不到 assets/cameraman.png 与 assets/boat.png";
        return false;
    }
    return true;
}

bool runBoth(Experiment& experiment, int siteCount, std::string& error, bool reset,
             std::vector<CGImageRef>& images) {
    std::shared_ptr<View> previousSource;
    std::shared_ptr<View> previousTarget;
    {
        std::lock_guard<std::mutex> lock(experiment.mutex);
        previousSource = experiment.sourceView;
        previousTarget = experiment.targetView;
    }

    auto source = std::make_shared<View>();
    auto target = std::make_shared<View>();
    if (!reset && previousSource) *source = *previousSource;
    if (!reset && previousTarget) *target = *previousTarget;

    std::string first;
    std::string second;
    if (!computeView(experiment.source, experiment.target, siteCount, *source, first, reset)) {
        error = "源方向失败：" + first;
        return false;
    }
    if (!computeView(experiment.target, experiment.source, siteCount, *target, second, reset)) {
        error = "目标方向失败：" + second;
        return false;
    }
    images.clear();
    images.push_back(createCellImage(*source));
    images.push_back(createCellImage(*target));

    std::lock_guard<std::mutex> lock(experiment.mutex);
    experiment.sourceView = std::move(source);
    experiment.targetView = std::move(target);
    return true;
}

int runCheck(int siteCount) {
    Experiment experiment;
    std::string error;
    std::vector<CGImageRef> images;
    if (!loadExperiment(experiment, error) || !runBoth(experiment, siteCount, error, true, images)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    std::printf("源 %s\n目标 %s\n", experiment.sourceView->note.c_str(),
                experiment.targetView->note.c_str());

    const int S = 640;
    for (int pass = 0; pass < 2; ++pass) {
        NSBitmapImageRep* rep = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:NULL
                          pixelsWide:S
                          pixelsHigh:S
                       bitsPerSample:8
                     samplesPerPixel:4
                            hasAlpha:YES
                            isPlanar:NO
                      colorSpaceName:NSCalibratedRGBColorSpace
                         bitmapFormat:0
                          bytesPerRow:S * 4
                         bitsPerPixel:32];
        if (!rep) return 1;
        NSGraphicsContext* context = [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
        [NSGraphicsContext saveGraphicsState];
        [NSGraphicsContext setCurrentContext:context];
        CGContextRef ctx = [context CGContext];
        CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
        CGContextDrawImage(ctx, CGRectMake(0, 0, S, S), images[static_cast<size_t>(pass)]);
        [NSGraphicsContext restoreGraphicsState];
        NSData* png = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
        const char* path = pass == 0 ? "/tmp/ot_source.png" : "/tmp/ot_target.png";
        [png writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
        std::printf("写出 %s\n", path);
    }
    return 0;
}

}  // namespace

@interface OtView : NSView
- (instancetype)initWithFrame:(NSRect)frame;
@end

@interface OtView ()
- (NSTextField*)label:(NSString*)text
                 size:(CGFloat)size
                 bold:(BOOL)bold
                color:(NSColor*)color;
- (void)runWithReset:(BOOL)reset;
@end

@implementation OtView {
    std::shared_ptr<Experiment> experiment;
    NSButton* computeButton;
    NSButton* refineButton;
    NSButton* originalToggle;
    NSButton* sitesToggle;
    NSTextField* statusLabel;
    NSSlider* siteSlider;
    NSTextField* countLabel;
    CGImageRef sourceImage;
    CGImageRef targetImage;
    CGImageRef sourceCells;
    CGImageRef targetCells;
    BOOL busy;
}

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;

    experiment = std::make_shared<Experiment>();
    std::string error;
    const bool loaded = loadExperiment(*experiment, error);
    if (!loaded) std::fprintf(stderr, "%s\n", error.c_str());
    sourceImage = createGrayImage(experiment->source);
    targetImage = createGrayImage(experiment->target);

    NSTextField* countTitle = [self label:@"站点数" size:12 bold:NO
                                  color:[NSColor colorWithCalibratedWhite:0.80 alpha:1.0]];
    siteSlider = [NSSlider sliderWithValue:100 minValue:30 maxValue:400 target:self
                                    action:@selector(sliderMoved:)];
    [siteSlider.widthAnchor constraintEqualToConstant:220].active = YES;
    countLabel = [self label:@"100" size:13 bold:YES
                        color:[NSColor colorWithCalibratedRed:1.0 green:0.78 blue:0.35 alpha:1]];
    [countLabel.widthAnchor constraintEqualToConstant:46].active = YES;

    computeButton = [NSButton buttonWithTitle:@"计算" target:self action:@selector(computePressed:)];
    computeButton.bezelStyle = NSBezelStyleRounded;
    refineButton = [NSButton buttonWithTitle:@"再优化" target:self action:@selector(refinePressed:)];
    refineButton.bezelStyle = NSBezelStyleRounded;
    refineButton.enabled = NO;

    NSStackView* topRow = [[NSStackView alloc] init];
    topRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    topRow.spacing = 10;
    topRow.alignment = NSLayoutAttributeCenterY;
    for (NSView* item in @[ countTitle, siteSlider, countLabel, computeButton, refineButton ]) {
        [topRow addArrangedSubview:item];
    }

    originalToggle = [NSButton checkboxWithTitle:@"显示原图" target:self action:@selector(toggleChanged:)];
    sitesToggle = [NSButton checkboxWithTitle:@"显示站点" target:self action:@selector(toggleChanged:)];
    sitesToggle.state = NSControlStateValueOn;
    statusLabel = [self label:@"正在准备…" size:11.5 bold:NO
                          color:[NSColor colorWithCalibratedWhite:0.62 alpha:1.0]];
    statusLabel.font = [NSFont monospacedSystemFontOfSize:11.5 weight:NSFontWeightRegular];
    statusLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    statusLabel.maximumNumberOfLines = 1;
    [statusLabel setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow - 1
                                           forOrientation:NSLayoutConstraintOrientationHorizontal];

    NSStackView* secondRow = [[NSStackView alloc] init];
    secondRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    secondRow.spacing = 14;
    secondRow.alignment = NSLayoutAttributeCenterY;
    for (NSView* item in @[ originalToggle, sitesToggle ]) {
        [secondRow addArrangedSubview:item];
    }

    NSStackView* toolbar = [[NSStackView alloc] init];
    toolbar.orientation = NSUserInterfaceLayoutOrientationVertical;
    toolbar.spacing = 8;
    toolbar.alignment = NSLayoutAttributeLeading;
    toolbar.translatesAutoresizingMaskIntoConstraints = NO;
    [toolbar addArrangedSubview:topRow];
    [toolbar addArrangedSubview:secondRow];
    [toolbar addArrangedSubview:statusLabel];
    [self addSubview:toolbar];
    // AppKit 的 topAnchor 以窗口顶部为原点，正数表示向下内缩。
    [NSLayoutConstraint activateConstraints:@[
        [toolbar.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:20],
        [toolbar.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-20],
        [toolbar.topAnchor constraintEqualToAnchor:self.topAnchor constant:16],
    ]];

    if (loaded) {
        [self runWithReset:YES];
    } else {
        statusLabel.stringValue = @"图像缺失：assets/cameraman.png / boat.png";
        computeButton.enabled = NO;
    }
    return self;
}

- (NSTextField*)label:(NSString*)text
                 size:(CGFloat)size
                 bold:(BOOL)bold
                color:(NSColor*)color {
    NSTextField* label = [[NSTextField alloc] init];
    label.stringValue = text;
    label.editable = NO;
    label.bezeled = NO;
    label.drawsBackground = NO;
    label.textColor = color;
    label.font = bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
    return label;
}

- (void)dealloc {
    if (sourceImage) CGImageRelease(sourceImage);
    if (targetImage) CGImageRelease(targetImage);
    if (sourceCells) CGImageRelease(sourceCells);
    if (targetCells) CGImageRelease(targetCells);
}

- (void)setBusy:(BOOL)value status:(NSString*)text {
    busy = value;
    computeButton.enabled = !value;
    BOOL hasResult = NO;
    {
        std::lock_guard<std::mutex> lock(experiment->mutex);
        hasResult = experiment->sourceView && experiment->sourceView->ready;
    }
    refineButton.enabled = !value && hasResult;
    statusLabel.stringValue = text;
}

- (void)runWithReset:(BOOL)reset {
    if (busy) return;
    const int siteCount = static_cast<int>(siteSlider.doubleValue);
    [self setBusy:YES status:[NSString stringWithFormat:@"正在计算 %d 个站点…", siteCount]];
    [self displayIfNeeded];

    std::shared_ptr<Experiment> shared = experiment;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::string error;
        std::vector<CGImageRef> images;
        const bool ok = runBoth(*shared, siteCount, error, reset, images);
        dispatch_async(dispatch_get_main_queue(), ^{
            if (!ok) {
                [self setBusy:NO status:[NSString stringWithUTF8String:error.c_str()]];
                [self setNeedsDisplay:YES];
                return;
            }
            if (sourceCells) CGImageRelease(sourceCells);
            if (targetCells) CGImageRelease(targetCells);
            sourceCells = images.size() > 0 ? images[0] : nullptr;
            targetCells = images.size() > 1 ? images[1] : nullptr;

            std::shared_ptr<View> source;
            std::shared_ptr<View> target;
            {
                std::lock_guard<std::mutex> lock(shared->mutex);
                source = shared->sourceView;
                target = shared->targetView;
            }
            // 必须走 UTF-8 转 NSString，%s 会按 MacRoman 解成乱码。
            NSString* text = [NSString stringWithFormat:@"源 %@    目标 %@",
                                                        [NSString stringWithUTF8String:source->note.c_str()],
                                                        [NSString stringWithUTF8String:target->note.c_str()]];
            [self setBusy:NO status:text];
            [self setNeedsDisplay:YES];
        });
    });
}

- (void)computePressed:(id)sender {
    [self runWithReset:YES];
}

- (void)refinePressed:(id)sender {
    [self runWithReset:NO];
}

- (void)sliderMoved:(NSSlider*)sender {
    countLabel.stringValue = [NSString stringWithFormat:@"%d", static_cast<int>(sender.doubleValue)];
}

- (void)toggleChanged:(id)sender {
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    [[NSColor colorWithCalibratedRed:0.043 green:0.055 blue:0.086 alpha:1] setFill];
    NSRectFill(self.bounds);

    std::shared_ptr<View> source;
    std::shared_ptr<View> target;
    {
        std::lock_guard<std::mutex> lock(experiment->mutex);
        source = experiment->sourceView;
        target = experiment->targetView;
    }

    const CGFloat margin = 22.0;
    const CGFloat gap = 20.0;
    const CGFloat bottomMargin = 34.0;                 // 底部留出说明文字
    const CGFloat areaTop = self.bounds.size.height - 104.0;  // 工具栏占掉的高度
    const CGFloat availableWidth = self.bounds.size.width - margin * 2 - gap;
    const CGFloat usable = std::max(areaTop - bottomMargin, 0.0);
    CGFloat side = std::max(std::min(availableWidth / 2.0, usable), 0.0);
    const CGFloat baseY = bottomMargin + std::max((usable - side) / 2.0, 0.0);
    const CGFloat startX = (self.bounds.size.width - (side * 2 + gap)) / 2.0;

    CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];
    if (!ctx) return;
    CGContextSetShouldAntialias(ctx, true);

    const BOOL showOriginal = originalToggle.state == NSControlStateValueOn;
    const BOOL showSites = sitesToggle.state == NSControlStateValueOn;

    struct Panel {
        NSString* title;
        NSString* hint;
        CGRect rect;
        const View* view;
        CGImageRef cells;
        CGImageRef original;
    };
    const Panel panels[2] = {
        {@"源 Power 图 · Cameraman → Boat", @"站点按 Cameraman 采样，质量来自 Boat",
         CGRectMake(startX, baseY, side, side), source.get(), sourceCells, sourceImage},
        {@"目标 Power 图 · Boat → Cameraman", @"站点按 Boat 采样，质量来自 Cameraman",
         CGRectMake(startX + side + gap, baseY, side, side), target.get(), targetCells, targetImage},
    };

    NSDictionary* titleAttrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:12.5 weight:NSFontWeightSemibold],
        NSForegroundColorAttributeName : [NSColor colorWithCalibratedRed:0.83 green:0.88 blue:0.96 alpha:1],
    };
    NSDictionary* hintAttrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:11],
        NSForegroundColorAttributeName : [NSColor colorWithCalibratedRed:0.45 green:0.55 blue:0.72 alpha:1],
    };

    if (side < 40.0) return;  // 窗口太小就不画面板，避免溢出
    for (int i = 0; i < 2; ++i) {
        const Panel& panel = panels[i];
        const CGRect rect = panel.rect;

        CGContextSetRGBFillColor(ctx, 0.02f, 0.03f, 0.05f, 1.0f);
        CGContextFillRect(ctx, rect);
        CGImageRef image = showOriginal ? panel.original : panel.cells;
        if (image) {
            CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
            CGContextDrawImage(ctx, rect, image);
        }

        if (showSites && panel.view) {
            const CGFloat radius = static_cast<CGFloat>(std::clamp(side / 320.0, 1.4, 3.2));
            for (size_t k = 0; k < panel.view->siteX.size(); ++k) {
                const CGFloat x = rect.origin.x + panel.view->siteX[k] * rect.size.width;
                const CGFloat y = rect.origin.y + (1.0 - panel.view->siteY[k]) * rect.size.height;
                CGContextSetRGBFillColor(ctx, 1.0f, 0.95f, 0.80f, 0.95f);
                CGContextFillEllipseInRect(ctx, CGRectMake(x - radius, y - radius, radius * 2, radius * 2));
                CGContextSetRGBStrokeColor(ctx, 0.05f, 0.07f, 0.12f, 0.85f);
                CGContextSetLineWidth(ctx, 0.6);
                CGContextStrokeEllipseInRect(ctx, CGRectMake(x - radius, y - radius, radius * 2, radius * 2));
            }
        }

        CGContextSetRGBStrokeColor(ctx, 0.28f, 0.36f, 0.52f, 0.55f);
        CGContextSetLineWidth(ctx, 1.0);
        CGContextStrokeRect(ctx, rect);

        if (CGRectGetMaxY(rect) + 22 <= areaTop) {
            [panel.title drawAtPoint:NSMakePoint(rect.origin.x, CGRectGetMaxY(rect) + 7)
                      withAttributes:titleAttrs];
        }
        [panel.hint drawAtPoint:NSMakePoint(rect.origin.x + 2, rect.origin.y - 18)
                 withAttributes:hintAttrs];
    }
}

@end

int main(int argc, char** argv) {
    @autoreleasepool {
        int siteCount = 100;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--check") == 0) {
                if (i + 1 < argc) siteCount = std::atoi(argv[i + 1]);
                return runCheck(siteCount > 0 ? siteCount : 100);
            }
        }

        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        const NSRect visible = [NSScreen mainScreen].visibleFrame;
        const CGFloat width = MIN(1180.0, visible.size.width - 40.0);
        const CGFloat height = MIN(820.0, visible.size.height - 40.0);
        OtView* view = [[OtView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
        NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, width, height)
                                                       styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        window.title = @"半离散最优传输";
        window.contentView = view;
        window.releasedWhenClosed = NO;
        window.contentMinSize = NSMakeSize(760, 520);
        [window center];
        [window makeKeyAndOrderFront:nil];
        [NSApp activate];
        [NSApp run];
    }
    return 0;
}

// 焦散透镜窗口。
//
// 均匀入射光经网格 OT 或半离散传输送到目标图案，逆 Snell 定律给出透镜斜率，
// 积成高度后再沿折射光线投到焦平面，得到模拟焦散。默认图案是 A1。
#import <Cocoa/Cocoa.h>

#include "CausticLens/caustic_lens.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace {

enum class LensPattern { Spot, Ring, A1, Photo };

struct ViewModel {
    Eigen::MatrixXd target;
    Eigen::MatrixXd caustic;
    Eigen::MatrixXd height;
    std::string note;
    bool ready = false;
};

Eigen::MatrixXd loadPhoto(const std::string& path) {
    Eigen::MatrixXd out;
    NSString* file = [NSString stringWithUTF8String:path.c_str()];
    NSImage* image = [[NSImage alloc] initWithContentsOfFile:file];
    if (!image) return out;
    int side = 512;
    for (NSImageRep* item in image.representations) {
        if (![item isKindOfClass:[NSBitmapImageRep class]]) continue;
        NSBitmapImageRep* bitmap = (NSBitmapImageRep*)item;
        if (bitmap.pixelsWide == bitmap.pixelsHigh && bitmap.pixelsWide >= 64) {
            side = static_cast<int>(bitmap.pixelsWide);
        }
    }
    const int S = side;
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

    out = Eigen::MatrixXd::Zero(S, S);
    const unsigned char* data = rep.bitmapData;
    const NSInteger stride = rep.bytesPerRow;
    for (int row = 0; row < S; ++row) {
        const unsigned char* pixels = data + row * stride;
        for (int col = 0; col < S; ++col) {
            const double alpha = pixels[col * 4 + 3] / 255.0;
            const double lum = (0.299 * pixels[col * 4] + 0.587 * pixels[col * 4 + 1] +
                                0.114 * pixels[col * 4 + 2]) /
                               255.0 * alpha;
            out(S - 1 - row, col) = lum;
        }
    }
    return out;
}

Eigen::MatrixXd patternImage(LensPattern pattern, std::string& error) {
    if (pattern == LensPattern::Spot) return caustic::spot_image(160, 0.72, 0.62, 0.07);
    if (pattern == LensPattern::Ring) return caustic::ring_image(160);
    const char* relative = pattern == LensPattern::A1 ? "data/A1_grey2.png" : "data/cameraman.png";
    const char* fallback =
        pattern == LensPattern::A1 ? "../data/A1_grey2.png" : "../data/cameraman.png";
    Eigen::MatrixXd photo = loadPhoto(relative);
    if (photo.size() == 0) photo = loadPhoto(fallback);
    if (photo.size() == 0) error = std::string("找不到 ") + relative;
    return photo;
}

CGImageRef makeImage(const Eigen::MatrixXd& field, bool relief) {
    if (field.size() == 0) return nullptr;
    const int rows = static_cast<int>(field.rows());
    const int cols = static_cast<int>(field.cols());
    double peak = field.maxCoeff();
    if (peak < 1e-12) peak = 1.0;
    std::vector<unsigned char> rgba(static_cast<size_t>(rows) * cols * 4, 255);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double t = std::clamp(field(rows - 1 - row, col) / peak, 0.0, 1.0);
            unsigned char r = 0;
            unsigned char g = 0;
            unsigned char b = 0;
            if (relief) {
                r = static_cast<unsigned char>(std::lround((0.05 + 0.95 * t) * 255.0));
                g = static_cast<unsigned char>(std::lround((0.10 + 0.55 * t) * 255.0));
                b = static_cast<unsigned char>(std::lround((0.28 - 0.18 * t) * 255.0));
            } else {
                const auto gray = static_cast<unsigned char>(std::lround(t * 255.0));
                r = g = b = gray;
            }
            const size_t index = (static_cast<size_t>(row) * cols + col) * 4;
            rgba[index] = r;
            rgba[index + 1] = g;
            rgba[index + 2] = b;
        }
    }
    CFDataRef data = CFDataCreate(nullptr, rgba.data(), static_cast<CFIndex>(rgba.size()));
    CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGImageRef image = CGImageCreate(cols, rows, 8, 32, static_cast<size_t>(cols) * 4, space,
                                     kCGBitmapByteOrderDefault | kCGImageAlphaLast, provider, nullptr,
                                     false, kCGRenderingIntentDefault);
    CGColorSpaceRelease(space);
    CGDataProviderRelease(provider);
    CFRelease(data);
    return image;
}

bool writePng(CGImageRef image, const char* path) {
    if (!image) return false;
    NSBitmapImageRep* rep = [[NSBitmapImageRep alloc] initWithCGImage:image];
    NSData* png = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    return [png writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
}

bool writeObj(const Eigen::MatrixXd& height, double width, const std::string& path) {
    std::ofstream out(path);
    if (!out) return false;
    const int rows = static_cast<int>(height.rows());
    const int cols = static_cast<int>(height.cols());
    const double dx = width / static_cast<double>(std::max(cols - 1, 1));
    const double dy = width / static_cast<double>(std::max(rows - 1, 1));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const double x = -0.5 * width + col * dx;
            const double y = -0.5 * width + row * dy;
            out << "v " << x << " " << y << " " << height(row, col) << "\n";
        }
    }
    for (int row = 0; row < rows - 1; ++row) {
        for (int col = 0; col < cols - 1; ++col) {
            const int a = row * cols + col + 1;
            const int b = a + 1;
            const int c = a + cols;
            const int d = c + 1;
            out << "f " << a << " " << b << " " << d << "\n";
            out << "f " << a << " " << d << " " << c << "\n";
        }
    }
    return true;
}

caustic::Options makeOptions(double focal, int resolution, caustic::Transport transport) {
    caustic::Options options;
    options.resolution = resolution;
    options.focal_length = focal;
    options.transport = transport;
    options.transport_iterations = 40;
    options.outer_iterations = 2;
    options.density_floor = 0.02;
    options.max_density_ratio = 40.0;
    return options;
}

int runCheck() {
    if (!caustic::integration_self_test()) {
        std::fprintf(stderr, "斜率积分没有还原出常数斜率\n");
        return 1;
    }
    caustic::Options options = makeOptions(1.5, 36, caustic::Transport::Grid);
    const caustic::Result result =
        caustic::design(caustic::spot_image(120, 0.72, 0.62, 0.07), options);
    double target_x = 0.0;
    double target_y = 0.0;
    caustic::detail::center_of_mass(result.target, target_x, target_y);
    const double error = std::hypot(result.caustic_x - target_x, result.caustic_y - target_y);
    std::printf("目标质心 (%.3f, %.3f)  焦散质心 (%.3f, %.3f)  偏差 %.4f  浮雕 %.4f\n",
                target_x, target_y, result.caustic_x, result.caustic_y, error, result.relief);
    writePng(makeImage(result.target, false), "/tmp/caustic_target.png");
    writePng(makeImage(result.caustic, false), "/tmp/caustic_sim.png");
    writePng(makeImage(result.height, true), "/tmp/caustic_lens.png");
    if (error > 0.04) {
        std::fprintf(stderr, "模拟焦散没有落到目标图案上\n");
        return 1;
    }
    return 0;
}

}  // namespace

@interface LensView : NSView
- (void)recompute;
@end

@implementation LensView {
    NSPopUpButton* patternPopup;
    NSPopUpButton* algorithmPopup;
    NSPopUpButton* resolutionPopup;
    NSSlider* focalSlider;
    NSTextField* focalLabel;
    NSButton* computeButton;
    NSButton* exportButton;
    NSTextField* statusLabel;
    CGImageRef targetImage;
    CGImageRef causticImage;
    CGImageRef lensImage;
    std::mutex mutex_;
    ViewModel model_;
    BOOL busy;
    double width_;
}

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;
    width_ = 1.0;

    NSTextField* patternTitle = [self label:@"图案" size:12 bold:NO];
    patternPopup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    [patternPopup addItemsWithTitles:@[ @"光斑", @"圆环", @"A1", @"Cameraman" ]];
    [patternPopup selectItemAtIndex:2];
    patternPopup.target = self;
    patternPopup.action = @selector(recompute);
    [patternPopup.widthAnchor constraintEqualToConstant:110].active = YES;

    NSTextField* algorithmTitle = [self label:@"算法" size:12 bold:NO];
    algorithmPopup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    [algorithmPopup addItemsWithTitles:@[ @"网格 OT", @"半离散" ]];
    algorithmPopup.target = self;
    algorithmPopup.action = @selector(recompute);
    [algorithmPopup.widthAnchor constraintEqualToConstant:100].active = YES;

    NSTextField* resolutionTitle = [self label:@"分辨率" size:12 bold:NO];
    resolutionPopup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    [resolutionPopup addItemsWithTitles:@[ @"256", @"384", @"512" ]];
    [resolutionPopup selectItemAtIndex:1];
    resolutionPopup.target = self;
    resolutionPopup.action = @selector(recompute);
    [resolutionPopup.widthAnchor constraintEqualToConstant:80].active = YES;

    NSTextField* focalTitle = [self label:@"焦距" size:12 bold:NO];
    focalSlider = [NSSlider sliderWithValue:1.5 minValue:0.8 maxValue:3.0 target:self
                                     action:@selector(focalMoved:)];
    [focalSlider.widthAnchor constraintEqualToConstant:180].active = YES;
    focalLabel = [self label:@"1.50" size:13 bold:YES];
    focalLabel.textColor = [NSColor colorWithCalibratedRed:1.0 green:0.78 blue:0.35 alpha:1];
    [focalLabel.widthAnchor constraintEqualToConstant:42].active = YES;

    computeButton = [NSButton buttonWithTitle:@"计算" target:self action:@selector(recompute)];
    computeButton.bezelStyle = NSBezelStyleRounded;
    exportButton = [NSButton buttonWithTitle:@"导出 OBJ" target:self action:@selector(exportPressed:)];
    exportButton.bezelStyle = NSBezelStyleRounded;
    exportButton.enabled = NO;

    NSStackView* row = [[NSStackView alloc] init];
    row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row.spacing = 10;
    row.alignment = NSLayoutAttributeCenterY;
    for (NSView* item in @[ patternTitle, patternPopup, algorithmTitle, algorithmPopup,
                            resolutionTitle, resolutionPopup, focalTitle, focalSlider, focalLabel,
                            computeButton, exportButton ]) {
        [row addArrangedSubview:item];
    }

    statusLabel = [self label:@"正在计算…" size:11.5 bold:NO];
    statusLabel.textColor = [NSColor colorWithCalibratedWhite:0.62 alpha:1];
    statusLabel.font = [NSFont monospacedSystemFontOfSize:11.5 weight:NSFontWeightRegular];
    statusLabel.lineBreakMode = NSLineBreakByTruncatingTail;

    NSStackView* toolbar = [[NSStackView alloc] init];
    toolbar.orientation = NSUserInterfaceLayoutOrientationVertical;
    toolbar.spacing = 8;
    toolbar.alignment = NSLayoutAttributeLeading;
    toolbar.translatesAutoresizingMaskIntoConstraints = NO;
    [toolbar addArrangedSubview:row];
    [toolbar addArrangedSubview:statusLabel];
    [self addSubview:toolbar];
    [NSLayoutConstraint activateConstraints:@[
        [toolbar.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:20],
        [toolbar.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-20],
        [toolbar.topAnchor constraintEqualToAnchor:self.topAnchor constant:16],
    ]];

    [self recompute];
    return self;
}

- (NSTextField*)label:(NSString*)text size:(CGFloat)size bold:(BOOL)bold {
    NSTextField* label = [[NSTextField alloc] init];
    label.stringValue = text;
    label.editable = NO;
    label.bezeled = NO;
    label.drawsBackground = NO;
    label.textColor = [NSColor colorWithCalibratedWhite:0.82 alpha:1];
    label.font = bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
    return label;
}

- (void)dealloc {
    if (targetImage) CGImageRelease(targetImage);
    if (causticImage) CGImageRelease(causticImage);
    if (lensImage) CGImageRelease(lensImage);
}

- (void)focalMoved:(NSSlider*)sender {
    focalLabel.stringValue = [NSString stringWithFormat:@"%.2f", sender.doubleValue];
}

- (int)selectedResolution {
    switch (resolutionPopup.indexOfSelectedItem) {
        case 0: return 256;
        case 2: return 512;
        default: return 384;
    }
}

- (LensPattern)selectedPattern {
    switch (patternPopup.indexOfSelectedItem) {
        case 1: return LensPattern::Ring;
        case 2: return LensPattern::A1;
        case 3: return LensPattern::Photo;
        default: return LensPattern::Spot;
    }
}

- (caustic::Transport)selectedTransport {
    return algorithmPopup.indexOfSelectedItem == 1 ? caustic::Transport::SemiDiscrete
                                                    : caustic::Transport::Grid;
}

- (void)recompute {
    if (busy) return;
    busy = YES;
    computeButton.enabled = NO;
    exportButton.enabled = NO;
    patternPopup.enabled = NO;
    algorithmPopup.enabled = NO;
    resolutionPopup.enabled = NO;
    focalSlider.enabled = NO;
    statusLabel.stringValue = @"正在计算透镜…";
    const LensPattern pattern = [self selectedPattern];
    const caustic::Transport transport = [self selectedTransport];
    const double focal = focalSlider.doubleValue;
    const int resolution = [self selectedResolution];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::string error;
        Eigen::MatrixXd image = patternImage(pattern, error);
        ViewModel next;
        if (image.size() == 0) {
            next.note = error.empty() ? "图案为空" : error;
        } else {
            try {
                const caustic::Result result =
                    caustic::design(image, makeOptions(focal, resolution, transport));
                next.target = result.target;
                next.caustic = result.caustic;
                next.height = result.height;
                next.ready = true;
                char note[256];
                std::snprintf(note, sizeof(note),
                              "%s    浮雕 %.3f    焦散质心 (%.2f, %.2f)",
                              result.method, result.relief, result.caustic_x, result.caustic_y);
                next.note = note;
            } catch (const std::exception& ex) {
                next.note = ex.what();
            }
        }
        CGImageRef target = makeImage(next.target, false);
        CGImageRef caustic = makeImage(next.caustic, false);
        CGImageRef lens = makeImage(next.height, true);
        dispatch_async(dispatch_get_main_queue(), ^{
            if (targetImage) CGImageRelease(targetImage);
            if (causticImage) CGImageRelease(causticImage);
            if (lensImage) CGImageRelease(lensImage);
            targetImage = target;
            causticImage = caustic;
            lensImage = lens;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                model_ = std::move(next);
                width_ = 1.0;
            }
            busy = NO;
            computeButton.enabled = YES;
            patternPopup.enabled = YES;
            algorithmPopup.enabled = YES;
            resolutionPopup.enabled = YES;
            focalSlider.enabled = YES;
            exportButton.enabled = model_.ready;
            statusLabel.stringValue = [NSString stringWithUTF8String:model_.note.c_str()];
            [self setNeedsDisplay:YES];
        });
    });
}

- (void)exportPressed:(id)sender {
    Eigen::MatrixXd height;
    double width = 1.0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_.ready) return;
        height = model_.height;
        width = width_;
    }
    const std::string path = "caustic_lens.obj";
    if (!writeObj(height, width, path)) {
        statusLabel.stringValue = @"导出失败";
        return;
    }
    statusLabel.stringValue = @"已写出 caustic_lens.obj";
}

- (void)drawRect:(NSRect)dirtyRect {
    [[NSColor colorWithCalibratedRed:0.043 green:0.055 blue:0.086 alpha:1] setFill];
    NSRectFill(self.bounds);

    const CGFloat margin = 18.0;
    const CGFloat gap = 16.0;
    const CGFloat top = self.bounds.size.height - 92.0;
    const CGFloat bottom = 36.0;
    const CGFloat available = std::max(self.bounds.size.width - margin * 2 - gap * 2, 0.0);
    CGFloat side = std::min(available / 3.0, std::max(top - bottom, 0.0));
    side = std::max(side, 0.0);
    const CGFloat total = side * 3 + gap * 2;
    const CGFloat startX = (self.bounds.size.width - total) / 2.0;
    const CGFloat baseY = bottom + std::max((top - bottom - side) / 2.0, 0.0);
    if (side < 40.0) return;

    CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];
    if (!ctx) return;
    struct Panel {
        NSString* title;
        NSString* hint;
        CGImageRef image;
    };
    const Panel panels[3] = {
        {@"目标图案", @"希望光在焦平面上形成的分布", targetImage},
        {@"模拟焦散", @"光线穿过透镜后落到焦平面", causticImage},
        {@"透镜高度", @"颜色越亮，表面越高", lensImage},
    };
    NSDictionary* titleAttrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:12.5 weight:NSFontWeightSemibold],
        NSForegroundColorAttributeName : [NSColor colorWithCalibratedRed:0.83 green:0.88 blue:0.96 alpha:1],
    };
    NSDictionary* hintAttrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:11],
        NSForegroundColorAttributeName : [NSColor colorWithCalibratedRed:0.45 green:0.55 blue:0.72 alpha:1],
    };
    for (int i = 0; i < 3; ++i) {
        const CGRect rect = CGRectMake(startX + i * (side + gap), baseY, side, side);
        CGContextSetRGBFillColor(ctx, 0.02f, 0.03f, 0.05f, 1.0f);
        CGContextFillRect(ctx, rect);
        if (panels[i].image) {
            CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
            CGContextDrawImage(ctx, rect, panels[i].image);
        }
        CGContextSetRGBStrokeColor(ctx, 0.28f, 0.36f, 0.52f, 0.55f);
        CGContextSetLineWidth(ctx, 1.0);
        CGContextStrokeRect(ctx, rect);
        [panels[i].title drawAtPoint:NSMakePoint(rect.origin.x, CGRectGetMaxY(rect) + 6)
                      withAttributes:titleAttrs];
        [panels[i].hint drawAtPoint:NSMakePoint(rect.origin.x, rect.origin.y - 18)
                     withAttributes:hintAttrs];
    }
}

@end

int runImage(const char* path, int resolution, caustic::Transport transport) {
    std::string error;
    Eigen::MatrixXd image = loadPhoto(path);
    if (image.size() == 0) {
        std::fprintf(stderr, "找不到 %s\n", path);
        return 1;
    }
    const caustic::Result result = caustic::design(image, makeOptions(1.5, resolution, transport));
    writePng(makeImage(result.target, false), "/tmp/caustic_target.png");
    writePng(makeImage(result.caustic, false), "/tmp/caustic_sim.png");
    writePng(makeImage(result.height, true), "/tmp/caustic_lens.png");
    std::printf("%s  分辨率 %d  浮雕 %.4f  焦散质心 (%.3f, %.3f)\n", result.method, resolution,
                result.relief, result.caustic_x, result.caustic_y);
    return 0;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--check") == 0) return runCheck();
            if (std::strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
                const int resolution = i + 2 < argc ? std::atoi(argv[i + 2]) : 160;
                const caustic::Transport transport =
                    i + 3 < argc && std::strcmp(argv[i + 3], "semi") == 0
                        ? caustic::Transport::SemiDiscrete
                        : caustic::Transport::Grid;
                return runImage(argv[i + 1], resolution > 8 ? resolution : 160, transport);
            }
        }
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        const NSRect visible = [NSScreen mainScreen].visibleFrame;
        const CGFloat width = MIN(1280.0, visible.size.width - 40.0);
        const CGFloat height = MIN(760.0, visible.size.height - 40.0);
        LensView* view = [[LensView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
        NSWindow* window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, width, height)
                                                       styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        window.title = @"焦散透镜";
        window.contentView = view;
        window.releasedWhenClosed = NO;
        window.contentMinSize = NSMakeSize(860, 520);
        [window center];
        [window makeKeyAndOrderFront:nil];
        [NSApp activate];
        [NSApp run];
    }
    return 0;
}

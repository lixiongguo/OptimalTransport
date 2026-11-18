# OptimalTransport

最优传输算法（C++ / WASM）。

顶层目录：

- `Solver`：AHT、FFT、Sinkhorn、grid、Semi；`grid_density.hpp` 和 `weighted_points.hpp` 是两种独立的分布，`ot_problem.hpp` 是共用求解入口
- `Application`：半离散窗口、焦散透镜窗口、网页和示例
- `data`：示例图像
- `geometry`、`test`、`utils`、`wasm`
- `neural_ot_viewer`：Neural OT 训练可视化网页（Hub「最优传输」里可直接打开；`./build.sh web-data` 重新训练并导出 `data.js`）

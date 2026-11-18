#!/usr/bin/env bash
# 本机窗口：半离散传输，以及用 Eigen 计算的焦散透镜。
#
#   ./build.sh gui-build   编译半离散窗口和焦散透镜
#   ./build.sh gui         编译并打开半离散窗口
#   ./build.sh check [N]   无窗口跑一次半离散传输
#   ./build.sh caustic     编译并检查焦散透镜
#   ./build.sh web-data    训练一次并导出网页实验 neural_ot_viewer/data.js
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-clang++}"
MODE="${1:-gui-build}"

flags=(-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter)
if [[ "$(uname -m)" == "arm64" ]]; then
  flags+=(-arch arm64)
fi

eigen_include() {
  if [[ -n "${EIGEN3_INCLUDE_DIR:-}" && -d "$EIGEN3_INCLUDE_DIR/Eigen" ]]; then
    echo "$EIGEN3_INCLUDE_DIR"
    return
  fi
  local candidate
  for candidate in /opt/homebrew/include/eigen3 /usr/local/include/eigen3 \
                   "$PWD/../../deps/eigen-3.4.0" "$PWD/../../deps/eigen-3.3.9" \
                   "$PWD/../libIGL_deform/build/_deps/eigen-src"; do
    if [[ -d "$candidate/Eigen" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "找不到 Eigen 头文件，可用 EIGEN3_INCLUDE_DIR 指定" >&2
  exit 1
}

build_viewer() {
  local eigen stb
  eigen="$(eigen_include)"
  stb="$PWD/../ddg-exercises/deps/polyscope/deps/stb"
  if [[ ! -f "$stb/stb_image.h" ]]; then
    echo "找不到 stb_image.h：$stb" >&2
    exit 1
  fi
  mkdir -p build/release
  echo "[build] $CXX ${flags[*]} -> build/release/ot_viewer"
  "$CXX" "${flags[@]}" -fobjc-arc -I . -I Solver -I Solver/AHT_OT -I utils -I "$eigen" -I "$stb" \
    Application/viewer/main.mm -framework Cocoa \
    -o build/release/ot_viewer
  echo "[build] done: build/release/ot_viewer"
}

build_caustic() {
  local eigen otmap
  eigen="$(eigen_include)"
  otmap="$PWD/../../Caustics/fast_caustic_design-master"
  if [[ ! -f "$otmap/build-local/libotlib.a" ]]; then
    echo "找不到 OTMap 库：$otmap/build-local/libotlib.a" >&2
    exit 1
  fi
  mkdir -p build/release
  echo "[build] $CXX ${flags[*]} -> build/release/caustic_lens"
  "$CXX" "${flags[@]}" -fobjc-arc \
    -I Solver -I Solver/SemiOT -I geometry -I "$eigen" \
    -I /opt/homebrew/include -I /opt/homebrew/include/geogram1 \
    -I "$otmap/otlib" -I "$otmap" \
    Application/caustic-lens/main.mm \
    Solver/CausticLens/caustic_transport.cpp \
    geometry/power_diagram.cpp \
    -framework Cocoa \
    "$otmap/build-local/libotlib.a" \
    -L/opt/homebrew/lib -lgeogram -lcholmod -lamd -lcolamd -lcamd -lccolamd -lmetis \
    -o build/release/caustic_lens
  echo "[build] done: build/release/caustic_lens"
}

build_examples() {
  local eigen stb imgui
  eigen="$(eigen_include)"
  stb="$PWD/../ddg-exercises/deps/polyscope/deps/stb"
  imgui="$PWD/../../Tests/DDG/TinyAD-Examples/libs/imgui-lean/src/imgui"
  if [[ ! -f "$stb/stb_image.h" ]]; then
    echo "找不到 stb_image.h：$stb" >&2
    exit 1
  fi
  if [[ ! -f "$imgui/imgui.cpp" ]]; then
    echo "找不到 ImGui：$imgui" >&2
    exit 1
  fi
  mkdir -p build/release
  echo "[build] $CXX ${flags[*]} -> build/release/ot_examples"
  "$CXX" "${flags[@]}" -DGL_SILENCE_DEPRECATION \
    -DIMGUI_IMPL_OPENGL_LOADER_CUSTOM='"gl_loader.h"' \
    -I . -I Application/imgui -I Solver -I Solver/AHT_OT -I Solver/Sinkhorn -I Solver/FFTOT \
    -I utils -I "$eigen" -I "$stb" -I "$imgui" -I /opt/homebrew/include \
    Application/imgui/ot_examples.cpp \
    utils/stb_impl.cpp \
    "$imgui/imgui.cpp" "$imgui/imgui_draw.cpp" "$imgui/imgui_widgets.cpp" \
    "$imgui/imgui_impl_glfw.cpp" "$imgui/imgui_impl_opengl3.cpp" \
    -L/opt/homebrew/lib -lglfw \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o build/release/ot_examples
  echo "[build] done: build/release/ot_examples"
}

# 生成网页实验 neural_ot_viewer/data.js（训练一次并导出轨迹）。
build_web_data() {
  local eigen
  eigen="$(eigen_include)"
  mkdir -p build/release
  echo "[build] $CXX ${flags[*]} -> build/release/neural_ot_export"
  "$CXX" "${flags[@]}" -I . -I utils -I "$eigen" \
    neural_ot_viewer/export.cpp \
    -o build/release/neural_ot_export
  echo "[build] 训练并导出 neural_ot_viewer/data.js"
  ./build/release/neural_ot_export neural_ot_viewer/data.js
}

case "$MODE" in
  gui-build)
    build_viewer
    build_caustic
    build_examples
    build_web_data
    ;;
  web-data)
    build_web_data
    ;;
  examples)
    build_examples
    ./build/release/ot_examples --check
    ;;
  gui)
    build_viewer
    exec ./build/release/ot_viewer
    ;;
  check)
    build_viewer
    ./build/release/ot_viewer --check "${2:-100}"
    ;;
  caustic)
    build_caustic
    ./build/release/caustic_lens --check
    ;;
  *)
    echo "usage: ./build.sh [gui-build|gui|check [sites]|caustic|examples|web-data]" >&2
    exit 2
    ;;
esac

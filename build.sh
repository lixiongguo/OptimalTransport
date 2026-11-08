#!/usr/bin/env bash
# 半离散最优传输的本机窗口（AppKit 绘制，无第三方依赖）。
#
#   ./build.sh gui-build   只编译 -> build/release/ot_viewer
#   ./build.sh gui         编译并打开窗口
#   ./build.sh check [N]   无窗口跑一次，写出 /tmp/ot_source.png 与 /tmp/ot_target.png
set -euo pipefail

cd "$(dirname "$0")"

CXX="${CXX:-clang++}"
MODE="${1:-gui-build}"

flags=(-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter)
if [[ "$(uname -m)" == "arm64" ]]; then
  flags+=(-arch arm64)
fi

build_viewer() {
  mkdir -p build/release
  echo "[build] $CXX ${flags[*]} -> build/release/ot_viewer"
  "$CXX" "${flags[@]}" -fobjc-arc -I . viewer/main.mm -framework Cocoa \
    -o build/release/ot_viewer
  echo "[build] done: build/release/ot_viewer"
}

case "$MODE" in
  gui-build)
    build_viewer
    ;;
  gui)
    build_viewer
    exec ./build/release/ot_viewer
    ;;
  check)
    build_viewer
    ./build/release/ot_viewer --check "${2:-100}"
    ;;
  *)
    echo "usage: ./build.sh [gui-build|gui|check [sites]]" >&2
    exit 2
    ;;
esac

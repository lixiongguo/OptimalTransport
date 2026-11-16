#!/usr/bin/env bash
# 把 Solver/AHT_OT 编译成 Application/ot-interpolation.html 用的 WASM 内核。
#
#   ./wasm/build_wasm.sh
#
# 产物：Application/assets/wasm/omt_solver.js + omt_solver.wasm
# 依赖：本机 emscripten（默认取 lixiongguo.github.io/cpp/emsdk）、Eigen 头文件。
set -euo pipefail

cd "$(dirname "$0")/.."

EMSDK_DIR="${EMSDK_DIR:-/Users/lgxgeogo/Desktop/MyDoc/lixiongguo.github.io/cpp/emsdk}"
if [[ -f "$EMSDK_DIR/emsdk_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$EMSDK_DIR/emsdk_env.sh" >/dev/null 2>&1
fi

EIGEN_INC="${EIGEN3_INCLUDE_DIR:-/opt/homebrew/include/eigen3}"
# image_ops.hpp 只用到 stb 的声明，这里不调用文件读写，因此不链接 stb 实现。
STB_INC="../ddg-exercises/deps/polyscope/deps/stb"

if [[ ! -d "$EIGEN_INC/Eigen" ]]; then
  echo "找不到 Eigen 头文件：$EIGEN_INC" >&2
  echo "可用 EIGEN3_INCLUDE_DIR=/path/to/eigen 指定" >&2
  exit 1
fi
if [[ ! -f "$STB_INC/stb_image.h" ]]; then
  echo "找不到 stb_image.h：$STB_INC" >&2
  exit 1
fi
if ! command -v em++ >/dev/null 2>&1; then
  echo "em++ 不可用，请先安装并激活 emscripten（或设置 EMSDK_DIR）" >&2
  exit 1
fi

OUT_DIR="Application/assets/wasm"
mkdir -p "$OUT_DIR"

echo "[wasm] em++ -> $OUT_DIR/omt_solver.js"
em++ -std=c++17 -O3 \
  -DEIGEN_DONT_VECTORIZE -DEIGEN_DONT_PARALLELIZE -DEIGEN_MPL2_ONLY \
  -I . -I Solver -I Solver/AHT_OT -I utils -I "$EIGEN_INC" -I "$STB_INC" \
  wasm/omt_bindings.cpp \
  -sMODULARIZE=1 -sEXPORT_NAME=OmTSolver \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 \
  -sMAXIMUM_MEMORY=2147483648 -sSTACK_SIZE=16777216 \
  -sENVIRONMENT=web,node \
  -sEXPORTED_FUNCTIONS='["_malloc","_free","_solve_omt_series","_get_omt_frame_width","_get_omt_frame_height","_get_omt_frame_count","_get_omt_result_size","_get_omt_result_data","_get_omt_last_time_ms","_get_omt_iterations","_omt_dispose"]' \
  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPU8"]' \
  -o "$OUT_DIR/omt_solver.js"

echo "[wasm] done"
ls -lh "$OUT_DIR/omt_solver.js" "$OUT_DIR/omt_solver.wasm"

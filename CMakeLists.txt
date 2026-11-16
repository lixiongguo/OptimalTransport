cmake_minimum_required(VERSION 3.16)
project(OptimalTransports LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Dependencies: Eigen3 (headers) and geogram (replaces the former CGAL backend)
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  if(DEFINED EIGEN3_INCLUDE_DIR)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
  elseif(EXISTS "${OT_ROOT}/../../deps/eigen-3.4.0/Eigen")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OT_ROOT}/../../deps/eigen-3.4.0")
  elseif(EXISTS "${OT_ROOT}/../../deps/eigen-3.3.9/Eigen")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OT_ROOT}/../../deps/eigen-3.3.9")
  else()
    message(FATAL_ERROR
      "Eigen3 not found. Install it or pass -DEIGEN3_INCLUDE_DIR=/path/to/eigen")
  endif()
endif()

# geogram: prefer its CMake package; otherwise fall back to GEOGRAM_ROOT
# (a directory containing include/ and lib/).
find_package(Geogram CONFIG QUIET)
if(TARGET Geogram::geogram)
  set(GEOGRAM_TARGET Geogram::geogram)
elseif(TARGET geogram)
  set(GEOGRAM_TARGET geogram)
elseif(DEFINED GEOGRAM_ROOT)
  add_library(geogram_ext INTERFACE)
  target_include_directories(geogram_ext INTERFACE
    "${GEOGRAM_ROOT}/include"
    "${GEOGRAM_ROOT}/include/geogram1")
  find_library(GEOGRAM_LIB NAMES geogram geogram_full
    PATHS "${GEOGRAM_ROOT}/lib" REQUIRED)
  target_link_libraries(geogram_ext INTERFACE "${GEOGRAM_LIB}")
  set(GEOGRAM_TARGET geogram_ext)
endif()

# ---------------------------------------------------------------------------
# geometry/ : geogram-backed primitives shared by the solvers (no CGAL)
# ---------------------------------------------------------------------------
if(DEFINED GEOGRAM_TARGET AND EXISTS "${OT_ROOT}/geometry/power_diagram.cpp")
  add_library(ot_geometry STATIC
    geometry/power_diagram.cpp
  )
  if(EXISTS "${OT_ROOT}/Solver/SemiOT/ma_power_diagram.cpp")
    target_sources(ot_geometry PRIVATE Solver/SemiOT/ma_power_diagram.cpp)
  endif()
  target_include_directories(ot_geometry PUBLIC
    ${OT_ROOT} ${OT_ROOT}/geometry ${OT_ROOT}/Solver ${OT_ROOT}/Solver/SemiOT)
  target_link_libraries(ot_geometry PUBLIC Eigen3::Eigen ${GEOGRAM_TARGET})
endif()

# ---------------------------------------------------------------------------
# Common input/output interface (header-only): Solver/ot_problem.hpp
# ---------------------------------------------------------------------------
add_library(ot_interface INTERFACE)
target_include_directories(ot_interface INTERFACE ${OT_ROOT} ${OT_ROOT}/Solver)
target_link_libraries(ot_interface INTERFACE Eigen3::Eigen)

if(TARGET ot_geometry AND EXISTS "${OT_ROOT}/Solver/SemiOT/ot_adapter.hpp")
  # Semi-discrete solver (header-only) + its adapter, on top of the geometry lib.
  add_library(ot_semi INTERFACE)
  target_include_directories(ot_semi INTERFACE ${OT_ROOT} ${OT_ROOT}/Solver ${OT_ROOT}/Solver/SemiOT)
  target_link_libraries(ot_semi INTERFACE ot_interface ot_geometry)
endif()

# FFT-OT grid-to-grid solver (header-only, Eigen-only).
add_library(ot_fftot INTERFACE)
target_include_directories(ot_fftot INTERFACE ${OT_ROOT} ${OT_ROOT}/Solver ${OT_ROOT}/Solver/FFTOT)
target_link_libraries(ot_fftot INTERFACE ot_interface)

# Convolutional Sinkhorn grid-to-grid solver (header-only, Eigen-only).
add_library(ot_sinkhorn INTERFACE)
target_include_directories(ot_sinkhorn INTERFACE ${OT_ROOT} ${OT_ROOT}/Solver ${OT_ROOT}/Solver/Sinkhorn)
target_link_libraries(ot_sinkhorn INTERFACE ot_interface)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(TARGET ot_geometry AND EXISTS "${OT_ROOT}/Application/examples/run_power_diagram.cpp")
  add_executable(run_power_diagram Application/examples/run_power_diagram.cpp)
  target_link_libraries(run_power_diagram PRIVATE ot_geometry)
endif()

if(TARGET ot_semi AND EXISTS "${OT_ROOT}/Application/examples/run_semi_ot.cpp")
  add_executable(run_semi_ot Application/examples/run_semi_ot.cpp)
  target_link_libraries(run_semi_ot PRIVATE ot_semi)
endif()

add_executable(run_fft_ot Application/examples/run_fft_ot.cpp)
target_link_libraries(run_fft_ot PRIVATE ot_fftot)

add_executable(run_sinkhorn Application/examples/run_sinkhorn.cpp)
target_link_libraries(run_sinkhorn PRIVATE ot_sinkhorn)

# Unified example calling the solvers through the common ot::Solver
# interface. Needs the extra backend dependencies (stb headers for AHT,
# surface_mesh + utils for gridOT), so it is opt-in.
option(OT_BUILD_ALL_SOLVERS "Build the example that drives all solvers" OFF)
if(OT_BUILD_ALL_SOLVERS AND TARGET ot_semi AND EXISTS "${OT_ROOT}/Application/examples/run_all_solvers.cpp")
  add_executable(run_all_solvers
    Application/examples/run_all_solvers.cpp
    utils/stb_impl.cpp                      # AHT image IO impl (stb)
    Solver/gridOT/otsolver_2dgrid.cpp       # grid solver
    Solver/gridOT/transport_map.cpp         # grid transport map
  )
  target_include_directories(run_all_solvers PRIVATE
    ${OT_ROOT}
    ${OT_ROOT}/Solver
    ${OT_ROOT}/Solver/AHT_OT
    ${OT_ROOT}/Solver/FFTOT
    ${OT_ROOT}/Solver/gridOT
    ${OT_ROOT}/Solver/SemiOT
    ${OT_ROOT}/utils)              # stb_image.h / stb_image_write.h live here
  target_link_libraries(run_all_solvers PRIVATE ot_semi ot_fftot)
endif()

# ---------------------------------------------------------------------------
# Layout:
#  * Solver/     AHT_OT, FFTOT, gridOT, SemiOT, and ot_problem.hpp
#  * Application viewer, web pages, and example drivers
#  * data/       images used by the viewer and the web pages
#  * geometry/ test/ utils/ wasm/
#  * AHT is Eigen-only and header-based. FFT-OT is the smallest grid demo.
#  * gridOT depends on surface_mesh; wire it once those sources are present.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15...3.30)
project(hyphi_native LANGUAGES CXX)

# Memory-safe modern C++ (C++ Core Guidelines): C++20 for std::span-style idioms;
# the 1d Forman core only needs std::vector / std::sqrt, so C++17 also compiles, but
# C++20 keeps it aligned with the frustrapy_native reference. libstdc++ assertions
# are turned on in Debug so out-of-bounds vector access traps.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)   # fall back to C++17 on older toolchains
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Locate Python and nanobind (nanobind ships its CMake config via `python -m nanobind`).
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# The CPU reference + bindings always compile. The accelerator translation units are
# appended only when their option is ON and the toolchain is actually present.
set(HYPHI_NATIVE_SOURCES src/bindings.cpp src/core.cpp)
set(HYPHI_NATIVE_DEFINES "")
set(HYPHI_NATIVE_LIBS "")

# --- Optional CUDA backend ---------------------------------------------------------
# OFF by default so a no-GPU machine (no nvcc) still builds the CPU core. When ON and
# nvcc is found, src/cuda/forman_cuda.cu compiles in and has_cuda() returns True; when
# nvcc is absent the build proceeds CPU-only with a warning (graceful, never a hard
# failure).
option(HYPHI_NATIVE_CUDA "Compile the optional CUDA kernel (requires nvcc)" OFF)
if(HYPHI_NATIVE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND HYPHI_NATIVE_SOURCES src/cuda/forman_cuda.cu)
    list(APPEND HYPHI_NATIVE_DEFINES HYPHI_NATIVE_CUDA)
    message(STATUS "hyphi_native: CUDA enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(WARNING
      "HYPHI_NATIVE_CUDA=ON but no CUDA compiler found; building CPU-only. "
      "Install the CUDA Toolkit (nvcc) and reconfigure to enable the GPU path.")
  endif()
endif()

# --- Optional Metal backend (Apple only) -------------------------------------------
# OFF by default. When ON and building on Apple, src/metal/forman_metal.cpp compiles
# in. It uses the metal-cpp single-header C++ bindings (no Objective-C++ needed, so a
# plain .cpp suffices) and embeds the MSL shader source as a string literal compiled at
# runtime with MTL::Device::newLibrary, so no separate .metallib build step is needed
# (src/metal/forman.metal is the standalone copy of that shader for reference or an
# optional offline .metallib build). Links the Metal, Foundation, and QuartzCore
# frameworks. metal-cpp headers are located via the HYPHI_METAL_CPP_DIR cache variable
# (point it at a metal-cpp checkout) or the METAL_CPP_DIR environment variable. The
# Metal path computes in float32.
option(HYPHI_NATIVE_METAL "Compile the optional Metal kernel (Apple Silicon)" OFF)
if(HYPHI_NATIVE_METAL)
  if(APPLE)
    list(APPEND HYPHI_NATIVE_SOURCES src/metal/forman_metal.cpp)
    list(APPEND HYPHI_NATIVE_DEFINES HYPHI_NATIVE_METAL)

    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
    list(APPEND HYPHI_NATIVE_LIBS
      ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

    # metal-cpp single-header bindings (https://developer.apple.com/metal/cpp/). The
    # .cpp includes <Metal/Metal.hpp>; point HYPHI_METAL_CPP_DIR at the checkout root.
    set(HYPHI_METAL_CPP_DIR "$ENV{METAL_CPP_DIR}" CACHE PATH
      "Path to a metal-cpp checkout (contains Metal/Metal.hpp)")
    if(HYPHI_METAL_CPP_DIR AND EXISTS "${HYPHI_METAL_CPP_DIR}/Metal/Metal.hpp")
      message(STATUS "hyphi_native: Metal enabled (metal-cpp at ${HYPHI_METAL_CPP_DIR})")
    else()
      message(WARNING
        "HYPHI_NATIVE_METAL=ON but metal-cpp headers were not found. Set "
        "-DHYPHI_METAL_CPP_DIR=/path/to/metal-cpp (or the METAL_CPP_DIR env var) to a "
        "checkout containing Metal/Metal.hpp. forman_metal.cpp will fail to compile "
        "without it.")
    endif()
  else()
    message(WARNING "HYPHI_NATIVE_METAL=ON but the host is not Apple; ignoring (CPU-only).")
  endif()
endif()

# --- The extension module ----------------------------------------------------------
# STABLE_ABI: build against the CPython stable ABI (one wheel per platform across
# Python minor versions). NB_STATIC links nanobind's runtime statically.
nanobind_add_module(hyphi_native STABLE_ABI NB_STATIC ${HYPHI_NATIVE_SOURCES})
target_include_directories(hyphi_native PRIVATE src)
if(HYPHI_NATIVE_DEFINES)
  target_compile_definitions(hyphi_native PRIVATE ${HYPHI_NATIVE_DEFINES})
endif()
target_compile_definitions(hyphi_native PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
if(HYPHI_NATIVE_METAL AND APPLE AND HYPHI_METAL_CPP_DIR)
  target_include_directories(hyphi_native PRIVATE "${HYPHI_METAL_CPP_DIR}")
endif()
if(HYPHI_NATIVE_LIBS)
  target_link_libraries(hyphi_native PRIVATE ${HYPHI_NATIVE_LIBS})
endif()

# Optional ASan/UBSan build for the CPU parity gate (mirrors frustrapy_native).
option(HYPHI_NATIVE_SANITIZE "Build with ASan/UBSan (for the parity gate)" OFF)
if(HYPHI_NATIVE_SANITIZE)
  target_compile_options(hyphi_native PRIVATE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(hyphi_native PRIVATE -fsanitize=address,undefined)
  target_compile_definitions(hyphi_native PRIVATE _GLIBCXX_ASSERTIONS)
endif()

# Install the extension at the top level so `import hyphi_native` works.
install(TARGETS hyphi_native LIBRARY DESTINATION .)

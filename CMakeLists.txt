cmake_minimum_required(VERSION 3.25)

project(
  linux-media-pipeline
  VERSION 0.1.0
  DESCRIPTION "Low-latency Linux GPU media pipeline"
  LANGUAGES CXX
)

option(LMP_BUILD_TESTS "Build tests" ON)
option(LMP_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(LMP_ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(LMP_ENABLE_FFMPEG "Enable in-process FFmpeg decoder when available" ON)
option(LMP_ENABLE_OPENCL "Enable OpenCL GPU filters when available" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LmpCompilerOptions)

set(LMP_HAS_FFMPEG OFF)
if(LMP_ENABLE_FFMPEG)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET
      libavformat
      libavcodec
      libavutil
      libswscale
    )
    if(FFMPEG_FOUND)
      set(LMP_HAS_FFMPEG ON)
    endif()
  endif()
endif()

set(LMP_HAS_OPENCL OFF)
if(LMP_ENABLE_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    set(LMP_HAS_OPENCL ON)
  endif()
endif()

add_library(lmp_core
  src/ai/onnx_runtime_engine.cpp
  src/ai/segmentation_mask.cpp
  src/capture/gopro_udp_source.cpp
  src/config/config_loader.cpp
  src/decoder/ffmpeg_decoder.cpp
  src/filters/background_blur_filter.cpp
  src/filters/box_blur_filter.cpp
  src/filters/brightness_filter.cpp
  src/filters/color_adjust_filter.cpp
  src/filters/contrast_filter.cpp
  src/filters/exposure_filter.cpp
  src/filters/filter_pipeline.cpp
  src/filters/filter_registry.cpp
  src/filters/fps_overlay_filter.cpp
  src/filters/gamma_filter.cpp
  src/filters/gaussian_blur_filter.cpp
  src/filters/grayscale_filter.cpp
  src/filters/histogram_filter.cpp
  src/filters/identity_filter.cpp
  src/filters/negative_filter.cpp
  src/filters/saturation_filter.cpp
  src/filters/sharpen_filter.cpp
  src/filters/sepia_filter.cpp
  src/filters/sobel_filter.cpp
  src/filters/text_overlay_filter.cpp
  src/filters/temperature_filter.cpp
  src/filters/timestamp_overlay_filter.cpp
  src/filters/tint_filter.cpp
  src/filters/white_balance_filter.cpp
  src/frame/frame.cpp
  src/gpu/buffer_pool.cpp
  src/gpu/gpu_buffer.cpp
  src/gpu/opencl_backend.cpp
  src/gpu/vulkan_backend.cpp
  src/output/v4l2_output.cpp
  src/version.cpp
)
add_library(lmp::core ALIAS lmp_core)

target_include_directories(lmp_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(lmp_core PUBLIC cxx_std_20)
target_compile_definitions(lmp_core PUBLIC
  LMP_HAS_FFMPEG=$<BOOL:${LMP_HAS_FFMPEG}>
  LMP_HAS_OPENCL=$<BOOL:${LMP_HAS_OPENCL}>
)
if(LMP_HAS_FFMPEG)
  target_link_libraries(lmp_core PUBLIC PkgConfig::FFMPEG)
endif()
if(LMP_HAS_OPENCL)
  target_link_libraries(lmp_core PUBLIC OpenCL::OpenCL)
endif()
lmp_apply_compiler_options(lmp_core)

add_executable(lmp
  src/main.cpp
)
target_link_libraries(lmp PRIVATE lmp::core)
lmp_apply_compiler_options(lmp)

include(CTest)
if(LMP_BUILD_TESTS AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(LMP_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

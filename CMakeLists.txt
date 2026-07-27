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
option(LMP_ENABLE_ONNXRUNTIME "Enable ONNX Runtime segmentation when available" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Preferred ONNX Runtime installation root")

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

set(LMP_HAS_ONNXRUNTIME OFF)
set(LMP_HAS_ONNXRUNTIME_MIGRAPHX OFF)
set(LMP_HAS_MIGRAPHX OFF)
set(LMP_HAS_ROCM_RUNTIME OFF)
set(ONNXRUNTIME_USE_PKGCONFIG OFF)
if(LMP_ENABLE_ONNXRUNTIME)
  if(ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime_cxx_api.h
      PATHS
        "${ONNXRUNTIME_ROOT}/include"
        "${ONNXRUNTIME_ROOT}/include/onnxruntime"
        "${ONNXRUNTIME_ROOT}/../include"
        /usr/include
      PATH_SUFFIXES
        onnxruntime
        onnxruntime/core/session
      NO_DEFAULT_PATH
    )
    find_library(ONNXRUNTIME_LIBRARY
      NAMES onnxruntime
      PATHS
        "${ONNXRUNTIME_ROOT}/lib64"
        "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}"
      NO_DEFAULT_PATH
    )
  endif()
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(LMP_HAS_ONNXRUNTIME ON)
  elseif(PkgConfig_FOUND)
    pkg_check_modules(ONNXRUNTIME QUIET IMPORTED_TARGET libonnxruntime)
    if(ONNXRUNTIME_FOUND)
      set(LMP_HAS_ONNXRUNTIME ON)
      set(ONNXRUNTIME_USE_PKGCONFIG ON)
    endif()
  endif()
  if(NOT LMP_HAS_ONNXRUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR
      NAMES onnxruntime_cxx_api.h
      PATH_SUFFIXES
        onnxruntime
        onnxruntime/core/session
    )
    find_library(ONNXRUNTIME_LIBRARY onnxruntime)
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
      set(LMP_HAS_ONNXRUNTIME ON)
    endif()
  endif()
endif()
find_library(ONNXRUNTIME_MIGRAPHX_PROVIDER_LIBRARY
  NAMES onnxruntime_providers_migraphx
  PATHS
    "${ONNXRUNTIME_ROOT}/lib64"
    "${ONNXRUNTIME_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}"
  PATH_SUFFIXES onnxruntime
)
if(ONNXRUNTIME_MIGRAPHX_PROVIDER_LIBRARY)
  set(LMP_HAS_ONNXRUNTIME_MIGRAPHX ON)
endif()
find_library(MIGRAPHX_LIBRARY migraphx)
if(MIGRAPHX_LIBRARY)
  set(LMP_HAS_MIGRAPHX ON)
endif()
find_library(ROCM_HSA_RUNTIME_LIBRARY hsa-runtime64)
if(ROCM_HSA_RUNTIME_LIBRARY)
  set(LMP_HAS_ROCM_RUNTIME ON)
endif()
message(STATUS "LMP_HAS_FFMPEG=${LMP_HAS_FFMPEG}")
message(STATUS "LMP_HAS_OPENCL=${LMP_HAS_OPENCL}")
message(STATUS "LMP_HAS_ONNXRUNTIME=${LMP_HAS_ONNXRUNTIME}")
message(STATUS "LMP_HAS_ONNXRUNTIME_MIGRAPHX=${LMP_HAS_ONNXRUNTIME_MIGRAPHX}")
message(STATUS "LMP_HAS_MIGRAPHX=${LMP_HAS_MIGRAPHX}")
message(STATUS "LMP_HAS_ROCM_RUNTIME=${LMP_HAS_ROCM_RUNTIME}")
message(STATUS "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
message(STATUS "ONNXRUNTIME_LIBRARY=${ONNXRUNTIME_LIBRARY}")
message(STATUS "ONNXRUNTIME_MIGRAPHX_PROVIDER_LIBRARY=${ONNXRUNTIME_MIGRAPHX_PROVIDER_LIBRARY}")

add_library(lmp_core
  src/ai/onnx_runtime_engine.cpp
  src/ai/segmentation_mask.cpp
  src/capture/gopro_udp_source.cpp
  src/config/config_loader.cpp
  src/decoder/ffmpeg_decoder.cpp
  src/filters/auto_frame_filter.cpp
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
  LMP_HAS_ONNXRUNTIME=$<BOOL:${LMP_HAS_ONNXRUNTIME}>
  LMP_HAS_ONNXRUNTIME_MIGRAPHX=$<BOOL:${LMP_HAS_ONNXRUNTIME_MIGRAPHX}>
  LMP_HAS_MIGRAPHX=$<BOOL:${LMP_HAS_MIGRAPHX}>
  LMP_HAS_ROCM_RUNTIME=$<BOOL:${LMP_HAS_ROCM_RUNTIME}>
)
if(LMP_HAS_FFMPEG)
  target_link_libraries(lmp_core PUBLIC PkgConfig::FFMPEG)
endif()
if(LMP_HAS_OPENCL)
  target_link_libraries(lmp_core PUBLIC OpenCL::OpenCL)
endif()
if(LMP_HAS_ONNXRUNTIME)
  if(ONNXRUNTIME_USE_PKGCONFIG AND TARGET PkgConfig::ONNXRUNTIME)
    target_link_libraries(lmp_core PUBLIC PkgConfig::ONNXRUNTIME)
  else()
    target_include_directories(lmp_core SYSTEM PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(lmp_core PUBLIC "${ONNXRUNTIME_LIBRARY}")
    get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    set_property(TARGET lmp_core APPEND PROPERTY
      BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
    )
  endif()
  target_link_libraries(lmp_core PUBLIC ${CMAKE_DL_LIBS})
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

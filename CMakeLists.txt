cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(MOT)

set(CMAKE_CXX_STANDARD 20)
set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -lboost_system -lboost_thread")
set(APP mot)
# set(CMAKE_PREFIX_PATH /usr/local/libtorch)
# set(CMAKE_PREFIX_PATH /usr)

option(USE_ONNX "USE_ONNX" OFF)
option(USE_MOT "USE_MOT" OFF)
if (USE_MOT)
    add_definitions(-DUSE_MOT)
endif (USE_MOT)

# CUDA
find_package(CUDA REQUIRED)
set(LIB ${LIB} ${CUDA_LIBRARIES})
set(INC ${INC} ${CUDA_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
set(LIB ${LIB} ${OpenCV_LIBRARIES})
set(INC ${INC} ${OpenCV_INCLUDE_DIRS})

# Torch
add_definitions(-DINFERENCE_ALPHAPOSE_TORCH)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS:     ${CMAKE_CXX_FLAGS}")
message(STATUS "TORCH_INCLUDE_DIRS:  ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES:     ${TORCH_LIBRARIES}")
# set(SRC ${SRC} AlphaPose.cpp)
set(LIB ${LIB} ${TORCH_LIBRARIES})
set(INC ${INC} ${TORCH_INCLUDE_DIRS})

# set(CMAKE_BUILD_TYPE Debug)

# For profiler on Ubuntu
if (CMAKE_BUILD_TYPE STREQUAL Debug AND NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif ()

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})

# Onnx
if (USE_ONNX)
    find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
    find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)
    set(INC ${INC} ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})
    set(LIB ${LIB} ${ONNX_RUNTIME_LIB})
endif (USE_ONNX)

set(SRC ${SRC} main.cpp)
set(SRC ${SRC} Yolov4TensorRT.cpp)
set(SRC ${SRC} HaTiny.cpp)
set(SRC ${SRC} Hungarian.cpp)
set(SRC ${SRC} KalmanTracker.cpp)
set(SRC ${SRC} TargetStorage.cpp)
set(SRC ${SRC} TrackerManager.cpp)

set(LIB ${LIB} ${TENSORRT_LIBRARY})
set(INC ${INC} ${TENSORRT_INCLUDE_DIR})

# GCC need to link against stdc++fs
if(MSVC)
set(STDCXXFS "")
else()
set(LIB ${LIB} "stdc++fs")
endif()

include_directories(${INC})

add_executable(${APP} ${SRC})
target_link_libraries(${APP} ${LIB})



# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set(NN_LINUX_TESTBENCHES
  fft fft_cpu gesture_recognition_fp32 kyber_nouv_rvv resnet50
  sentence_inference_fp32 sentence_inference_int8 sentence_gemmini)
if(NOT NN_TESTBENCH IN_LIST NN_LINUX_TESTBENCHES)
  message(FATAL_ERROR
    "Unsupported Linux NN_TESTBENCH='${NN_TESTBENCH}'. Supported: ${NN_LINUX_TESTBENCHES}")
endif()
if(NOT NN_LINK_MODE MATCHES "^(static|dynamic)$")
  message(FATAL_ERROR "NN_LINK_MODE must be static or dynamic")
endif()
if(NN_TESTBENCH STREQUAL "fft" AND NN_BACKEND STREQUAL "cpu")
  message(FATAL_ERROR "Use the independent fft_cpu testbench for the CPU implementation")
endif()
if(NN_TESTBENCH STREQUAL "fft_cpu" AND NOT NN_BACKEND STREQUAL "cpu")
  message(FATAL_ERROR "The fft_cpu testbench requires NN_BACKEND=cpu")
endif()
if(NN_BACKEND STREQUAL "cpu" AND NOT NN_CONFIG STREQUAL "default")
  message(STATUS "CPU backend uses '${NN_CONFIG}'; auto-vectorization remains disabled")
endif()

nn_linux_profile("${NN_CONFIG}" NN_ARCH NN_TUNE)
target_compile_options(riscv_nn_ops PRIVATE -march=${NN_ARCH} -mabi=lp64d)
if(NN_TUNE)
  target_compile_options(riscv_nn_ops PRIVATE -mtune=${NN_TUNE})
endif()

if(NN_BACKEND STREQUAL "gemmini")
  if(NOT NN_TESTBENCH STREQUAL "sentence_gemmini")
    message(FATAL_ERROR
      "NN_BACKEND=gemmini currently supports NN_TESTBENCH=sentence_gemmini")
  endif()
  if(NOT NN_CONFIG STREQUAL "default")
    message(FATAL_ERROR "Gemmini Spike/pk must use NN_CONFIG=default (RV64GC)")
  endif()

  add_executable(sentence_gemmini
    "${CMAKE_CURRENT_SOURCE_DIR}/testbench/sentence_gemmini/sentence_gemmini.c"
    $<TARGET_OBJECTS:riscv_nn_gemmini_ops>)
  target_include_directories(sentence_gemmini PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/testbench/sentence_gemmini"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops"
    "${CMAKE_CURRENT_SOURCE_DIR}/header"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops/bareMetalC")
  target_include_directories(sentence_gemmini SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops")
  target_compile_definitions(sentence_gemmini PRIVATE
    SPIKE_PK=1 PREALLOCATE=1 MULTITHREAD=1 PRINT_TILE=0
    NN_BACKEND_CPU=0 NN_BACKEND_VECTOR=0 NN_BACKEND_GEMMINI=1)
  target_compile_options(sentence_gemmini PRIVATE
    -O3 -march=${NN_ARCH} -mabi=lp64d
    -fno-tree-vectorize -fno-tree-slp-vectorize)
  target_link_options(sentence_gemmini PRIVATE -march=${NN_ARCH} -mabi=lp64d -static)
  target_link_libraries(sentence_gemmini PRIVATE m)
  set_target_properties(sentence_gemmini PROPERTIES
    C_STANDARD 99 C_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}/build/linux-pk/sentence_gemmini/${NN_CONFIG}/gemmini/static")

  find_program(NN_SPIKE_EXECUTABLE spike)
  find_program(NN_PK_EXECUTABLE pk
    HINTS "$ENV{RISCV}/riscv64-unknown-elf/bin")
  if(NN_SPIKE_EXECUTABLE AND NN_PK_EXECUTABLE)
    add_custom_target(run-spike-gemmini
      COMMAND "${NN_SPIKE_EXECUTABLE}"
        --isa=rv64gc_zicntr_zihpm --extension=gemmini
        "${NN_PK_EXECUTABLE}"
        "$<TARGET_FILE:sentence_gemmini>"
      DEPENDS sentence_gemmini USES_TERMINAL VERBATIM)
  endif()
  return()
endif()

if(NN_TESTBENCH STREQUAL "gesture_recognition_fp32")
  set(NN_TESTBENCH_SOURCE testbench/gesture_model/gesture_recognition_fp32.c)
elseif(NN_TESTBENCH STREQUAL "kyber_nouv_rvv")
  set(NN_TESTBENCH_SOURCE testbench/kyber/kyber_nouv_rvv.c)
elseif(NN_TESTBENCH STREQUAL "resnet50")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/testbench/resnet50/resnet50_weights.h")
    message(FATAL_ERROR
      "resnet50 requires generated testbench/resnet50/resnet50_weights.h; run the export/generation tools under py/resnet50 first")
  endif()
  set(NN_TESTBENCH_SOURCE testbench/resnet50/resnet50.c)
else()
  set(NN_TESTBENCH_SOURCE testbench/${NN_TESTBENCH}/${NN_TESTBENCH}.c)
endif()

add_executable(${NN_TESTBENCH}
  "${CMAKE_CURRENT_SOURCE_DIR}/${NN_TESTBENCH_SOURCE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn_runtime_linux.c"
  $<TARGET_OBJECTS:riscv_nn_ops>)
target_include_directories(${NN_TESTBENCH} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/header"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_SOURCE_DIR}/testbench"
  "${CMAKE_CURRENT_SOURCE_DIR}/testbench/gesture_model"
  "${CMAKE_CURRENT_SOURCE_DIR}/testbench/kyber"
  "${CMAKE_CURRENT_SOURCE_DIR}/testbench/resnet50"
  "${CMAKE_CURRENT_SOURCE_DIR}/testbench/sentence_inference_fp32"
  "${CMAKE_CURRENT_SOURCE_DIR}/testbench/sentence_inference_int8")
target_compile_definitions(${NN_TESTBENCH} PRIVATE ${NN_BACKEND_DEFINES})
target_compile_options(${NN_TESTBENCH} PRIVATE -O3 -Wall -Wextra -march=${NN_ARCH} -mabi=lp64d)
if(NN_TUNE)
  target_compile_options(${NN_TESTBENCH} PRIVATE -mtune=${NN_TUNE})
endif()
if(NOT NN_AUTO_VECTORIZE)
  target_compile_options(${NN_TESTBENCH} PRIVATE
    -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin)
endif()
target_link_options(${NN_TESTBENCH} PRIVATE -march=${NN_ARCH} -mabi=lp64d)
if(NN_LINK_MODE STREQUAL "static")
  target_link_options(${NN_TESTBENCH} PRIVATE -static)
endif()
target_link_libraries(${NN_TESTBENCH} PRIVATE m -Wl,-u,_printf_float c)
set_target_properties(${NN_TESTBENCH} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/build/linux-pk/${NN_TESTBENCH}/${NN_CONFIG}/${NN_BACKEND}/${NN_LINK_MODE}")

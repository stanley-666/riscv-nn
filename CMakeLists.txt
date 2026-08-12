# SPDX-FileContributor: Person: Stanley Lee
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

set(NN_PLATFORM "linux-pk" CACHE STRING "Runtime platform: linux-pk or baremetal")
set_property(CACHE NN_PLATFORM PROPERTY STRINGS linux-pk baremetal)
set(NN_BACKEND "vector" CACHE STRING "Operator backend: cpu, vector, or all")
set_property(CACHE NN_BACKEND PROPERTY STRINGS cpu vector all)
set(NN_CONFIG "zvl128b" CACHE STRING "Linux ISA profile")
set(NN_TESTBENCH "sentence_inference_fp32" CACHE STRING "Testbench/application name")
set(NN_LINK_MODE "static" CACHE STRING "Linux link mode: static or dynamic")
set_property(CACHE NN_LINK_MODE PROPERTY STRINGS static dynamic)
option(NN_AUTO_VECTORIZE "Allow compiler-generated vector instructions" OFF)
option(NN_ENABLE_GEMMINI "Build the common Gemmini operator adapter" OFF)

if(NN_PLATFORM STREQUAL "baremetal")
  project(riscv_nn LANGUAGES C ASM)
else()
  project(riscv_nn LANGUAGES C)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT NN_PLATFORM MATCHES "^(linux-pk|baremetal)$")
  message(FATAL_ERROR "NN_PLATFORM must be linux-pk or baremetal")
endif()
if(NOT NN_BACKEND MATCHES "^(cpu|vector|gemmini|all)$")
  message(FATAL_ERROR "NN_BACKEND must be cpu, vector, gemmini, or all")
endif()

if(NN_BACKEND STREQUAL "gemmini")
  set(NN_ENABLE_GEMMINI ON CACHE BOOL "Build the common Gemmini operator adapter" FORCE)
endif()

file(GLOB NN_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.c")
list(REMOVE_ITEM NN_CORE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/main.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn_runtime_linux.c")
file(GLOB_RECURSE NN_CPU_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/cpu/ops/*.c")
file(GLOB_RECURSE NN_VECTOR_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/vector/ops/*.c")

if(NN_BACKEND STREQUAL "cpu")
  list(REMOVE_ITEM NN_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn_infer_vpu.c")
  set(NN_BACKEND_SOURCES ${NN_CPU_SOURCES})
  set(NN_BACKEND_DEFINES NN_BACKEND_CPU=1 NN_BACKEND_VECTOR=0)
elseif(NN_BACKEND STREQUAL "vector")
  list(REMOVE_ITEM NN_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn_infer_cpu.c")
  set(NN_BACKEND_SOURCES ${NN_VECTOR_SOURCES})
  set(NN_BACKEND_DEFINES NN_BACKEND_CPU=0 NN_BACKEND_VECTOR=1)
elseif(NN_BACKEND STREQUAL "all")
  set(NN_BACKEND_SOURCES ${NN_CPU_SOURCES} ${NN_VECTOR_SOURCES})
  set(NN_BACKEND_DEFINES NN_BACKEND_CPU=1 NN_BACKEND_VECTOR=1)
else()
  list(REMOVE_ITEM NN_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn_infer_cpu.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nn_infer_vpu.c")
  set(NN_BACKEND_SOURCES "")
  set(NN_BACKEND_DEFINES
    NN_BACKEND_CPU=0 NN_BACKEND_VECTOR=0 NN_BACKEND_GEMMINI=1)
endif()

add_library(riscv_nn_ops OBJECT ${NN_CORE_SOURCES} ${NN_BACKEND_SOURCES})
target_include_directories(riscv_nn_ops PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/header"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
target_compile_definitions(riscv_nn_ops PUBLIC ${NN_BACKEND_DEFINES})
target_compile_options(riscv_nn_ops PRIVATE -O3 -Wall -Wextra)
if(NOT NN_AUTO_VECTORIZE)
  target_compile_options(riscv_nn_ops PRIVATE
    -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin)
endif()

if(NN_ENABLE_GEMMINI)
  add_library(riscv_nn_gemmini_ops OBJECT
    csrc/backends/riscv/gemmini/ops/nn_ops_gemmini.c)
  target_include_directories(riscv_nn_gemmini_ops PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/header"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
  target_include_directories(riscv_nn_gemmini_ops SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops")
  target_compile_options(riscv_nn_gemmini_ops PRIVATE
    -O3 -Wall -Wextra -march=rv64gc -mabi=lp64d)
  if(NN_PLATFORM STREQUAL "baremetal")
    target_compile_definitions(riscv_nn_gemmini_ops PRIVATE BAREMETAL=1 NDEBUG=1)
    target_compile_options(riscv_nn_gemmini_ops PRIVATE
      -mcmodel=medany -ffreestanding -msmall-data-limit=0
      -fno-common -fno-builtin-printf
      -fno-tree-loop-distribute-patterns -nostdlib -nostartfiles)
  endif()
  set_target_properties(riscv_nn_gemmini_ops PROPERTIES
    C_STANDARD 11 C_EXTENSIONS ON)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RiscvProfiles.cmake")
if(NN_PLATFORM STREQUAL "linux-pk")
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/LinuxPk.cmake")
else()
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Baremetal.cmake")
endif()

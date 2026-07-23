# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set(NN_HARDWARE_CONFIG "V128D128B" CACHE STRING "Bare-metal hardware profile")
set_property(CACHE NN_HARDWARE_CONFIG PROPERTY STRINGS cpu V128D128B V256D128B V512D128B GEMMINI)
set(NN_BAREMETAL_CPU_HZ "50000000UL" CACHE STRING "Bare-metal CPU clock")
set(NN_SDCARD_DEVICE "" CACHE STRING "Whole SD-card block device")
set(NN_SDCARD_BLOCK "34" CACHE STRING "SD-card destination block")
set(NN_FLASH_CONFIRM "NO" CACHE STRING "Set YES to authorize SD-card write")

if(NN_BACKEND STREQUAL "gemmini")
  if(NOT NN_TESTBENCH MATCHES "^(sentence_gemmini|fft_batched_int8_gemmini)$")
    message(FATAL_ERROR
      "Bare-metal Gemmini supports sentence_gemmini and fft_batched_int8_gemmini")
  endif()
  if(NOT NN_HARDWARE_CONFIG STREQUAL "GEMMINI")
    message(FATAL_ERROR
      "Bare-metal NN_BACKEND=gemmini requires NN_HARDWARE_CONFIG=GEMMINI")
  endif()

  file(GLOB NN_GEMMINI_MINILIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/minilib/*.c")
  set(NN_GEMMINI_TARGET "${NN_TESTBENCH}_baremetal")
  set(NN_GEMMINI_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/build/baremetal/${NN_TESTBENCH}")
  if(NN_TESTBENCH STREQUAL "sentence_gemmini")
    set(NN_GEMMINI_BASENAME "GEMMINI_nn_gemmini_baremetal")
  else()
    set(NN_GEMMINI_BASENAME "GEMMINI_${NN_TESTBENCH}_baremetal")
  endif()
  set(NN_GEMMINI_ELF "${NN_GEMMINI_OUTPUT_DIR}/${NN_GEMMINI_BASENAME}.elf")
  set(NN_GEMMINI_BIN "${NN_GEMMINI_OUTPUT_DIR}/${NN_GEMMINI_BASENAME}.bin")
  set(NN_GEMMINI_DUMP "${NN_GEMMINI_OUTPUT_DIR}/${NN_GEMMINI_BASENAME}.dump")
  set(NN_GEMMINI_MAP "${NN_GEMMINI_OUTPUT_DIR}/${NN_GEMMINI_BASENAME}.map")

  add_executable(${NN_GEMMINI_TARGET}
    testbench/${NN_TESTBENCH}/${NN_TESTBENCH}.c
    baremetal/syscalls.c
    ${NN_GEMMINI_MINILIB_SOURCES}
    baremetal/trap.c
    baremetal/start.S
    $<TARGET_OBJECTS:riscv_nn_gemmini_ops>)
  target_include_directories(${NN_GEMMINI_TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/testbench/${NN_TESTBENCH}"
    "${CMAKE_CURRENT_SOURCE_DIR}/testbench/fft_batched_int8"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops"
    "${CMAKE_CURRENT_SOURCE_DIR}/header"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops/bareMetalC")
  target_include_directories(${NN_GEMMINI_TARGET} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/riscv/gemmini/ops")
  target_compile_definitions(${NN_GEMMINI_TARGET} PRIVATE
    BAREMETAL=1 PREALLOCATE=1 MULTITHREAD=1 PRINT_TILE=0
    BAREMETAL_CPU_HZ=${NN_BAREMETAL_CPU_HZ}
    NN_BACKEND_CPU=0 NN_BACKEND_VECTOR=0 NN_BACKEND_GEMMINI=1)
  target_compile_options(${NN_GEMMINI_TARGET} PRIVATE
    -O3 -ffast-math -march=rv64gc -mabi=lp64d -mcmodel=medany
    -ffreestanding -msmall-data-limit=0 -fno-common
    -fno-builtin-printf -fno-tree-loop-distribute-patterns
    -fno-tree-vectorize -fno-tree-slp-vectorize -nostdlib -nostartfiles)
  target_link_options(${NN_GEMMINI_TARGET} PRIVATE
    -march=rv64gc -mabi=lp64d -mcmodel=medany -nostdlib -nostartfiles -static
    -T "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/linker.ld"
    -Wl,--gc-sections "-Wl,-Map,${NN_GEMMINI_MAP}")
  target_link_libraries(${NN_GEMMINI_TARGET} PRIVATE
    -Wl,--start-group c m gcc -Wl,--end-group)
  set_target_properties(${NN_GEMMINI_TARGET} PROPERTIES
    C_STANDARD 99 C_EXTENSIONS ON
    OUTPUT_NAME "${NN_GEMMINI_BASENAME}" SUFFIX ".elf"
    RUNTIME_OUTPUT_DIRECTORY "${NN_GEMMINI_OUTPUT_DIR}")

  add_custom_command(OUTPUT "${NN_GEMMINI_BIN}"
    COMMAND "${CMAKE_OBJCOPY}" -O binary "${NN_GEMMINI_ELF}" "${NN_GEMMINI_BIN}"
    DEPENDS ${NN_GEMMINI_TARGET} VERBATIM)
  add_custom_target(baremetal-bin ALL DEPENDS "${NN_GEMMINI_BIN}")
  add_custom_command(OUTPUT "${NN_GEMMINI_DUMP}"
    COMMAND "${CMAKE_COMMAND}"
      "-DOBJDUMP=${CMAKE_OBJDUMP}" "-DELF=${NN_GEMMINI_ELF}"
      "-DOUTPUT=${NN_GEMMINI_DUMP}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteObjdump.cmake"
    DEPENDS ${NN_GEMMINI_TARGET} VERBATIM)
  add_custom_target(baremetal-dump DEPENDS "${NN_GEMMINI_DUMP}")
  add_custom_target(baremetal-flash
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/flash_baremetal.sh"
      "${NN_GEMMINI_BIN}" "${NN_SDCARD_DEVICE}"
      "${NN_SDCARD_BLOCK}" "${NN_FLASH_CONFIRM}"
    DEPENDS baremetal-bin USES_TERMINAL VERBATIM)
  return()
elseif(NN_BACKEND STREQUAL "cpu")
  if(NOT NN_TESTBENCH MATCHES "^fft_cpu(_int8)?$")
    message(FATAL_ERROR "Bare-metal CPU backend currently supports fft_cpu and fft_cpu_int8")
  endif()
  if(NOT NN_HARDWARE_CONFIG STREQUAL "cpu")
    message(FATAL_ERROR "Bare-metal CPU FFT requires NN_HARDWARE_CONFIG=cpu")
  endif()
elseif(NOT NN_BACKEND STREQUAL "vector")
  message(FATAL_ERROR "Bare-metal adapters currently require NN_BACKEND=cpu, vector, or gemmini")
endif()
nn_baremetal_profile("${NN_HARDWARE_CONFIG}" NN_ARCH)

if(NN_TESTBENCH STREQUAL "sentence_inference_fp32")
  set(NN_APP_DIR baremetal/apps/sentence_fp32)
  set(NN_MODEL_DIR testbench/sentence_inference_fp32)
elseif(NN_TESTBENCH STREQUAL "sentence_inference_int8")
  set(NN_APP_DIR baremetal/apps/sentence_int8)
  set(NN_MODEL_DIR testbench/sentence_inference_int8)
elseif(NN_TESTBENCH STREQUAL "gesture_model")
  set(NN_APP_DIR baremetal/apps/gesture_fp32)
  set(NN_MODEL_DIR testbench/gesture_model)
elseif(NN_TESTBENCH STREQUAL "kyber")
  set(NN_APP_DIR baremetal/apps/kyber)
  set(NN_MODEL_DIR testbench/kyber)
  set(NN_APP_EXTRA_SOURCE testbench/kyber/kyber_nouv_rvv.c)
elseif(NN_TESTBENCH STREQUAL "fft")
  set(NN_APP_DIR baremetal/apps/fft)
  set(NN_MODEL_DIR testbench/fft)
  set(NN_APP_EXTRA_SOURCE testbench/fft/fft.c)
elseif(NN_TESTBENCH STREQUAL "fft_gemv")
  set(NN_APP_DIR baremetal/apps/fft_gemv)
  set(NN_MODEL_DIR testbench/fft_gemv)
  set(NN_APP_EXTRA_SOURCE testbench/fft_gemv/fft_gemv.c)
elseif(NN_TESTBENCH STREQUAL "fft_batched")
  set(NN_APP_DIR baremetal/apps/fft_batched)
  set(NN_MODEL_DIR testbench/fft_batched)
  set(NN_APP_EXTRA_SOURCE testbench/fft_batched/fft_batched.c)
elseif(NN_TESTBENCH STREQUAL "fft_batched_int8")
  set(NN_APP_DIR baremetal/apps/fft_batched_int8)
  set(NN_MODEL_DIR testbench/fft_batched_int8)
  set(NN_APP_EXTRA_SOURCE testbench/fft_batched_int8/fft_batched_int8.c)
elseif(NN_TESTBENCH STREQUAL "fft_cpu")
  set(NN_APP_DIR baremetal/apps/fft_cpu)
  set(NN_MODEL_DIR testbench/fft_cpu)
  set(NN_APP_EXTRA_SOURCE testbench/fft_cpu/fft_cpu.c)
elseif(NN_TESTBENCH STREQUAL "fft_cpu_int8")
  set(NN_APP_DIR baremetal/apps/fft_cpu_int8)
  set(NN_MODEL_DIR testbench/fft_cpu_int8)
  set(NN_APP_EXTRA_SOURCE testbench/fft_cpu_int8/fft_cpu_int8.c)
else()
  message(FATAL_ERROR
    "Unsupported bare-metal NN_TESTBENCH='${NN_TESTBENCH}'. Supported: sentence_inference_fp32;sentence_inference_int8;gesture_model;kyber;fft;fft_batched;fft_batched_int8;fft_cpu;fft_cpu_int8;fft_gemv")
endif()

file(GLOB NN_APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${NN_APP_DIR}/*.c")
file(GLOB NN_MINILIB_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/minilib/*.c")
set(NN_BAREMETAL_SOURCES
  ${NN_APP_SOURCES}
  ${NN_APP_EXTRA_SOURCE}
  baremetal/main.c
  baremetal/nn_runtime_baremetal.c
  baremetal/syscalls.c
  ${NN_MINILIB_SOURCES}
  baremetal/trap.c
  baremetal/start.S)

set(NN_BAREMETAL_STANDALONE_TESTBENCHES fft fft_batched fft_batched_int8 fft_cpu fft_cpu_int8 fft_gemv)
if(NN_TESTBENCH IN_LIST NN_BAREMETAL_STANDALONE_TESTBENCHES)
  set_target_properties(riscv_nn_ops PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(NN_BACKEND STREQUAL "cpu")
  set(NN_BAREMETAL_ENABLE_VECTOR 0)
  set(NN_BAREMETAL_BASENAME "${NN_HARDWARE_CONFIG}_nn_cpu_baremetal")
else()
  set(NN_BAREMETAL_ENABLE_VECTOR 1)
  set(NN_BAREMETAL_BASENAME "${NN_HARDWARE_CONFIG}_nn_rvv_baremetal")
endif()
set(NN_BAREMETAL_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/baremetal/${NN_TESTBENCH}")
if(NN_TESTBENCH IN_LIST NN_BAREMETAL_STANDALONE_TESTBENCHES)
  add_executable(${NN_TESTBENCH}_baremetal ${NN_BAREMETAL_SOURCES})
else()
  add_executable(${NN_TESTBENCH}_baremetal
    ${NN_BAREMETAL_SOURCES} $<TARGET_OBJECTS:riscv_nn_ops>)
endif()
target_include_directories(${NN_TESTBENCH}_baremetal PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/header"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/${NN_APP_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/${NN_MODEL_DIR}")
target_compile_definitions(${NN_TESTBENCH}_baremetal PRIVATE
  ${NN_BACKEND_DEFINES} BAREMETAL ENABLE_VECTOR=${NN_BAREMETAL_ENABLE_VECTOR}
  BAREMETAL_CPU_HZ=${NN_BAREMETAL_CPU_HZ}
  BUILD_DATE="cmake" BUILD_TIME="ninja")
set(NN_BAREMETAL_FLAGS
  -O3 -march=${NN_ARCH} -mabi=lp64d -mtune=rocket
  -msmall-data-limit=0 -Wall -Wextra -ffreestanding -nostdlib
  -nostartfiles -mcmodel=medany)
target_include_directories(riscv_nn_ops PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/include")
target_compile_definitions(riscv_nn_ops PRIVATE
  BAREMETAL ENABLE_VECTOR=1 BAREMETAL_CPU_HZ=${NN_BAREMETAL_CPU_HZ})
target_compile_options(riscv_nn_ops PRIVATE ${NN_BAREMETAL_FLAGS})
target_compile_options(${NN_TESTBENCH}_baremetal PRIVATE ${NN_BAREMETAL_FLAGS})
if(NOT NN_AUTO_VECTORIZE)
  target_compile_options(${NN_TESTBENCH}_baremetal PRIVATE
    -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin)
endif()
set(NN_MAP_FILE "${NN_BAREMETAL_OUTPUT_DIR}/${NN_BAREMETAL_BASENAME}.map")
target_link_options(${NN_TESTBENCH}_baremetal PRIVATE
  ${NN_BAREMETAL_FLAGS}
  -T "${CMAKE_CURRENT_SOURCE_DIR}/baremetal/linker.ld"
  -Wl,--gc-sections "-Wl,-Map,${NN_MAP_FILE}")
target_link_libraries(${NN_TESTBENCH}_baremetal PRIVATE
  -Wl,--start-group c m gcc -Wl,--end-group)
set_target_properties(${NN_TESTBENCH}_baremetal PROPERTIES
  OUTPUT_NAME "${NN_BAREMETAL_BASENAME}"
  SUFFIX ".elf"
  RUNTIME_OUTPUT_DIRECTORY "${NN_BAREMETAL_OUTPUT_DIR}")

set(NN_ELF "${NN_BAREMETAL_OUTPUT_DIR}/${NN_BAREMETAL_BASENAME}.elf")
set(NN_BIN "${NN_BAREMETAL_OUTPUT_DIR}/${NN_BAREMETAL_BASENAME}.bin")
set(NN_DUMP "${NN_BAREMETAL_OUTPUT_DIR}/${NN_BAREMETAL_BASENAME}.dump")
add_custom_command(OUTPUT "${NN_BIN}"
  COMMAND "${CMAKE_OBJCOPY}" -O binary "${NN_ELF}" "${NN_BIN}"
  DEPENDS ${NN_TESTBENCH}_baremetal VERBATIM)
add_custom_target(baremetal-bin ALL DEPENDS "${NN_BIN}")
add_custom_command(OUTPUT "${NN_DUMP}"
  COMMAND "${CMAKE_COMMAND}"
    "-DOBJDUMP=${CMAKE_OBJDUMP}" "-DELF=${NN_ELF}" "-DOUTPUT=${NN_DUMP}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteObjdump.cmake"
  DEPENDS ${NN_TESTBENCH}_baremetal VERBATIM)
add_custom_target(baremetal-dump DEPENDS "${NN_DUMP}")

add_custom_target(baremetal-flash
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/flash_baremetal.sh"
    "${NN_BIN}" "${NN_SDCARD_DEVICE}" "${NN_SDCARD_BLOCK}" "${NN_FLASH_CONFIRM}"
  DEPENDS baremetal-bin USES_TERMINAL VERBATIM)

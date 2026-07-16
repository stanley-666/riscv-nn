# Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
# SPDX-License-Identifier: Apache-2.0

# Canonical list of platform-independent neural-network sources. Platform
# Makefiles add exactly one nn_runtime implementation to this list. Operator
# implementations are grouped by architecture and execution backend.
NN_CORE_SRCS := $(filter-out \
    $(NN_COMMON_SRC_DIR)/main.c \
    $(NN_COMMON_SRC_DIR)/nn_runtime_linux.c, \
    $(wildcard $(NN_COMMON_SRC_DIR)/*.c))
NN_RISCV_CPU_SRCS := $(sort \
    $(wildcard $(NN_COMMON_SRC_DIR)/backends/riscv/cpu/ops/*.c) \
    $(wildcard $(NN_COMMON_SRC_DIR)/backends/riscv/cpu/ops/*/*.c))
NN_RISCV_VECTOR_SRCS := $(sort \
    $(wildcard $(NN_COMMON_SRC_DIR)/backends/riscv/vector/ops/*.c) \
    $(wildcard $(NN_COMMON_SRC_DIR)/backends/riscv/vector/ops/*/*.c))

BACKEND ?= all
VALID_BACKENDS := cpu vector all
ifeq ($(filter $(BACKEND),$(VALID_BACKENDS)),)
  $(error Unsupported BACKEND='$(BACKEND)'. Supported: $(VALID_BACKENDS))
endif

ifeq ($(BACKEND),cpu)
  NN_CORE_SRCS := $(filter-out $(NN_COMMON_SRC_DIR)/nn_infer_vpu.c,$(NN_CORE_SRCS))
  NN_BACKEND_SRCS := $(NN_RISCV_CPU_SRCS)
  NN_BACKEND_CPPFLAGS := -DNN_BACKEND_CPU=1 -DNN_BACKEND_VECTOR=0
else ifeq ($(BACKEND),vector)
  NN_CORE_SRCS := $(filter-out $(NN_COMMON_SRC_DIR)/nn_infer_cpu.c,$(NN_CORE_SRCS))
  NN_BACKEND_SRCS := $(NN_RISCV_VECTOR_SRCS)
  NN_BACKEND_CPPFLAGS := -DNN_BACKEND_CPU=0 -DNN_BACKEND_VECTOR=1
else
  NN_BACKEND_SRCS := $(NN_RISCV_CPU_SRCS) $(NN_RISCV_VECTOR_SRCS)
  NN_BACKEND_CPPFLAGS := -DNN_BACKEND_CPU=1 -DNN_BACKEND_VECTOR=1
endif

NN_COMMON_SRCS := $(NN_CORE_SRCS) $(NN_BACKEND_SRCS)

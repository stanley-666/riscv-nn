# RISC-V Linux Cross Compiler
CROSS_COMPILE ?= riscv64-unknown-linux-gnu-
CC      := $(CROSS_COMPILE)gcc
OBJDUMP := $(CROSS_COMPILE)objdump
OBJCOPY := $(CROSS_COMPILE)objcopy
TB_INC  := $(shell find testbench -type d)
INCLUDE := -I./header -I./csrc $(addprefix -I,$(TB_INC))

# riscv64-unknown-elf-objdump -d benchmark_rv64gcv | less


# Default profile
CONFIG ?= zvl512b_cycle
# Source & Target
NN_COMMON_SRC_DIR := csrc
include make/common.mk
CORE_SRC := $(NN_COMMON_SRCS) csrc/nn_runtime_linux.c
TB_SRCS  := $(shell find testbench -name '*.c')
TB_BINS  := $(notdir $(TB_SRCS:.c=))
LINUX_BUILD_DIR := build/linux-pk
OBJ_DIR  := $(LINUX_BUILD_DIR)/$(CONFIG)/$(BACKEND)/obj
CORE_OBJS := $(patsubst csrc/%.c,$(OBJ_DIR)/csrc/%.o,$(CORE_SRC))

# Architecture and ABI settings
ifeq ($(CONFIG), default)
  ARCH := rv64gc_zicntr_zihpm
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl64b)
  ARCH := rv64gcv_zicntr_zihpm_zvbb_zvl64b_zve64d
  ABI  := lp64d
  TUNE := 
endif

ifeq ($(CONFIG), zvl128b)
  ARCH := rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl256b)
  ARCH := rv64gcv_zicntr_zihpm_zvbb_zvl256b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl512b)
  ARCH := rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl512b_cycle)
  ARCH := rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), RVV)
  ARCH := rv64imafdcbvzicsr_zifencei_zicntr_zihpm_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), MINV64D64RocketGENESYS2Config)
  ARCH := rv64imafdcbzicsr_zifencei_zicntr_zihpm_zvl64b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
  ABI  := lp64d
  TUNE := rocket 
endif

ifeq ($(CONFIG), DSPV128D128RocketGENESYS2Config)
  ARCH := rv64imafdcbvzicsr_zifencei_zicntr_zihpm_zvl128b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
  ABI  := lp64d
  TUNE := rocket
endif

# RISC-V gnu Compiler flags
CFLAGS  := -O3 -march=$(ARCH) -mabi=$(ABI) -Wall -Wextra -std=c11 $(INCLUDE)
CFLAGS  += $(NN_BACKEND_CPPFLAGS)
# RVV is emitted only by explicit intrinsics unless a caller opts in to GCC
# auto-vectorization. This keeps scalar/reference kernels valid as baselines.
AUTO_VECTORIZE ?= 0
ifeq ($(AUTO_VECTORIZE),0)
  CFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin
endif
# Linker flags for Linux dynamic and pk/static builds
DYNAMIC_LDFLAGS :=
STATIC_LDFLAGS  := -static
# Keep the legacy target behavior configurable.
LINK_MODE ?= static
ifeq ($(LINK_MODE),static)
  LDFLAGS := $(STATIC_LDFLAGS)
else
  LDFLAGS := $(DYNAMIC_LDFLAGS)
endif
# Libraries
LIBS    := -lm -u _printf_float -lc


ifneq ($(strip $(TUNE)),)
  CFLAGS += -mtune=$(TUNE)
endif

all: $(TB_BINS)

TESTBENCH ?= sentence_inference_fp32
LINUX_TESTBENCHES := gesture_recognition_fp32 kyber_nouv_rvv resnet50 \
                    sentence_inference_fp32 sentence_inference_fp32_time \
                    sentence_inference_int8

linux:
	@if ! echo " $(LINUX_TESTBENCHES) " | grep -q " $(TESTBENCH) "; then \
		echo "Unsupported Linux/pk TESTBENCH='$(TESTBENCH)'"; \
		echo "Supported: $(LINUX_TESTBENCHES)"; \
		exit 2; \
	fi
	$(MAKE) $(TESTBENCH) CONFIG=$(CONFIG) LINK_MODE=$(LINK_MODE) BACKEND=$(BACKEND)

baremetal:
	$(MAKE) -C baremetal vector

baremetal-elf:
	$(MAKE) -C baremetal elf

baremetal-dump:
	$(MAKE) -C baremetal dump

baremetal-flash:
	$(MAKE) -C baremetal flash

define TB_template
$(notdir $(1:.c=)): $(LINUX_BUILD_DIR)/$(notdir $(1:.c=))/$(CONFIG)/$(BACKEND)/$(LINK_MODE)/$(notdir $(1:.c=))

$(notdir $(1:.c=))_dynamic: $(LINUX_BUILD_DIR)/$(notdir $(1:.c=))/$(CONFIG)/$(BACKEND)/dynamic/$(notdir $(1:.c=))

$(LINUX_BUILD_DIR)/$(notdir $(1:.c=))/$(CONFIG)/$(BACKEND)/dynamic/$(notdir $(1:.c=)): $(patsubst testbench/%.c,$(OBJ_DIR)/testbench/%.o,$(1)) $(CORE_OBJS)
	@mkdir -p $$(@D)
	$(CC) $(CFLAGS) $(DYNAMIC_LDFLAGS) -o $$@ $$^ ${LIBS}

$(notdir $(1:.c=))_static: $(LINUX_BUILD_DIR)/$(notdir $(1:.c=))/$(CONFIG)/$(BACKEND)/static/$(notdir $(1:.c=))

$(LINUX_BUILD_DIR)/$(notdir $(1:.c=))/$(CONFIG)/$(BACKEND)/static/$(notdir $(1:.c=)): $(patsubst testbench/%.c,$(OBJ_DIR)/testbench/%.o,$(1)) $(CORE_OBJS)
	@mkdir -p $$(@D)
	$(CC) $(CFLAGS) $(STATIC_LDFLAGS) -o $$@ $$^ ${LIBS}
endef
$(foreach src,$(TB_SRCS),$(eval $(call TB_template,$(src))))

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

LINUX_OUTPUTS := $(foreach name,$(TB_BINS),$(LINUX_BUILD_DIR)/$(name)/$(CONFIG)/$(BACKEND)/$(LINK_MODE)/$(name))

dump: $(addsuffix .dump,$(LINUX_OUTPUTS))

bin: $(addsuffix .bin,$(LINUX_OUTPUTS))

%.dump: %
	$(OBJDUMP) -d $< > $@

%.bin: %
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f $(TB_BINS) $(addsuffix .dump,$(TB_BINS)) $(addsuffix .bin,$(TB_BINS))
	rm -rf build

.PHONY: all clean dump bin linux baremetal baremetal-elf baremetal-dump baremetal-flash

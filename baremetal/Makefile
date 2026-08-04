ROOT_DIR          := $(abspath $(CURDIR)/..)
SRC_DIR           := $(ROOT_DIR)/csrc
HEADER_DIR        := $(ROOT_DIR)/header
BACKEND           ?= vector
NN_COMMON_SRC_DIR := $(SRC_DIR)
include $(ROOT_DIR)/make/common.mk

TESTBENCH        ?= sentence_inference_fp32
ifeq ($(TESTBENCH),sentence_inference_fp32)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/sentence_inference_fp32/sentence_inference_fp32.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/sentence_inference_fp32
else ifeq ($(TESTBENCH),sentence_inference_int8)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/sentence_inference_int8/sentence_inference_int8.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/sentence_inference_int8
else ifeq ($(TESTBENCH),gesture_model)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/gesture_model/gesture_recognition_fp32.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/gesture_model
else ifeq ($(TESTBENCH),kyber)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/kyber/kyber_nouv_rvv.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/kyber
else ifeq ($(TESTBENCH),fft)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/fft/fft.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft
else ifeq ($(TESTBENCH),fft_gemv)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/fft_gemv/fft_gemv.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft_gemv
else ifeq ($(TESTBENCH),fft_batched)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/fft_batched/fft_batched.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft_batched
else ifeq ($(TESTBENCH),fft_batched_int8)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/fft_batched_int8/fft_batched_int8.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft_batched_int8
else ifeq ($(TESTBENCH),fft_cpu)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/fft_cpu/fft_cpu.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft_cpu
else ifeq ($(TESTBENCH),fft_cpu_int8)
  TESTBENCH_SRC          := $(ROOT_DIR)/testbench/fft_cpu_int8/fft_cpu_int8.c
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft_cpu_int8
else
  $(error No bare-metal testbench source for TESTBENCH='$(TESTBENCH)'. Supported: sentence_inference_fp32 sentence_inference_int8 gesture_model kyber fft fft_batched fft_batched_int8 fft_cpu fft_cpu_int8 fft_gemv)
endif

MODEL_DATA_DIR   ?= $(DEFAULT_MODEL_DATA_DIR)

CROSS_COMPILE    ?= /opt/riscv_baremetal_medany/bin/riscv64-unknown-elf-
CC               := $(CROSS_COMPILE)gcc
OBJCOPY          := $(CROSS_COMPILE)objcopy
OBJDUMP          := $(CROSS_COMPILE)objdump

HARDWARE_CONFIG  ?= V128D128B
BAREMETAL_CPU_HZ ?= 50000000UL
ABI              ?= lp64d

ifeq ($(HARDWARE_CONFIG),V512D128B)
  ARCH := rv64gcv_zvl512b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
endif
ifeq ($(HARDWARE_CONFIG),V256D128B)
  ARCH := rv64gcv_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
endif
ifeq ($(HARDWARE_CONFIG),V128D128B)
  ARCH := rv64gcv_zvl128b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
endif
ifeq ($(HARDWARE_CONFIG),cpu)
  ARCH := rv64gc_zicntr_zihpm
endif
ARCH             ?= rv64gcv

ifeq ($(BACKEND),cpu)
  BAREMETAL_ENABLE_VECTOR := 0
  TARGET_SUFFIX := nn_cpu_baremetal
else
  BAREMETAL_ENABLE_VECTOR := 1
  TARGET_SUFFIX := nn_rvv_baremetal
endif

BUILD_DIR        ?= $(ROOT_DIR)/build/baremetal
TARGET           ?= $(BUILD_DIR)/$(TESTBENCH)/$(HARDWARE_CONFIG)_$(TARGET_SUFFIX)
OBJ_DIR          := $(BUILD_DIR)/$(TESTBENCH)/$(HARDWARE_CONFIG)/obj

INCLUDES         := -I$(HEADER_DIR) -I$(SRC_DIR) -I$(CURDIR)/include \
                    -I$(MODEL_DATA_DIR)
COMMON_CFLAGS    := -O3 -march=$(ARCH) -mabi=$(ABI) -mtune=rocket \
                    -msmall-data-limit=0 -Wall -Wextra -std=c11 \
                    -DBAREMETAL -DENABLE_VECTOR=$(BAREMETAL_ENABLE_VECTOR) -ffreestanding \
                    -nostdlib -nostartfiles -mcmodel=medany \
                    -DBAREMETAL_CPU_HZ=$(BAREMETAL_CPU_HZ) \
                    -DNN_TESTBENCH_NAME=\"$(TESTBENCH)\" \
                    -DNN_HARDWARE_CONFIG_NAME=\"$(HARDWARE_CONFIG)\" \
                    -DBUILD_DATE=\"$(shell date +%Y%m%d)\" \
                    -DBUILD_TIME=\"$(shell date +%H%M%S)\" $(INCLUDES)

# Match the CMake helper policy while allowing an explicit command-line
# override for comparison builds.
ifeq ($(BACKEND),vector)
  AUTO_VECTORIZE ?= 1
else
  AUTO_VECTORIZE ?= 0
endif
ifeq ($(AUTO_VECTORIZE),0)
  COMMON_CFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin
endif

COMMON_CFLAGS    += $(NN_BACKEND_CPPFLAGS)
CORE_SRCS        := $(NN_COMMON_SRCS)
MINILIB_SRCS     := $(sort $(wildcard $(CURDIR)/minilib/*.c))
RUNTIME_SRCS     := $(CURDIR)/nn_runtime_baremetal.c $(CURDIR)/platform_info.c $(CURDIR)/syscalls.c \
                    $(MINILIB_SRCS) $(CURDIR)/trap.c
STARTUP          := $(CURDIR)/start.S
ALL_C_SRCS       := $(CORE_SRCS) $(TESTBENCH_SRC) $(RUNTIME_SRCS)
OBJS             := $(patsubst $(ROOT_DIR)/%.c,$(OBJ_DIR)/%.o,$(ALL_C_SRCS))
STARTUP_OBJ      := $(OBJ_DIR)/baremetal/start.o
DEPS             := $(OBJS:.o=.d)
LINKER           := $(CURDIR)/linker.ld
LDFLAGS          := -T $(LINKER) -Wl,--gc-sections -Wl,-Map,$(TARGET).map
LIBS             := -Wl,--start-group -lc -lm -lgcc -Wl,--end-group

SDCARD_DEVICE    ?=
SDCARD_BLOCK     ?= 34
FLASH_CONFIRM    ?= NO

.PHONY: all vector elf bin dump flash clean

all: vector
vector: $(TARGET).bin
elf: $(TARGET).elf
bin: $(TARGET).bin
dump: $(TARGET).dump

flash:
	@if [ -z "$(SDCARD_DEVICE)" ]; then \
		echo "SDCARD_DEVICE is required (example: /dev/sdb)."; \
		exit 2; \
	fi
	@if [ ! -b "$(SDCARD_DEVICE)" ]; then \
		echo "Not a block device: $(SDCARD_DEVICE)"; \
		exit 2; \
	fi
	@if [ "$$(lsblk -dnro TYPE "$(SDCARD_DEVICE)")" != "disk" ]; then \
		echo "Refusing to flash a partition; select the whole SD card disk."; \
		exit 2; \
	fi
	@if lsblk -nrpo MOUNTPOINT "$(SDCARD_DEVICE)" | grep -q '[^[:space:]]'; then \
		echo "Refusing to flash: $(SDCARD_DEVICE) or one of its partitions is mounted."; \
		lsblk -o NAME,TYPE,SIZE,MOUNTPOINT "$(SDCARD_DEVICE)"; \
		exit 2; \
	fi
	@if [ "$(FLASH_CONFIRM)" != "YES" ]; then \
		echo "Flash check passed for $(SDCARD_DEVICE)."; \
		lsblk -o NAME,MODEL,TYPE,SIZE,MOUNTPOINT "$(SDCARD_DEVICE)"; \
		echo "Re-run with FLASH_CONFIRM=YES to write $(TARGET).bin at block $(SDCARD_BLOCK)."; \
		exit 2; \
	fi
	$(MAKE) $(TARGET).bin
	@echo "Writing $(TARGET).bin to $(SDCARD_DEVICE), 512-byte block $(SDCARD_BLOCK)..."
	sudo dd if=$(TARGET).bin of=$(SDCARD_DEVICE) bs=512 seek=$(SDCARD_BLOCK) \
		conv=notrunc,fsync status=progress
	@echo "Flash completed."

$(TARGET).elf: $(STARTUP_OBJ) $(OBJS) $(LINKER)
	@mkdir -p $(@D)
	$(CC) $(COMMON_CFLAGS) $(STARTUP_OBJ) $(OBJS) $(LDFLAGS) $(LIBS) -o $@

$(OBJ_DIR)/%.o: $(ROOT_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(COMMON_CFLAGS) -MMD -MP -c $< -o $@

$(STARTUP_OBJ): $(STARTUP)
	@mkdir -p $(@D)
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

$(TARGET).dump: $(TARGET).elf
	$(OBJDUMP) -d -S $< > $@

clean:
	rm -f $(TARGET).elf $(TARGET).map $(TARGET).bin $(TARGET).dump
	rm -rf $(OBJ_DIR)

-include $(DEPS)

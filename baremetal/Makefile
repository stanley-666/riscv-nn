ROOT_DIR          := $(abspath $(CURDIR)/..)
SRC_DIR           := $(ROOT_DIR)/csrc
HEADER_DIR        := $(ROOT_DIR)/header

TESTBENCH        ?= sentence_inference_fp32
ifeq ($(TESTBENCH),sentence_inference_fp32)
  DEFAULT_APP_DIR        := $(CURDIR)/apps/sentence_fp32
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/sentence_inference_fp32
else ifeq ($(TESTBENCH),sentence_inference_int8)
  DEFAULT_APP_DIR        := $(CURDIR)/apps/sentence_int8
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/sentence_inference_int8
else ifeq ($(TESTBENCH),gesture_model)
  DEFAULT_APP_DIR        := $(CURDIR)/apps/gesture_fp32
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/gesture_model
else ifeq ($(TESTBENCH),kyber)
  DEFAULT_APP_DIR        := $(CURDIR)/apps/kyber
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/kyber
  DEFAULT_APP_EXTRA_SRCS := $(ROOT_DIR)/testbench/kyber/kyber_nouv_rvv.c
else ifeq ($(TESTBENCH),fft)
  DEFAULT_APP_DIR        := $(CURDIR)/apps/fft
  DEFAULT_MODEL_DATA_DIR := $(ROOT_DIR)/testbench/fft
  DEFAULT_APP_EXTRA_SRCS := $(ROOT_DIR)/testbench/fft/fft.c
else
  $(error No bare-metal adapter for TESTBENCH='$(TESTBENCH)'. Supported: sentence_inference_fp32 sentence_inference_int8 gesture_model kyber fft)
endif

APP_DIR          ?= $(DEFAULT_APP_DIR)
MODEL_DATA_DIR   ?= $(DEFAULT_MODEL_DATA_DIR)
APP_EXTRA_SRCS   ?= $(DEFAULT_APP_EXTRA_SRCS)

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
ARCH             ?= rv64gcv

BUILD_DIR        ?= $(ROOT_DIR)/build/baremetal
TARGET           ?= $(BUILD_DIR)/$(TESTBENCH)/$(HARDWARE_CONFIG)_nn_rvv_baremetal
OBJ_DIR          := $(BUILD_DIR)/$(TESTBENCH)/$(HARDWARE_CONFIG)/obj

INCLUDES         := -I$(HEADER_DIR) -I$(SRC_DIR) -I$(CURDIR)/include \
                    -I$(APP_DIR) -I$(MODEL_DATA_DIR)
COMMON_CFLAGS    := -O3 -march=$(ARCH) -mabi=$(ABI) -mtune=rocket \
                    -msmall-data-limit=0 -Wall -Wextra -std=c11 \
                    -DBAREMETAL -DENABLE_VECTOR=1 -ffreestanding \
                    -nostdlib -nostartfiles -mcmodel=medany \
                    -DBAREMETAL_CPU_HZ=$(BAREMETAL_CPU_HZ) \
                    -DBUILD_DATE=\"$(shell date +%Y%m%d)\" \
                    -DBUILD_TIME=\"$(shell date +%H%M%S)\" $(INCLUDES)

# Explicit RVV intrinsics remain enabled; only compiler-generated
# auto-vectorization of scalar/reference code is disabled by default.
AUTO_VECTORIZE   ?= 0
ifeq ($(AUTO_VECTORIZE),0)
  COMMON_CFLAGS += -fno-tree-vectorize -fno-tree-slp-vectorize -fno-builtin
endif

NN_COMMON_SRC_DIR := $(SRC_DIR)
include $(ROOT_DIR)/make/common.mk
COMMON_CFLAGS    += $(NN_BACKEND_CPPFLAGS)
CORE_SRCS        := $(NN_COMMON_SRCS)
APP_SRCS         := $(wildcard $(APP_DIR)/*.c) $(APP_EXTRA_SRCS)
MINILIB_SRCS     := $(sort $(wildcard $(CURDIR)/minilib/*.c))
RUNTIME_SRCS     := $(CURDIR)/main.c $(CURDIR)/nn_runtime_baremetal.c $(CURDIR)/syscalls.c \
                    $(MINILIB_SRCS) $(CURDIR)/trap.c
STARTUP          := $(CURDIR)/start.S
ALL_C_SRCS       := $(CORE_SRCS) $(APP_SRCS) $(RUNTIME_SRCS)
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

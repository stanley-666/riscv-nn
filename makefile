# RISC-V Linux Cross Compiler
CROSS_COMPILE ?= riscv64-unknown-linux-gnu-
CC      := $(CROSS_COMPILE)gcc
OBJDUMP := $(CROSS_COMPILE)objdump
OBJCOPY := $(CROSS_COMPILE)objcopy
MAKEFLAGS += -j8

TB_INC  := $(shell find testbench -type d)
INCLUDE := -I./header -I./csrc $(addprefix -I,$(TB_INC))

# riscv64-unknown-elf-objdump -d benchmark_rv64gcv | less


# Default profile
CONFIG ?= zvl512b
# Source & Target
CORE_SRC := $(filter-out csrc/main.c,$(wildcard csrc/*.c))
TB_SRCS  := $(shell find testbench -name '*.c')
TB_BINS  := $(notdir $(TB_SRCS:.c=))
OBJ_DIR  := build
CORE_OBJS := $(patsubst csrc/%.c,$(OBJ_DIR)/csrc/%.o,$(CORE_SRC))

# Architecture and ABI settings
ifeq ($(CONFIG), default)
  ARCH := rv64gc
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl64b)
  ARCH := rv64gcv_zvbb_zvl64b_zve64d
  ABI  := lp64d
  TUNE := 
endif

ifeq ($(CONFIG), zvl128b)
  ARCH := rv64gcv_zvbb_zvl128b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl256b)
  ARCH := rv64gcv_zvbb_zvl256b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), zvl512b)
  ARCH := rv64gcv_zvbb_zvl512b_zve64d
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), RVV)
  ARCH := rv64imafdcv_zicsr_zifencei_zihpm_zaamo_zalrsc_zfh_zca_zcd_zba_zbb_zbs_zvbb_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvkb
  ABI  := lp64d
  TUNE := rocket
endif

ifeq ($(CONFIG), MINV64D64RocketGENESYS2Config)
  ARCH := rv64imafdcbzicsr_zifencei_zihpm_zvl64b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
  ABI  := lp64d
  TUNE := rocket 
endif

ifeq ($(CONFIG), DSPV128D128RocketGENESYS2Config)
  ARCH := rv64imafdcbvzicsr_zifencei_zihpm_zvl128b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb
  ABI  := lp64d
  TUNE := rocket
endif

# RISC-V gnu Compiler flags
CFLAGS  := -O3 -march=$(ARCH) -mabi=$(ABI) -Wall -Wextra -std=c11 $(INCLUDE)
# Static linking
LDFLAGS := -static
# Libraries
LIBS    := -lm -u _printf_float -lc


ifneq ($(strip $(TUNE)),)
  CFLAGS += -mtune=$(TUNE)
endif

all: $(TB_BINS)

define TB_template
$(notdir $(1:.c=)): $(patsubst testbench/%.c,$(OBJ_DIR)/testbench/%.o,$(1)) $(CORE_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $$@ $$^ ${LIBS}
endef
$(foreach src,$(TB_SRCS),$(eval $(call TB_template,$(src))))

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

dump: $(addsuffix .dump,$(TB_BINS))

bin: $(addsuffix .bin,$(TB_BINS))

%.dump: %
	$(OBJDUMP) -d $< > $@

%.bin: %
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f $(TB_BINS) $(addsuffix .dump,$(TB_BINS)) $(addsuffix .bin,$(TB_BINS))
	rm -rf $(OBJ_DIR)

.PHONY: all clean dump bin

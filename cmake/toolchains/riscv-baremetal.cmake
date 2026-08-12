# SPDX-FileContributor: Person: Stanley Lee
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(RISCV_BAREMETAL_PREFIX "/opt/riscv_baremetal_medany/bin/riscv64-unknown-elf-"
  CACHE STRING "Bare-metal cross-tool prefix")
set(CMAKE_C_COMPILER "${RISCV_BAREMETAL_PREFIX}gcc")
set(CMAKE_ASM_COMPILER "${RISCV_BAREMETAL_PREFIX}gcc")
set(CMAKE_OBJCOPY "${RISCV_BAREMETAL_PREFIX}objcopy" CACHE FILEPATH "objcopy")
set(CMAKE_OBJDUMP "${RISCV_BAREMETAL_PREFIX}objdump" CACHE FILEPATH "objdump")

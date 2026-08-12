# SPDX-FileContributor: Person: Stanley Lee
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(RISCV_LINUX_PREFIX "riscv64-unknown-linux-gnu-" CACHE STRING "Linux cross-tool prefix")
set(CMAKE_C_COMPILER "${RISCV_LINUX_PREFIX}gcc")
set(CMAKE_OBJCOPY "${RISCV_LINUX_PREFIX}objcopy" CACHE FILEPATH "objcopy")
set(CMAKE_OBJDUMP "${RISCV_LINUX_PREFIX}objdump" CACHE FILEPATH "objdump")

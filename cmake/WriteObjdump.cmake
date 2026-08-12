# SPDX-FileContributor: Person: Stanley Lee
# Copyright 2026 Stanley Lee
# SPDX-License-Identifier: Apache-2.0

execute_process(
  COMMAND "${OBJDUMP}" -d -S "${ELF}"
  OUTPUT_FILE "${OUTPUT}"
  RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  file(REMOVE "${OUTPUT}")
  message(FATAL_ERROR "objdump failed with status ${result}")
endif()

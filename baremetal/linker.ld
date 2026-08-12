/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
/* Linker script tailored for Genesys2 (Rocket RV64) memory map.
 * Adjust MEMORY region if targeting a different platform. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x40000000
}

SECTIONS
{
  .text : {
    KEEP(*(.text.start))
    *(.text .text.*)
    *(.rodata .rodata.*)
  } > RAM

  .data : {
    __data_start = .;
    *(.data .data.*)
    __data_end = .;
  } > RAM

  .sdata : {
    __sdata_start = .;
    *(.srodata .srodata.*)
    *(.sdata .sdata.*)
    __sdata_end = .;
  } > RAM

  .sbss (NOLOAD) : {
    __sbss_start = .;
    *(.sbss .sbss.*)
    __sbss_end = .;
  } > RAM

  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    __bss_end = .;
  } > RAM

  PROVIDE(_heap_start = __bss_end);
  PROVIDE(_heap_end = ORIGIN(RAM) + LENGTH(RAM) - 0x06400000); /* reserve 100MB for stack */
  PROVIDE(_stack_top = ORIGIN(RAM) + LENGTH(RAM));
  PROVIDE(__global_pointer$ = (__sdata_start + __sbss_end) / 2);
}

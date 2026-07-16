.section .text.start
.globl _start
.option norvc

.extern __bss_start
.extern __bss_end
.extern __sbss_start
.extern __sbss_end
.extern __global_pointer$
.extern _stack_top
.extern main
.extern trap_handler

_start:
    la sp, _stack_top
    la gp, __global_pointer$
    la t0, trap_vector
    csrw mtvec, t0

    # Enable floating-point (FS) state (vector state enabled later in C when needed)
    csrr t0, mstatus
    li   t1, (3 << 13)            # FS = 0b11 (dirty)
    or   t0, t0, t1
    csrw mstatus, t0

    # Clear .sbss section
    la t0, __sbss_start
    la t1, __sbss_end
1:
    bgeu t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b
2:
    la t0, __bss_start
    la t1, __bss_end
3:
    bgeu t0, t1, 4f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 3b
4:
    call main
5:
    wfi
    j 5b

trap_vector:
    addi sp, sp, -32
    sd ra, 0(sp)
    csrr a0, mcause
    csrr a1, mepc
    call trap_handler
    csrw mepc, a0
    ld ra, 0(sp)
    addi sp, sp, 32
    mret

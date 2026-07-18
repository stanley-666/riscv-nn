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

    # Vector floating-point instructions require both FS and VS to be non-Off.
    # Enable FP state before entering C because printf/libm may also use the FPU.
    csrr t0, mstatus
    li   t1, (3 << 13)            # mstatus.FS[14:13] = Dirty
    or   t0, t0, t1
#if defined(__riscv_vector)
    # Enable architectural vector state. Any vector instruction or vector CSR
    # access traps as illegal when mstatus.VS is Off.
    li   t1, (3 << 9)             # mstatus.VS[10:9] = Dirty
    or   t0, t0, t1
#endif
    csrw mstatus, t0

#if defined(__riscv_vector)
    # RVV 1.0 permits vstart, vxrm, and vxsat to contain arbitrary reset values.
    # Clear vstart so the first vector instruction begins at element zero.
    csrw vstart, zero

    # vcsr[2:1]=vxrm and vcsr[0]=vxsat. Zero selects RNU fixed-point rounding
    # and clears the accrued saturation flag used by integer NN kernels.
    csrw vcsr, zero

    # Establish a legal initial vtype/vl state. Each intrinsic later issues the
    # vsetvl required for its own SEW and LMUL; this also resets vstart to zero.
    vsetvli zero, zero, e8, m1, ta, ma
#endif

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
    # This runtime has no context-switching trap path. Preserve only enough
    # state to report the fault, then stop instead of returning with a partial
    # integer/FP/vector context.
    addi sp, sp, -32
    sd ra, 0(sp)
    csrr a0, mcause
    csrr a1, mepc
    call trap_handler
6:
    wfi
    j 6b

#include <stdint.h>
#include <stdio.h>

uint64_t trap_handler(uint64_t mcause, uint64_t mepc) {
    uint64_t mstatus, mtval;
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    __asm__ volatile("csrr %0, mtval" : "=r"(mtval));

    printf("\n[trap] mcause=0x%lx mepc=0x%lx mstatus=0x%lx mtval=0x%lx\n",
           (unsigned long)mcause,
           (unsigned long)mepc,
           (unsigned long)mstatus,
           (unsigned long)mtval);

    return mepc;
}

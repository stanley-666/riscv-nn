/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#include <stdint.h>
#include <stdio.h>

_Noreturn void trap_handler(uint64_t mcause, uint64_t mepc) {
    uint64_t mstatus, mtval;
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    __asm__ volatile("csrr %0, mtval" : "=r"(mtval));

    printf("\n[trap] mcause=0x%lx mepc=0x%lx mstatus=0x%lx mtval=0x%lx\n",
           (unsigned long)mcause,
           (unsigned long)mepc,
           (unsigned long)mstatus,
           (unsigned long)mtval);

    printf("[trap] fatal: context restore is not implemented; system halted.\n");
    for (;;) {
        __asm__ volatile("wfi");
    }
}

#include <stdint.h>
#include <stdio.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#ifndef BAREMETAL_CPU_HZ
#define BAREMETAL_CPU_HZ 50000000UL
#endif

#ifndef NN_TESTBENCH_NAME
#define NN_TESTBENCH_NAME "unknown"
#endif

#ifndef NN_HARDWARE_CONFIG_NAME
#define NN_HARDWARE_CONFIG_NAME "unknown"
#endif

static inline uint64_t read_csr_misa(void)
{
    uint64_t value;
    __asm__ volatile("csrr %0, misa" : "=r"(value));
    return value;
}

static inline uint64_t read_csr_mvendorid(void)
{
    uint64_t value;
    __asm__ volatile("csrr %0, mvendorid" : "=r"(value));
    return value;
}

static inline uint64_t read_csr_marchid(void)
{
    uint64_t value;
    __asm__ volatile("csrr %0, marchid" : "=r"(value));
    return value;
}

static inline uint64_t read_csr_mimpid(void)
{
    uint64_t value;
    __asm__ volatile("csrr %0, mimpid" : "=r"(value));
    return value;
}

static inline uint64_t read_csr_mhartid(void)
{
    uint64_t value;
    __asm__ volatile("csrr %0, mhartid" : "=r"(value));
    return value;
}

static void format_misa(uint64_t misa, char *buffer, size_t capacity)
{
    static const char canonical_order[] = "iemafdqlcbjtpvn";
    size_t used = 0;
    unsigned xlen = 0;
    const unsigned mxl = (unsigned)(misa >> 62);

    if (mxl == 1)
        xlen = 32;
    else if (mxl == 2)
        xlen = 64;
    else if (mxl == 3)
        xlen = 128;
#if defined(__riscv_xlen)
    else
        xlen = __riscv_xlen;
#endif

    if (capacity == 0)
        return;

    if (used + 2 < capacity) {
        buffer[used++] = 'r';
        buffer[used++] = 'v';
    }
    if (xlen == 32 && used + 2 < capacity) {
        buffer[used++] = '3';
        buffer[used++] = '2';
    } else if (xlen == 64 && used + 2 < capacity) {
        buffer[used++] = '6';
        buffer[used++] = '4';
    } else if (xlen == 128 && used + 3 < capacity) {
        buffer[used++] = '1';
        buffer[used++] = '2';
        buffer[used++] = '8';
    }

    for (size_t i = 0; canonical_order[i] != '\0'; ++i) {
        const unsigned bit = (unsigned)(canonical_order[i] - 'a');
        if ((misa & (UINT64_C(1) << bit)) && used + 1 < capacity)
            buffer[used++] = canonical_order[i];
    }
    buffer[used] = '\0';
}

void baremetal_print_hardware_info(void)
{
    const uint64_t misa = read_csr_misa();
    const uint64_t bitmanip_bit = UINT64_C(1) << ('B' - 'A');
    const uint64_t hypervisor_bit = UINT64_C(1) << ('H' - 'A');
    const uint64_t supervisor_bit = UINT64_C(1) << ('S' - 'A');
    const uint64_t user_bit = UINT64_C(1) << ('U' - 'A');
    char isa_string[32];
    format_misa(misa, isa_string, sizeof(isa_string));

    printf("\n=== Bare-metal hardware information ===\n");
    printf("  testbench            : %s\n", NN_TESTBENCH_NAME);
    printf("  build profile        : %s\n", NN_HARDWARE_CONFIG_NAME);
    printf("  configured CPU clock : %lu Hz\n", (unsigned long)BAREMETAL_CPU_HZ);
    printf("  validation policy    : report only; continue\n");
    printf("  misa                 : 0x%016lx\n", (unsigned long)misa);
    printf("  ISA extensions       : %s\n", isa_string);
    if (misa & bitmanip_bit) {
        printf("  bitmanip B bundle    : present (Zba/Zbb/Zbs)\n");
    } else {
#if defined(__riscv_zba) || defined(__riscv_zbb) || defined(__riscv_zbs)
        printf("  bitmanip B bundle    : not advertised by misa\n");
        printf("  Zba/Zbb/Zbs check    : instruction probes required\n");
#endif
    }
#if defined(__riscv_zvbb)
    printf("  Zvbb check           : instruction probe required\n");
#endif
    printf("  privilege modes      : m%s%s\n",
           (misa & supervisor_bit) ? "/s" : "",
           (misa & user_bit) ? "/u" : "");
    if (misa & hypervisor_bit)
        printf("  hypervisor extension : present\n");
    printf("  mvendorid            : 0x%016lx\n", (unsigned long)read_csr_mvendorid());
    printf("  marchid              : 0x%016lx\n", (unsigned long)read_csr_marchid());
    printf("  mimpid               : 0x%016lx\n", (unsigned long)read_csr_mimpid());
    printf("  mhartid              : %lu\n", (unsigned long)read_csr_mhartid());
#if defined(__riscv_xlen)
    printf("  XLEN                 : %d\n", __riscv_xlen);
#endif
#if defined(__riscv_flen)
    printf("  FLEN                 : %d\n", __riscv_flen);
#endif
#if defined(__riscv_vector)
    const uint64_t vector_bit = UINT64_C(1) << ('V' - 'A');
    if (misa & vector_bit) {
        const size_t vlen_bytes = __riscv_vlenb();
        printf("  RVV                  : present\n");
        printf("  VLEN                 : %lu bits\n",
               (unsigned long)(vlen_bytes * 8));
    }
#endif
    printf("=======================================\n\n");
}

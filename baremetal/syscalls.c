#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <unistd.h>

#include "platform.h"

extern char _heap_start;
extern char _heap_end;

static inline void uart_init_once(void) {
    static bool initialized = false;
    if (initialized)
        return;
    volatile uint32_t *txctrl = (uint32_t *)(UART_CTRL_ADDR + UART_REG_TXCTRL);
    *txctrl = UART_TXEN;
    initialized = true;
}

static inline void uart_putc(uint8_t ch) {
    volatile uint32_t *txfifo = (uint32_t *)(UART_CTRL_ADDR + UART_REG_TXFIFO);
    while ((int32_t)(*txfifo) < 0)
        ;
    *txfifo = ch;
}

void *_sbrk(ptrdiff_t incr) {
    static char *current = &_heap_start;
    char *prev = current;
    if ((current + incr) < &_heap_start || (current + incr) > &_heap_end) {
        errno = ENOMEM;
        return (void *)-1;
    }
    current += incr;
    return prev;
}

int _write(int fd, const void *buf, size_t len) {
    if (fd != STDOUT_FILENO && fd != STDERR_FILENO)
        return len;

    uart_init_once();
    const uint8_t *data = (const uint8_t *)buf;
    for (size_t i = 0; i < len; i++) {
        if (data[i] == '\n')
            uart_putc('\r');
        uart_putc(data[i]);
    }
    return (int)len;
}

int _close(int fd) {
    (void)fd;
    return 0;
}

int _fstat(int fd, struct stat *st) {
    (void)fd;
    if (!st) {
        errno = EFAULT;
        return -1;
    }
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int fd) {
    return (fd == STDIN_FILENO || fd == STDOUT_FILENO || fd == STDERR_FILENO);
}

off_t _lseek(int fd, off_t offset, int whence) {
    (void)fd;
    (void)offset;
    (void)whence;
    errno = ESPIPE;
    return -1;
}

int _read(int fd, void *buf, size_t len) {
    (void)fd;
    (void)buf;
    (void)len;
    return 0;
}

int _kill(int pid, int sig) {
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

int _getpid(void) {
    return 1;
}

void _exit(int status) {
    (void)status;
    while (1) {
        __asm__ volatile("wfi");
    }
}

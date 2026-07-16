/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
*/

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "platform.h"

static void console_init_once(void) {
    static bool initialized = false;
    if (initialized)
        return;
    volatile uint32_t *txctrl = (uint32_t *)(UART_CTRL_ADDR + UART_REG_TXCTRL);
    *txctrl = UART_TXEN;
    initialized = true;
}

static inline void console_raw_putc(uint8_t ch) {
    volatile uint32_t *txfifo = (uint32_t *)(UART_CTRL_ADDR + UART_REG_TXFIFO);
    while ((int32_t)(*txfifo) < 0)
        ;
    *txfifo = ch;
}

static void console_putc(char ch, int *count) {
    console_init_once();
    if (ch == '\n')
        console_raw_putc('\r');
    console_raw_putc((uint8_t)ch);
    if (count)
        (*count)++;
}

static void console_puts(const char *s, int *count) {
    if (!s)
        s = "(null)";
    while (*s)
        console_putc(*s++, count);
}

static bool is_digit(char ch) {
    return (ch >= '0' && ch <= '9');
}

static void print_unsigned(uint64_t value, unsigned base, bool uppercase, int width, bool zero_pad, int *count) {
    char buffer[32];
    int idx = 0;
    if (base < 2)
        base = 10;
    do {
        unsigned digit = (unsigned)(value % base);
        buffer[idx++] = (char)(digit < 10 ? ('0' + digit)
                                          : (uppercase ? 'A' : 'a') + digit - 10);
        value /= base;
    } while (value && idx < (int)sizeof(buffer));
    while (idx < width)
        buffer[idx++] = zero_pad ? '0' : ' ';
    while (idx--)
        console_putc(buffer[idx], count);
}

static void print_signed(int64_t value, int *count) {
    uint64_t magnitude;
    if (value < 0) {
        console_putc('-', count);
        magnitude = (uint64_t)(-value);
    } else {
        magnitude = (uint64_t)value;
    }
    print_unsigned(magnitude, 10, false, 0, false, count);
}

static void print_float(double value, int precision, int *count) {
    if (precision < 0)
        precision = 6;
    if (precision > 6)
        precision = 6;

    if (value < 0.0) {
        console_putc('-', count);
        value = -value;
    }

    double rounding = 0.5;
    for (int i = 0; i < precision; ++i)
        rounding /= 10.0;
    value += rounding;

    uint64_t whole = (uint64_t)value;
    double fractional = value - (double)whole;
    print_unsigned(whole, 10, false, 0, false, count);

    if (precision == 0)
        return;

    console_putc('.', count);
    for (int i = 0; i < precision; ++i) {
        fractional *= 10.0;
        int digit = (int)fractional;
        console_putc((char)('0' + digit), count);
        fractional -= digit;
    }
}

int printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int count = 0;

    while (*fmt) {
        if (*fmt != '%') {
            console_putc(*fmt++, &count);
            continue;
        }

        ++fmt;
        if (*fmt == '%') {
            console_putc('%', &count);
            ++fmt;
            continue;
        }

        bool long_mod = false;
        int width = 0;
        int precision = -1;
        bool zero_pad = false;
        bool precision_specified = false;

        if (*fmt == '0') {
            zero_pad = true;
            ++fmt;
        }

        while (is_digit(*fmt)) {
            width = width * 10 + (*fmt - '0');
            ++fmt;
        }

        if (*fmt == '.') {
            precision_specified = true;
            precision = 0;
            ++fmt;
            while (is_digit(*fmt)) {
                precision = precision * 10 + (*fmt - '0');
                ++fmt;
            }
        }

        if (*fmt == 'l') {
            long_mod = true;
            ++fmt;
        }

        char spec = *fmt++;
        switch (spec) {
            case 'c': {
                int ch = va_arg(args, int);
                console_putc((char)ch, &count);
                break;
            }
            case 's': {
                const char *str = va_arg(args, const char *);
                console_puts(str, &count);
                break;
            }
            case 'd':
            case 'i': {
                if (long_mod) {
                    long value = va_arg(args, long);
                    print_signed((int64_t)value, &count);
                } else {
                    int value = va_arg(args, int);
                    print_signed((int64_t)value, &count);
                }
                break;
            }
            case 'u': {
                if (long_mod) {
                    unsigned long value = va_arg(args, unsigned long);
                    print_unsigned((uint64_t)value, 10, false, width, zero_pad, &count);
                } else {
                    unsigned int value = va_arg(args, unsigned int);
                    print_unsigned((uint64_t)value, 10, false, width, zero_pad, &count);
                }
                break;
            }
            case 'x':
            case 'X': {
                bool uppercase = (spec == 'X');
                if (long_mod) {
                    unsigned long value = va_arg(args, unsigned long);
                    print_unsigned((uint64_t)value, 16, uppercase, width, zero_pad, &count);
                } else {
                    unsigned int value = va_arg(args, unsigned int);
                    print_unsigned((uint64_t)value, 16, uppercase, width, zero_pad, &count);
                }
                break;
            }
            case 'f': {
                double value = va_arg(args, double);
                print_float(value, precision_specified ? precision : 6, &count);
                break;
            }
            default:
                console_putc(spec, &count);
                break;
        }
    }

    va_end(args);
    return count;
}

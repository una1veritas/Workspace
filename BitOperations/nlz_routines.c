/*
 *  File: nlz_routines.c
 *
 *  Copyright (C) NAKAMURA Minoru <nminoru@nminoru.jp>
 */
#include <stdint.h>


int dummy(uint32_t x)
{
    return 0;
}


int nlz1(uint32_t x)
{
    x = x | ( x >>  1 );
    x = x | ( x >>  2 );
    x = x | ( x >>  4 );
    x = x | ( x >>  8 );
    x = x | ( x >> 16 );

#if 0
    return __builtin_popcount(~x);
#else
    int ret;
    __asm__ ("popcntl %[input], %[output]" : [output] "=r"(ret) : [input] "r"(~x)); 
#endif
    return ret;
}


int nlz2(uint32_t x)
{
    uint32_t y;
    int n = 32;
    y = x >> 16; if (y != 0){ n = n - 16 ; x = y; }
    y = x >>  8; if (y != 0){ n = n -  8 ; x = y; }
    y = x >>  4; if (y != 0){ n = n -  4 ; x = y; }
    y = x >>  2; if (y != 0){ n = n -  2 ; x = y; }
    y = x >>  1; if (y != 0){ return n - 2; }
    return n - x;
}


int nlz3(uint32_t x)
{
    uint32_t y, m, n;

    y = - (x >> 16);
    m = (y >> 16) & 16;
    n = 16 - m;
    x = x >> m;

    y = x - 0x100;
    m = (y >> 16) & 8;
    n = n + m;
    x = x << m;

    y = x - 0x1000;
    m = (y >> 16) & 4;
    n = n + m;
    x = x  << m;

    y = x - 0x4000;
    m = (y >> 16) & 2;
    n = n + m;
    x = x  << m;

    y = x >> 14;
    m = y & ~(y >> 1);

    return n + 2 - m;
}


int nlz4(uint32_t x)
{
    int n;

#define USE_DOUBLE

#ifdef USE_DOUBLE
    union {
        uint64_t as_uint64;
        double   as_double;
    } data;
    data.as_double = (double)x + 0.5;
    n = 1054 - (data.as_uint64 >> 52);
#else
    union {
        uint32_t as_uint32;
        float    as_float;
    } data;
    data.as_float = (float)x + 0.5;
    n = 158 - (data.as_uint32 >> 23);
#endif

    return n;
}


int nlz5(uint32_t x)
{
    if(x==0)
        return 32;

    uint32_t c = (x & 0xAAAAAAAA) ? 0x01 : 0;
    if (x & 0xCCCCCCCC)
        c |= 0x02;
    if(x & 0xF0F0F0F0)
        c |= 0x04;
    if(x & 0xFF00FF00)
        c |= 0x08;

    return (x & 0xFFFF0000) ? (c | 0x10) : c;
}

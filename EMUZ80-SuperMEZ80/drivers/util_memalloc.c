/*
 * Copyright (c) 2023 @hanyazou
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "utils.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

static char *memalloc_buf = NULL;
static char *memalloc_ptr = NULL;
static unsigned int memalloc_size = 0;

//#define DEBUG
//#define DEBUG_VERBOSE

#define  err(args...) do { printf("E/memalloc: "   args); } while(0)
#define warn(args...) do { printf("W/memalloc: "   args); } while(0)
#define info(args...) do { printf("I/memalloc: "   args); } while(0)
#ifdef DEBUG
#define  dbg(args...) do { printf("D/memalloc: "   args); } while(0)
#else
#define  dbg(args...) do { } while(0)
#endif
#ifdef DEBUG_VERBOSE
#define  verbose(args...) do { printf("V/memalloc: "   args); } while(0)
#else
#define  verbose(args...) do { } while(0)
#endif

void util_memalloc_init(void *buf, unsigned int size)
{
    assert(memalloc_buf == NULL);
    dbg("%s: buf=0x%lx, size=%d\n\r", __func__, (unsigned long)buf, size);
    memalloc_buf = buf;
    memalloc_ptr = buf;
    memalloc_size = size;
}

void *util_memalloc(unsigned int size)
{
    assert(memalloc_buf != NULL);
    verbose("%s: %d bytes requested, current ptr=0x%lx, free=%d\n\r", __func__, size,
            (unsigned long)memalloc_ptr, memalloc_buf + memalloc_size - memalloc_ptr);
    if (memalloc_buf + memalloc_size < memalloc_ptr + size) {
        warn("%s: %d bytes requested, current ptr=0x%lx, free=%d\n\r", __func__, size,
             (unsigned long)memalloc_ptr, memalloc_buf + memalloc_size - memalloc_ptr);
        warn("%s: INSUFFICIENT MEMORY\n\r", __func__);
        return NULL;
    }

    void *result = memalloc_ptr;
    dbg("%s: 0x%lx %d bytes allocated\n\r", __func__, (unsigned long)result, size);
    memalloc_ptr += size;
    return result;
}

void util_memfree(void *ptr)
{
    assert(memalloc_buf != NULL);
    verbose("%s: current ptr=0x%lx, freed ptr=0x%lx, %d bytes\n\r", __func__,
            (unsigned long)memalloc_ptr, (unsigned long)ptr, memalloc_ptr - (char *)ptr);
    if ((char *)ptr < memalloc_buf || memalloc_buf + memalloc_size <= (char *)ptr) {
        err("%s: freed ptr 0x%lx is out of range (buf=0x%lx, size=%d)\n\r", __func__,
            (unsigned long)ptr, (unsigned long)memalloc_buf, memalloc_size);
        return;
    }

    if ((char *)ptr < memalloc_ptr) {
        dbg("%s: 0x%lx %d bytes freed\n\r", __func__, (unsigned long)ptr,
            memalloc_ptr - (char *)ptr);
        memalloc_ptr = ptr;
    }
    verbose("%s: current ptr=0x%lx, free=%d\n\r", __func__,
            (unsigned long)memalloc_ptr, memalloc_buf + memalloc_size - memalloc_ptr);
}

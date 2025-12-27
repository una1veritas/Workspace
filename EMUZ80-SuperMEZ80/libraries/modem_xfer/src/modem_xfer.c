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

#include <modem_xfer.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdarg.h>

// #define DEBUG
#include "modem_xfer_debug.h"

int modem_xfer_discard(void)
{
    int res = 0;
    uint8_t rxb;
    uint8_t tmp[16];
    while (modem_xfer_rx(&rxb, 300) == 1) {
        if (res < sizeof(tmp)) {
            tmp[res] = rxb;
        }
        res++;
    }
    modem_xfer_hex_dump(MODEM_XFER_LOG_DEBUG, tmp, sizeof(tmp));

    return res;
}

int modem_xfer_recv_bytes(uint8_t *buf, int n, int timeout_ms)
{
    int i;
    int res;

    for (i = 0; i < n; i++) {
        res = modem_xfer_rx(&buf[i], timeout_ms);
        if (res == 0) {
            //  time out (there might be no sender)
            return i;
        }
        if (res < 0) {
            // error
            return res;
        }
    }

    return i;
}

void modem_xfer_hex_dump(int log_level, uint8_t *buf, int n)
{
    int i;

    #if !defined(DEBUG)
    if (log_level <= MODEM_XFER_LOG_DEBUG) {
        return;
    }
    #endif

    for (i = 0; i < n; i += 16) {
        modem_xfer_printf(log_level, "%04X: "
            "%02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X "
            "%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c\n",
            i,
            buf[i+0], buf[i+1], buf[i+2], buf[i+3], buf[i+4], buf[i+5], buf[i+6], buf[i+7],
            buf[i+8], buf[i+9], buf[i+10], buf[i+11], buf[i+12], buf[i+13], buf[i+14], buf[i+15],
            isprint(buf[i+0]) ? buf[i+0] : '.', isprint(buf[i+1]) ? buf[i+1] : '.',
            isprint(buf[i+2]) ? buf[i+2] : '.', isprint(buf[i+3]) ? buf[i+3] : '.',
            isprint(buf[i+4]) ? buf[i+4] : '.', isprint(buf[i+5]) ? buf[i+5] : '.',
            isprint(buf[i+6]) ? buf[i+6] : '.', isprint(buf[i+7]) ? buf[i+7] : '.',
            isprint(buf[i+8]) ? buf[i+8] : '.', isprint(buf[i+9]) ? buf[i+9] : '.',
            isprint(buf[i+10]) ? buf[i+10] : '.', isprint(buf[i+11]) ? buf[i+11] : '.',
            isprint(buf[i+12]) ? buf[i+12] : '.', isprint(buf[i+13]) ? buf[i+12] : '.',
            isprint(buf[i+14]) ? buf[i+14] : '.', isprint(buf[i+15]) ? buf[i+15] : '.');
    }
}

uint16_t modem_xfer_crc16(uint16_t crc, const void *buf, unsigned int count)
{
    uint8_t *p = (uint8_t*)buf;
    uint8_t *endp = p + count;

    while (p < endp) {
        crc = (crc >> 8)|(crc << 8);
        crc ^= *p++;
        crc ^= ((crc & 0xff) >> 4);
        crc ^= (crc << 12);
        crc ^= ((crc & 0xff) << 5);
    }

    return crc;
}

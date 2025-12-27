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

#ifndef __MODEM_XFER_H__
#define __MODEM_XFER_H__

#include <stdint.h>

#define MODEM_XFER_BUF_SIZE 128
#define MODEM_XFER_UNKNOWN_FILE_SIZE ((uint32_t)0xffffffff)

enum {
    MODEM_XFER_LOG_ERROR,
    MODEM_XFER_LOG_WARNING,
    MODEM_XFER_LOG_INFO,
    MODEM_XFER_LOG_DEBUG,
    MODEM_XFER_LOG_VERBOSE,
};

enum {
    MODEM_XFER_STAT_IVALID,
    MODEM_XFER_STAT_INIT,
    MODEM_XFER_STAT_XFER,
    MODEM_XFER_STAT_END,
};

enum {
    MODEM_XFER_RES_OK = 0,
    MODEM_XFER_RES_CANCELED,
    MODEM_XFER_RES_TIMEOUT,
    MODEM_XFER_RES_EIO,
    MODEM_XFER_RES_EPTOROCOL,
    MODEM_XFER_RES_ESEQUENCE,
};

typedef struct {
    uint8_t stat;
    uint8_t seqno;
    uint8_t *buf;
    int num_files_xfered;
    char file_name[13];
    uint32_t file_offset;
    unsigned long file_size;
    uint32_t num_bytes_xfered;
} ymodem_context;

extern int ymodem_receive(uint8_t buf[MODEM_XFER_BUF_SIZE]);
extern void ymodem_receive_init(ymodem_context *ctx, uint8_t buf[MODEM_XFER_BUF_SIZE]);
extern int ymodem_receive_block(ymodem_context *ctx, unsigned int *sizep);

extern void ymodem_send_init(ymodem_context *ctx, uint8_t buf[MODEM_XFER_BUF_SIZE]);
extern int ymodem_send_header(ymodem_context *ctx, char *file_name, uint32_t size);
extern int ymodem_send_block(ymodem_context *ctx);
extern int ymodem_send_end(ymodem_context *ctx);
extern void ymodem_send_cancel(ymodem_context *ctx);

extern int modem_xfer_discard(void);
extern int modem_xfer_recv_bytes(uint8_t *buf, int n, int timeout_ms);
extern void modem_xfer_hex_dump(int log_level, uint8_t *buf, int n);
extern uint16_t modem_xfer_crc16(uint16_t crc, const void *buf, unsigned int count);
extern int modem_xfer_tx(uint8_t);
extern int modem_xfer_rx(uint8_t *, int timeout_ms);
extern int modem_xfer_save(char*, uint32_t, uint8_t*, uint16_t);
extern void modem_xfer_printf(int log_level, const char *format, ...)
    __attribute__ ((format (printf, 2, 3)));

#endif  // __MODEM_XFER_H__

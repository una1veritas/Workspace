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

#define REQ  'C'
#define SOH  0x01
#define STX  0x02
#define EOT  0x04
#define ACK  0x06
#define NAK  0x15
#define CAN  0x18

#define BUFSIZE 128
#define SOH_SIZE 128
#define STX_SIZE 1024

//#define DEBUG
//#define DEBUG_VERBOSE

#include "modem_xfer_debug.h"
#include "ymodem.h"

int ymodem_receive(uint8_t buf[MODEM_XFER_BUF_SIZE])
{
    int res;
    unsigned int n;

    ymodem_context ctx;
    ymodem_receive_init(&ctx, buf);
    while ((res = ymodem_receive_block(&ctx, &n)) == MODEM_XFER_RES_OK) {
        if (ctx.file_name[0] == '\0') {
            return MODEM_XFER_RES_OK;
        }
        res = modem_xfer_save(ctx.file_name, ctx.file_offset, ctx.buf, n);
        if (res != MODEM_XFER_RES_OK) {
            ymodem_send_cancel(&ctx);
            return res;
        }
    }

    return res;
}

void ymodem_receive_init(ymodem_context *ctx, uint8_t buf[MODEM_XFER_BUF_SIZE])
{
    dbg("--: %s:\n",  __func__);
    ctx->stat = MODEM_XFER_STAT_INIT;
    ctx->buf = buf;
    ctx->num_files_xfered = 0;
    ctx->seqno = 0;
}

int ymodem_receive_block(ymodem_context *ctx, unsigned int *sizep)
{
    int res, retry;
    uint8_t *buf = ctx->buf;
    uint16_t crc;
    uint8_t crc_buf[2];

    if (ctx->stat == MODEM_XFER_STAT_END) {
        *sizep = 0;
        return MODEM_XFER_RES_OK;
    }
    if (ctx->stat == MODEM_XFER_STAT_XFER) {
        ctx->file_offset += MODEM_XFER_BUF_SIZE;
    }

 entry:
    retry = 0;
    while (retry++ < (ctx->stat == MODEM_XFER_STAT_INIT ? 25 : 5)) {
        if (ctx->stat == MODEM_XFER_STAT_INIT) {
            dbg("%02X: send REQ\n", ctx->seqno);
            modem_xfer_tx(REQ);
        }

        /*
         * receive block herader
         */
        if (modem_xfer_recv_bytes(buf, 1, 1000) != 1) {
            dbg("%02X: header timeout\n", ctx->seqno);
            continue;
        }

        if (ctx->stat == MODEM_XFER_STAT_XFER && buf[0] == EOT) {
            dbg("%02X: EOT\n", ctx->seqno);
            modem_xfer_tx(NAK);
            modem_xfer_recv_bytes(&buf[0], 1, 1000);
            if (buf[0] != EOT) {
                warn("WARNING: EOT expected but received %02X\n", buf[0]);
            }
            modem_xfer_tx(ACK);
            ctx->num_files_xfered++;
            ctx->stat = MODEM_XFER_STAT_INIT;
            ctx->seqno = 0;
            goto entry;
        }
        if (buf[0] != SOH) {
            dbg("%02X: invalid header %02X\n", ctx->seqno, buf[0]);
            goto retry;
        }

        /*
         * receive sequence number
         */
        if (modem_xfer_recv_bytes(&buf[1], 2, 300) != 2) {
            dbg("%02X: seqno timeout\n", ctx->seqno);
            goto retry;
        }
        dbg("%02X: %02X %02X %02X\n", ctx->seqno, buf[0], buf[1], buf[1]);
        //if (buf[1] != ctx->seqno && buf[2] != ((~ctx->seqno) + 1)) {
        if (buf[1] != ctx->seqno && buf[2] != ((~ctx->seqno))) {
            dbg("%02X: invalid sequence number\n", ctx->seqno);
            goto retry;
        }

        /*
         * receive payload
         */
        int n = modem_xfer_recv_bytes(buf, BUFSIZE, 1000);
        if (n != BUFSIZE) {
            info("%02X: payload timeout, n=%d\n", ctx->seqno, n);
            goto retry;
        }
        dbg("%02X: %d bytes received\n", ctx->seqno, BUFSIZE);
        #ifdef DEBUG_VERBOSE
        modem_xfer_hex_dump(MODEM_XFER_LOG_DEBUG, buf, BUFSIZE);
        #endif
        crc = modem_xfer_crc16(0, buf, BUFSIZE);
        if (modem_xfer_recv_bytes(crc_buf, 2, 1000) != 2) {
            err("%02X: CEC timeout\n", ctx->seqno);
            goto retry;
        }
        dbg("%02X: crc16: %04x %s %04x\n", ctx->seqno, crc_buf[0] * 256 + crc_buf[1],
            (crc_buf[0] * 256 + crc_buf[1]) == crc ? "==" : "!=", crc);
        if ((crc_buf[0] * 256 + crc_buf[1]) != crc) {
            goto retry;
        }
        modem_xfer_tx(ACK);

        if (ctx->stat == MODEM_XFER_STAT_INIT) {
            memcpy(ctx->file_name, buf, sizeof(ctx->file_name));
            ctx->file_name[sizeof(ctx->file_name) - 1] = '\0';
            if (ctx->file_name[0] == 0x00) {
                info("total %d file%s received\n", ctx->num_files_xfered,
                     1 < ctx->num_files_xfered ? "s" : "");
                modem_xfer_tx(ACK);
                ctx->stat = MODEM_XFER_STAT_END;
                return MODEM_XFER_RES_OK;
            }
            buf[BUFSIZE - 1] = '\0';  // fail safe
            modem_xfer_hex_dump(MODEM_XFER_LOG_DEBUG, buf, 16);
            dbg("file info string: %s\n", &buf[strlen((char *)buf) + 1]);
            if (sscanf((char*)&buf[strlen((char *)buf) + 1], "%lu", &ctx->file_size) != 1) {
                warn("WARNING: unknown file size\n");
                ctx->file_size = 0;
            }
            ctx->seqno++;
            ctx->file_offset = 0;
            ctx->stat = MODEM_XFER_STAT_XFER;
            dbg("%02X: send REQ\n", ctx->seqno);
            modem_xfer_tx(REQ);
            info("receiving file '%s', %lu bytes\n", ctx->file_name, ctx->file_size);
            goto entry;
        } else {
            if (ctx->file_size == 0 || ctx->file_offset < ctx->file_size) {
                if (ctx->file_size != 0 && ctx->file_size < ctx->file_offset + BUFSIZE) {
                    *sizep = (unsigned int)(ctx->file_size - ctx->file_offset);
                } else {
                    *sizep = BUFSIZE;
                }
                ctx->seqno++;
                return MODEM_XFER_RES_OK;
            }
        }
        ctx->seqno++;
        continue;
    retry:
        res = modem_xfer_discard();
        dbg("%02X: discard %d bytes and send NAK\n", ctx->seqno, res);
        modem_xfer_tx(NAK);
    }

    ymodem_send_cancel(ctx);

    return MODEM_XFER_RES_CANCELED;
}

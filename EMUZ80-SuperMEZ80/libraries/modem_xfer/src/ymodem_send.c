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

//#define DEBUG
//#define DEBUG_VERBOSE

#include "modem_xfer_debug.h"
#include "ymodem.h"

static int __ymodem_send_block(ymodem_context *ctx);

void ymodem_send_init(ymodem_context *ctx, uint8_t buf[MODEM_XFER_BUF_SIZE])
{
    dbg("--: %s:\n",  __func__);
    ctx->stat = MODEM_XFER_STAT_INIT;
    ctx->buf = buf;
    ctx->num_files_xfered = 0;
    ctx->seqno = 0;
    ctx->num_bytes_xfered = 0;
}

int ymodem_send_eot(ymodem_context *ctx)
{
    int n;
    uint8_t buf[1];
    int retry = 5;

    while (0 < retry--) {
        dbg("%02X: %s: send EOT (1/2)\n",  ctx->seqno, __func__);
        modem_xfer_tx(EOT);
        n = modem_xfer_recv_bytes(&buf[0], 1, 5000);
        if (n != 1) {
            dbg("%02X: %s: timeout\n",  ctx->seqno, __func__);
            continue;
        }
        if (buf[0] == ACK) {
            // NAK was expected, but this might be OK
            dbg("%02X: %s: received ACK (this might be OK)\n",  ctx->seqno, __func__);
            return MODEM_XFER_RES_OK;
        }
        if (buf[0] == CAN) {
            dbg("%02X: %s: received CAN\n",  ctx->seqno, __func__);
            return MODEM_XFER_RES_CANCELED;
        }
        if (buf[0] != NAK) {
            dbg("%02X: %s: received 0x%02x, retry ...\n",  ctx->seqno, __func__, buf[0]);
            continue;
        }
        dbg("%02X: %s: received NAK\n",  ctx->seqno, __func__);
        dbg("%02X: %s: send EOT (2/2)\n",  ctx->seqno, __func__);
        modem_xfer_tx(EOT);
        n = modem_xfer_recv_bytes(&buf[0], 1, 5000);
        if (n != 1) {
            dbg("%02X: %s: timeout\n",  ctx->seqno, __func__);
            continue;
        }
        if (buf[0] == CAN) {
            dbg("%02X: %s: received CAN\n",  ctx->seqno, __func__);
            return MODEM_XFER_RES_CANCELED;
        }
        if (buf[0] != ACK) {
            dbg("%02X: %s: received 0x%02x, retry ...\n",  ctx->seqno, __func__, buf[0]);
            continue;
        }
        dbg("%02X: %s: received ACK (completed)\n",  ctx->seqno, __func__);
        return MODEM_XFER_RES_OK;
    }

    return MODEM_XFER_RES_TIMEOUT;
}

void ymodem_send_cancel(ymodem_context *ctx)
{
    uint8_t buf[1];
    dbg("%02X: %s:\n",  ctx->seqno, __func__);
    info("cancel\n");
    modem_xfer_tx(CAN);
    modem_xfer_tx(CAN);
    modem_xfer_rx(buf, 1000);
}

int ymodem_send_wait_req(ymodem_context *ctx, int timeout_sec)
{
    int n;
    uint8_t buf[1];
    int retry = 0;

    dbg("%02X: %s:\n",  ctx->seqno, __func__);
    while (timeout_sec == 0 || retry++ < timeout_sec) {
        n = modem_xfer_recv_bytes(&buf[0], 1, 1000);
        if (n != 1) {
            dbg("%02X: %s: timeout\n",  ctx->seqno, __func__);
            continue;
        }
        if (buf[0] == REQ) {
            dbg("%02X: %s: received REQ\n", ctx->seqno, __func__);
            return MODEM_XFER_RES_OK;
        }
        if (buf[0] == CAN) {
            info("%02X: %s: received CAN 0x%02x\n", ctx->seqno, __func__, buf[0]);
            return MODEM_XFER_RES_CANCELED;
        }
        if (buf[0] == 0x00 ||  // break
            buf[0] == 0x03 ||  // ^C
            buf[0] == 0x1a) {  // ^Z, EOF
            info("%02X: %s: interrupted by 0x%02x\n", ctx->seqno, __func__, buf[0]);
            return MODEM_XFER_RES_CANCELED;
        }
    }

    info("%02X: %s: TIMEOUT\n",  ctx->seqno, __func__);
    return MODEM_XFER_RES_TIMEOUT;
}

int ymodem_send_header(ymodem_context *ctx, char *file_name, uint32_t size)
{
    int n, res;
    int retry = 5;
    uint16_t crc;
    int timeout_sec = 5;

    if (ctx->stat == MODEM_XFER_STAT_XFER) {
        dbg("%02X: %s: send EOT\n",  ctx->seqno, __func__);
        res = ymodem_send_eot(ctx);
        if (res != MODEM_XFER_RES_OK) {
            return res;
        }
        ctx->stat = MODEM_XFER_STAT_INIT;
        ctx->num_files_xfered++;
    } else {
        timeout_sec = 60;
    }
    if (ctx->stat != MODEM_XFER_STAT_INIT) {
        return MODEM_XFER_RES_ESEQUENCE;
    }
    res = ymodem_send_wait_req(ctx, timeout_sec);
    if (res != MODEM_XFER_RES_OK) {
        return res;
    }

    ctx->seqno = 0;
    dbg("%02X: %s: '%s' %lu\n",  ctx->seqno, __func__, file_name, (unsigned long)size);
    memset(ctx->buf, 0x00, MODEM_XFER_BUF_SIZE);
    if (size == MODEM_XFER_UNKNOWN_FILE_SIZE) {
        snprintf((char *)ctx->buf, MODEM_XFER_BUF_SIZE, "%s%c", file_name, '\0');
    } else {
        snprintf((char *)ctx->buf, MODEM_XFER_BUF_SIZE, "%s%c%lu", file_name, '\0',
                 (unsigned long)size);
    }
    res = __ymodem_send_block(ctx);
    if (res != MODEM_XFER_RES_OK) {
        return res;
    }

    if (file_name[0] == '\0' && size == 0) {
        dbg("%02X: %s: sent last header\n",  ctx->seqno, __func__);
        ctx->stat = MODEM_XFER_STAT_END;
        return MODEM_XFER_RES_OK;
    }

    if (size != MODEM_XFER_UNKNOWN_FILE_SIZE) {
        info("sending file '%s', %lu bytes\n", file_name, (unsigned long)size);
    } else {
        info("sending file '%s' ...\n", file_name);
    }
    ctx->stat = MODEM_XFER_STAT_XFER;
    res = ymodem_send_wait_req(ctx, 5);

    return res;
}

static int __ymodem_send_block(ymodem_context *ctx)
{
    int n, res;
    uint8_t buf[1];
    int retry = 5;
    uint16_t crc;

    while (0 < retry--) {
        dbg("%02X: %s: SOH %d bytes\n",  ctx->seqno, __func__, MODEM_XFER_BUF_SIZE);
        modem_xfer_tx(SOH);
        modem_xfer_tx(ctx->seqno);
        //modem_xfer_tx((~ctx->seqno) + 1);
        modem_xfer_tx((~ctx->seqno));
        for (unsigned int i = 0; i < MODEM_XFER_BUF_SIZE; i++) {
            modem_xfer_tx(ctx->buf[i]);
        }
        crc = modem_xfer_crc16(0, ctx->buf, MODEM_XFER_BUF_SIZE);
        modem_xfer_tx((crc >> 8) & 0xff);
        modem_xfer_tx((crc >> 0) & 0xff);
        n = modem_xfer_recv_bytes(&buf[0], 1, 5000);
        if (n != 1) {
            continue;
        }
        if (buf[0] == ACK) {
            dbg("%02X: %s: received ACK (completed)\n",  ctx->seqno, __func__);
            ctx->seqno++;
            return MODEM_XFER_RES_OK;
        }
        if (buf[0] == CAN) {
            dbg("%02X: %s: received CAN\n",  ctx->seqno, __func__);
            return MODEM_XFER_RES_CANCELED;
        }
        if (buf[0] == NAK) {
            dbg("%02X: %s: received NAK\n",  ctx->seqno, __func__);
        } else {
            dbg("%02X: %s: received 0x%02x\n",  ctx->seqno, __func__, buf[0]);
        }
    }

    info("%02X: %s: TIMEOUT\n",  ctx->seqno, __func__);
    return MODEM_XFER_RES_TIMEOUT;
}

int ymodem_send_block(ymodem_context *ctx)
{
    int res = __ymodem_send_block(ctx);
    if (res == MODEM_XFER_RES_OK) {
        ctx->num_bytes_xfered += MODEM_XFER_BUF_SIZE;
    }
    return res;
}

int ymodem_send_end(ymodem_context *ctx)
{
    int res;

    dbg("%02X: %s: all file(s) were sent\n",  ctx->seqno, __func__);
    res = ymodem_send_header(ctx, "", 0);
    if (res != MODEM_XFER_RES_OK) {
        ymodem_send_cancel(ctx);
    }
    dbg("%02X: %s: COMPLETED\n",  ctx->seqno, __func__);
    info("total %d file%s, %ld bytes sent\n", ctx->num_files_xfered,
         1 < ctx->num_files_xfered ? "s" : "", (unsigned long)ctx->num_bytes_xfered);

    return res;
}

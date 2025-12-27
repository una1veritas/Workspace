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

#include <supermez80.h>
#include <stdio.h>
#include <assert.h>

#include <ff.h>
#include <utils.h>
#include <modem_xfer.h>

enum {
    AUX_CLOSED,
    AUX_FILE_WRITING,
    AUX_FILE_READING,
    AUX_MODEM_SENDING,
    AUX_MODEM_RECEIVING,
};

static uint8_t aux_status = AUX_CLOSED;
static FIL *aux_filep = NULL;
static uint8_t aux_error = 0;
static timer_t aux_timer;
static FSIZE_t aux_in_offset = 0;
static uint8_t aux_in_line_feed = 0;
static char *aux_file_name = NULL;

#define ERROR(a) if (!aux_error) { aux_error = 1; a; }

static void aux_modem_timer_callback(timer_t *timer) {
    #ifdef CPM_IO_AUX_DEBUG
    modem_xfer_printf(MODEM_XFER_LOG_DEBUG, "%s: close\n", __func__);
    #endif
    if (aux_status == AUX_CLOSED) {
        return;
    }
    modem_close();
    aux_status = AUX_CLOSED;
}

static void aux_file_timer_callback(timer_t *timer) {
    FRESULT fr;

    #ifdef CPM_IO_AUX_DEBUG
    printf("%s: close %s\n\r", __func__, aux_file_name);
    #endif
    if (aux_status == AUX_CLOSED) {
        return;
    }

    fr = f_close(aux_filep);
    put_file(aux_filep);
    aux_filep = NULL;
    aux_status = AUX_CLOSED;
    if (fr != FR_OK) {
        ERROR(util_fatfs_error(fr, "f_close() failed"));
        return;
    }
    aux_error = 0;
}

static int aux_out_conv(uint8_t *c) {
    if (*c == 0x00 || *c == '\r') {
        // ignore 00h and 0Dh (Carriage Return)
        #ifdef CPM_IO_AUX_DEBUG_VERBOSE
        printf("%s: ignore %02Xh\n\r", __func__, *c);
        #endif
        return 1;
    }

    if (*c == 0x1a) {
        // 1Ah (EOF)
        #ifdef CPM_IO_AUX_DEBUG_VERBOSE
        printf("%s: %02Xh is EOF\n\r", __func__, *c);
        #endif
        timer_expire(&aux_timer);  // close the file immediately
        return 1;
    }

    return 0;
}

static int aux_in_conv(uint8_t *c) {
    if (*c == 0x00) {
        #ifdef CPM_IO_AUX_DEBUG_VERBOSE
        printf("%s: ignore %02Xh\n\r", __func__, *c);
        #endif
        return 1;
    }
    if (*c == '\n') {
        // convert LF (\n 0Ah) to CRLF (\r 0Dh, \n 0Ah)
        #ifdef CPM_IO_AUX_DEBUG_VERBOSE
        printf("%s: convert LF (\\n 0Ah) to CRLF (\\r 0Dh, \\n 0Ah)\n\r", __func__);
        #endif
        *c = '\r';
        aux_in_line_feed = 1;
    }

    return 0;
}

static int aux_file_open(char *file_name, BYTE mode) {
    FRESULT fr;

    if (aux_status != AUX_CLOSED) {
        ERROR(printf("aux: internal status error\n\r"));
        return 0;
    }

    if (aux_filep == NULL) {
        aux_filep = get_file();
        if (aux_filep == NULL) {
            ERROR(printf("aux: can not allocate file\n\r"));
            aux_status = AUX_CLOSED;
            return -1;
        }
        aux_error = 0;
        aux_file_name = file_name;
        #ifdef CPM_IO_AUX_DEBUG
        printf("%s: open %s\n\r", __func__, aux_file_name);
        #endif
        fr = f_open(aux_filep, aux_file_name, mode);
        if (fr != FR_OK) {
            ERROR(util_fatfs_error(fr, "f_open() failed"));
            put_file(aux_filep);
            aux_filep = NULL;
            aux_status = AUX_CLOSED;
            return -1;
        }
        aux_error = 0;
    }
    if (mode & FA_WRITE) {
        aux_status = AUX_FILE_WRITING;
    } else {
        fr = f_lseek(aux_filep, aux_in_offset);
        if (fr != FR_OK) {
            ERROR(util_fatfs_error(fr, "f_lseek(/AUXIN.TXT) failed"));
            aux_in_offset = 0;
            f_rewind(aux_filep);
        }
        aux_status = AUX_FILE_READING;
    }

    return 0;
}

void aux_file_write(uint8_t c) {
    FRESULT fr;

    // open the file before aux_out_conv() to ensure that the file shall be created
    if (aux_status != AUX_FILE_WRITING) {
        timer_expire(&aux_timer);  // close the file immediately
        if (aux_file_open("/AUXOUT.TXT", FA_WRITE|FA_OPEN_APPEND) != 0) {
            return;
        }
    }
    timer_set_relative(&aux_timer, aux_file_timer_callback, 1000);

    if (aux_out_conv(&c)) {
        return;
    }

    UINT bw;
    fr = f_write(aux_filep, &c, 1, &bw);
    #ifdef CPM_IO_AUX_DEBUG_VERBOSE
    printf("%s: f_write: fr=%d, bw=%d, c=%02Xh\n\r", __func__, fr, bw, c);
    #endif
    if (fr != FR_OK || bw != 1) {
        ERROR(util_fatfs_error(fr, "f_write(/AUXOUT.TXT) failed"));
        return;
    }
    aux_error = 0;
}

void aux_file_read(uint8_t *c) {
    FRESULT fr;
    UINT br;

 read_one_more:
    if (aux_status != AUX_FILE_READING) {
        aux_in_line_feed = 0;
        timer_expire(&aux_timer);  // close the file immediately
        if (aux_file_open("/AUXIN.TXT", FA_READ|FA_OPEN_ALWAYS) != 0) {
            return;
        }
    }
    timer_set_relative(&aux_timer, aux_file_timer_callback, 1000);

    if (aux_in_line_feed) {
        // insert LF (\n 0Ah)
        aux_in_line_feed = 0;
        *c = '\n';
        #ifdef CPM_IO_AUX_DEBUG_VERBOSE
        printf("%s: insert LF (\\n %02Xh)\n\r", __func__, *c);
        #endif
        return;
    }

    fr = f_read(aux_filep, c, 1, &br);
    #ifdef CPM_IO_AUX_DEBUG_VERBOSE
    if (br == 0) {
        printf("%s: f_read: fr=%d, br=0\n\r", __func__, fr);
    } else {
        printf("%s: f_read: fr=%d, br=%d, c=%02Xh\n\r", __func__, fr, br, *c);
    }
    #endif
    if (fr != FR_OK || br != 1) {
        if (fr != FR_OK) {
            ERROR(util_fatfs_error(fr, "f_read(/AUXIN.TXT) failed"));
        }
        if (br == 0) {
            // reaching end of file
            timer_expire(&aux_timer);  // close the file immediately
            aux_in_offset = 0;  // rewind file position
        }
        #ifdef CPM_IO_AUX_DEBUG_VERBOSE
        printf("%s: reached EOF\n\r", __func__);
        #endif
        *c = 0x1a;  // return EOF at end of file or some error
        return;
    }
    aux_in_offset++;
    aux_error = 0;
    if (aux_in_conv(c)) {
        goto read_one_more;
    }
    #ifdef CPM_IO_AUX_DEBUG_VERBOSE
    printf("%s: return %02Xh\n\r", __func__, *c);
    #endif
}

void aux_modem_write(uint8_t c) {
    if (aux_out_conv(&c)) {
        return;
    }
    if (aux_status != AUX_MODEM_SENDING) {
        timer_expire(&aux_timer);  // close the file immediately
        printf("waiting for file transfer request via the terminal ...\n\r");
        if (modem_send_open("AUXOUT.TXT", MODEM_XFER_UNKNOWN_FILE_SIZE) != 0) {
            ERROR(printf("modem_send_open() failed\n\r"));
            return;
        }
        aux_status = AUX_MODEM_SENDING;
    }

    if (modem_write(&c, 1) != 1) {
        ERROR(printf("modem_write() failed\n\r"));
        goto exit;
    }
    aux_error = 0;

 exit:
    timer_set_relative(&aux_timer, aux_modem_timer_callback, 1000);
}

void aux_modem_read(uint8_t *c) {
    FRESULT fr;
    int n;

 read_one_more:
    if (aux_in_line_feed) {
        // insert LF (\n 0Ah)
        aux_in_line_feed = 0;
        *c = '\n';
        goto  exit;
    }

    if (aux_status != AUX_MODEM_RECEIVING) {
        aux_in_line_feed = 0;
        modem_xfer_printf(MODEM_XFER_LOG_DEBUG, "|%s(%d)\r\n", __func__, __LINE__);
        timer_expire(&aux_timer);  // close immediately
        #ifdef CPM_IO_AUX_DEBUG
        printf("%s: modem_recv_open()\n\r", __func__);
        printf("key_input_count=%d drop_count=%d\r\n", key_input_count, key_input_drop_count);
        #endif
        if (modem_recv_open() != 0) {
            printf("modem_recv_open() failed\n\r");
            *c = 0x1a;  // EOF
            goto  exit;
        }
        aux_status = AUX_MODEM_RECEIVING;
    }
    timer_set_relative(&aux_timer, aux_modem_timer_callback, 5000);

    if ((n = modem_read(c, 1)) != 1) {
        if (n < 0) {
            printf("modem_read() failed\n\r");
        }
        timer_expire(&aux_timer);  // close immediately
        *c = 0x1a;  // EOF
        goto  exit;
    }
    timer_set_relative(&aux_timer, aux_modem_timer_callback, 10000);
    aux_error = 0;
    if (aux_in_conv(c)) {
        goto read_one_more;
    }

 exit:
    #ifdef CPM_IO_AUX_DEBUG_VERBOSE
    modem_xfer_printf(MODEM_XFER_LOG_DEBUG, "aux mr %02Xh %c %d\r\n", *c, isprint(*c) ? *c : '.',
                      key_input_io_read_count);
    #endif
    return;
}

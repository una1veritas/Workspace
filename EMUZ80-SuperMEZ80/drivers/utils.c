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

#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include "utils.h"

void util_hexdump(const char *header, const void *addr, unsigned int size)
{
    char chars[17];
    const uint8_t *buf = addr;
    size = ((size + 15) & ~0xfU);
    for (int i = 0; i < size; i++) {
        if ((i % 16) == 0)
            printf("%s%04x:", header, i);
        printf(" %02x", buf[i]);
        if (0x20 <= buf[i] && buf[i] <= 0x7e) {
            chars[i % 16] = buf[i];
        } else {
            chars[i % 16] = '.';
        }
        if ((i % 16) == 15) {
            chars[16] = '\0';
            printf(" %s\n\r", chars);
        }
    }
}

void util_addrdump(const char *header, uint32_t addr_offs, const void *addr, unsigned int size)
{
    char chars[17];
    const uint8_t *buf = addr;
    size = ((size + 15) & ~0xfU);
    for (unsigned int i = 0; i < size; i++) {
        if ((i % 16) == 0)
            printf("%s%06lx:", header, addr_offs + i);
        printf(" %02x", buf[i]);
        if (0x20 <= buf[i] && buf[i] <= 0x7e) {
            chars[i % 16] = buf[i];
        } else {
            chars[i % 16] = '.';
        }
        if ((i % 16) == 15) {
            chars[16] = '\0';
            printf(" %s\n\r", chars);
        }
    }
}

void util_hexdump_sum(const char *header, const void *addr, unsigned int size)
{
    util_hexdump(header, addr, size);

    uint8_t sum = 0;
    const uint8_t *p = addr;
    for (int i = 0; i < size; i++)
        sum += *p++;
    printf("%s%53s CHECKSUM: %02x\n\r", header, "", sum);
}

int util_stricmp(const char *a, const char *b)
{
  int ua, ub;
  do {
      ua = toupper((unsigned char)*a++);
      ub = toupper((unsigned char)*b++);
   } while (ua == ub && ua != '\0');
   return ua - ub;
}

void util_fatfs_error(FRESULT fres, char *msg)
{
    struct {
        FRESULT fres;
        char *errmsg;
    } errmsgs[] = {
        { FR_OK,                    "OK" },
        { FR_DISK_ERR,              "DISK_ERR" },
        { FR_INT_ERR,               "INT_ERR" },
        { FR_NOT_READY,             "NOT_READY" },
        { FR_NO_FILE,               "NO_FILE" },
        { FR_NO_PATH,               "NO_PATH" },
        { FR_INVALID_NAME,          "INVALID_NAME" },
        { FR_DENIED,                "DENIED" },
        { FR_EXIST,                 "EXIST" },
        { FR_INVALID_OBJECT,        "INVALID_OBJECT" },
        { FR_WRITE_PROTECTED,       "WRITE_PROTECTED" },
        { FR_TIMEOUT,               "TIMEOUT" },
        { FR_TOO_MANY_OPEN_FILES,   "TOO_MANY_OPEN_FILES" },
        { FR_INVALID_PARAMETER,     "INVALID_PARAMETER" },
    };

    int i;
    for (i = 0; i < UTIL_ARRAYSIZEOF(errmsgs); i++) {
        if (errmsgs[i].fres == fres)
            break;
    }

    if (i < UTIL_ARRAYSIZEOF(errmsgs)) {
        printf("%s, %s\n\r", msg, errmsgs[i].errmsg);
    } else {
        printf("%s, %d\n\r", msg, fres);
    }
}

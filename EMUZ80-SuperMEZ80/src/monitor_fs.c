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
#include <string.h>
#include <modem_xfer.h>
#include <stdarg.h>
#include <utils.h>

#define mon_fatfs_error(fr, msg) util_fatfs_error(fr, msg)

int mon_cmd_send(int argc, char *args[])
{
    FRESULT fr;
    FILINFO fileinfo;
    FIL *filep;

    if (args[0] == NULL || *args[0] == '\0') {
        printf("usage: send file\n\r");
        return MON_CMD_OK;
    }

    fr = f_stat(args[0], &fileinfo);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_stat() failed");
        return MON_CMD_OK;
    }

    filep = get_file();
    if (filep == NULL) {
        printf("too many files\n\r");
        return MON_CMD_OK;
    }
    fr = f_open(filep, args[0], FA_READ);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_open() failed");
        put_file(filep);
        return MON_CMD_OK;
    }

    char *file_name = strrchr(args[0], '/');
    if (file_name != NULL) {
        file_name++;
    } else {
        file_name = args[0];
    }
    printf("waiting for file transfer request via the terminal ...\n\r");
    if (modem_send_open(file_name, (uint32_t)fileinfo.fsize) != 0) {
        printf("modem_send_open() failed\n\r");
        put_file(filep);
        return MON_CMD_OK;
    }

    uint32_t xfer_size = 0;
    while (xfer_size < (uint32_t)fileinfo.fsize) {
        UINT n;
        fr = f_read(filep, modem_buf, MODEM_XFER_BUF_SIZE, &n);
        if (fr != FR_OK ||
            (n != MODEM_XFER_BUF_SIZE && xfer_size + n != (uint32_t)fileinfo.fsize)) {
            modem_xfer_printf(MODEM_XFER_LOG_ERROR, "read(%s) failed at %lu/%lu\n\r",
                              args[0], (unsigned long)xfer_size, (unsigned long)fileinfo.fsize);
            modem_cancel();
            break;
        }
        if (n < MODEM_XFER_BUF_SIZE) {
            memset(&modem_buf[n], 0x00, MODEM_XFER_BUF_SIZE - n);
        }
        if (modem_send() != 0) {
            break;
        }
        xfer_size += MODEM_XFER_BUF_SIZE;
    }

    modem_close();
    put_file(filep);

    return MON_CMD_OK;
}

int mon_cmd_recv(int argc, char *args[])
{
    if (modem_recv_open() != 0) {
        printf("modem_recv_open() failed\n\r");
        return MON_CMD_OK;
    }
    if (modem_recv_to_save() != 0) {
        printf("\n\rmodem_recv_to_save() failed\n\r");
    }
    modem_close();

    return MON_CMD_OK;
}

int mon_cmd_pwd(int argc, char *args[])
{
    FRESULT fr;
    const unsigned char bufsize = 128;
    char *buf = util_memalloc(bufsize);

    fr = f_getcwd(buf, bufsize);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_getcwd() failed");
    } else {
        printf("%s\n\r", buf);
    }

    util_memfree(buf);

    return MON_CMD_OK;
}

int mon_cmd_cd(int argc, char *args[])
{
    FRESULT fr;
    char *dir = "/";

    if (args[0] != NULL && *args[0] != '\0')
        dir = args[0];

    fr = f_chdir(dir);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_chdir() failed");
        return MON_CMD_OK;
    }

    return MON_CMD_OK;
}

static void show_fileinfo(FILINFO *fileinfo, uint8_t in_detail)
{
    if (in_detail) {
        printf("%cr%c %c%c %8ld %u-%02u-%02u %02u:%02u %s\n\r",
               (fileinfo->fattrib & AM_DIR) ? 'd' : '-',
               (fileinfo->fattrib & AM_RDO) ? '-' : 'w',
               (fileinfo->fattrib & AM_HID) ? 'h' : '-',
               (fileinfo->fattrib & AM_SYS) ? 's' : '-',
               (uint32_t)fileinfo->fsize,
               (fileinfo->fdate >> 9) + 1980,
               (fileinfo->fdate >> 5) & 15,
               fileinfo->fdate & 31,
               fileinfo->ftime >> 11,
               (fileinfo->ftime >> 5) & 63,
               fileinfo->fname);
    } else {
        printf("%s%s\n\r", fileinfo->fname, (fileinfo->fattrib & AM_DIR) ? "/" : "");
    }
}

int mon_cmd_ls(int argc, char *args[])
{
    FRESULT fr;
    DIR fsdir;
    FILINFO fileinfo;
    uint8_t in_detail = 0;
    char *dir = ".";

    int i = 0;
    if (args[i] != NULL && strcmp(args[i], "-l") == 0) {
        i++;
        in_detail = 1;
    }
    if (args[i] != NULL && *args[i] != '\0') {
        dir = args[i];
    }

    if (strcmp(dir, ".") != 0) {
        fr = f_stat(dir, &fileinfo);
        if (fr != FR_OK) {
            mon_fatfs_error(fr, "f_stat() failed");
            return MON_CMD_OK;
        }
        if (!(fileinfo.fattrib & AM_DIR)) {
            show_fileinfo(&fileinfo, in_detail);
            return MON_CMD_OK;
        }
    }

    fr = f_opendir(&fsdir, dir);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_opendir() failed");
        return MON_CMD_OK;
    }

    while ((fr = f_readdir(&fsdir, &fileinfo)) == FR_OK && fileinfo.fname[0] != 0) {
        show_fileinfo(&fileinfo, in_detail);
    }
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_readdir() failed");
    }
    fr = f_closedir(&fsdir);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_closedir() failed");
    }

    return MON_CMD_OK;
}

int mon_cmd_mkdir(int argc, char *args[])
{
    FRESULT fr;

    if (args[0] == NULL || *args[0] == '\0') {
        printf("usage: mkdir directory\n\r");
        return MON_CMD_OK;
    }

    fr = f_mkdir(args[0]);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_mkdir() failed");
        return MON_CMD_OK;
    }

    return MON_CMD_OK;
}

int mon_cmd_rm(int argc, char *args[])
{
    FRESULT fr;
    DIR fsdir;
    FILINFO fileinfo;
    uint8_t recursive = 0;
    char *file;

    int i = 0;
    if (args[i] != NULL && strcmp(args[i], "-r") == 0) {
        i++;
        recursive = 1;
    }
    if (args[i] == NULL || args[i] == 0) {
        printf("usage: rm [-r] file or directory\n\r");
        return MON_CMD_OK;
    }
    file = args[i];

    int nest;
    int removed;
    if (recursive) {
    redo:
        nest = 1;
        for (char *p = file; *p != '\0'; p++) {
            if (*p == '/') {
                nest++;
            }
        }
        fr = f_chdir(file);
        if (fr != FR_OK) {
            mon_fatfs_error(fr, "f_chdir() failed");
            return MON_CMD_OK;
        }

    remove_files_in_current_directory:
        removed = 0;
        fr = f_opendir(&fsdir, ".");
        if (fr != FR_OK) {
            mon_fatfs_error(fr, "f_opendir() failed");
            return MON_CMD_OK;
        }

        while (1) {
            fr = f_readdir(&fsdir, &fileinfo);
            if (fr != FR_OK) {
                mon_fatfs_error(fr, "f_readdir() failed");
                break;
            }
            if (fileinfo.fname[0] == 0) {
                break;
            }
            removed++;
            fr = f_unlink(fileinfo.fname);
            if (fr != FR_OK) {
                if (!(fileinfo.fattrib & AM_DIR)) {
                    mon_fatfs_error(fr, "f_unlink() failed");
                    break;
                }
                fr = f_closedir(&fsdir);
                if (fr != FR_OK) {
                    mon_fatfs_error(fr, "f_closedir() failed");
                    break;
                }
                fr = f_chdir(fileinfo.fname);
                if (fr != FR_OK) {
                    mon_fatfs_error(fr, "f_chdir() failed");
                    break;
                }
                nest++;
                goto remove_files_in_current_directory;
            }
        }
        fr = f_closedir(&fsdir);
        if (fr != FR_OK) {
            mon_fatfs_error(fr, "f_closedir() failed");
        }

        for (int i = 0; i < nest; i++) {
            fr = f_chdir("..");
            if (fr != FR_OK) {
                mon_fatfs_error(fr, "f_chdir() failed");
                return MON_CMD_OK;
            }
        }
        if (removed) {
            goto redo;
        }
    }

    fr = f_unlink(file);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_unlink() failed");
    }

    return MON_CMD_OK;
}

int mon_cmd_mv(int argc, char *args[])
{
    FRESULT fr;

    if (args[0] == NULL || *args[0] == '\0' || args[1] == NULL || *args[1] == '\0') {
        printf("usage: mv old_name new_name\n\r");
        return MON_CMD_OK;
    }

    fr = f_rename(args[0], args[1]);
    if (fr != FR_OK) {
        mon_fatfs_error(fr, "f_rename() failed");
        return MON_CMD_OK;
    }

    return MON_CMD_OK;
}

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
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <sys/select.h>
#include <string.h>
#include <stdarg.h>

static int tx_fd = -1;
static int rx_fd = -1;
uint32_t prev_random = 654321;
uint32_t tx_error_rate;
uint32_t rx_error_rate;

static void own_srand(uint32_t seed) {
    prev_random = seed;
}

static uint32_t own_rand() {
    return prev_random = prev_random * 1664525U + 1013904223U;
}

static int open_fifo(void)
{
    const char *TX = "/tmp/modem_test-tx";
    const char *RX = "/tmp/modem_test-rx";

    mkfifo(TX, 0660);
    tx_fd = open(TX, O_RDWR);
    if(tx_fd < 0) {
        printf("open(%s) failed (errno=%d)\n", TX, errno);
    }
    mkfifo(RX, 0660);
    rx_fd = open(RX, O_RDWR);
    if(rx_fd < 0) {
        printf("open(%s) failed (errno=%d)\n", RX, errno);
    }

    return (0 <= tx_fd) && (0 <= rx_fd) ? 0 : -1;
}

static int open_socket(int port)
{
    int listen_fd = 0;
    struct sockaddr_in addr;

    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    memset(&addr, '0', sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(port);

    int optval = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));
    bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(listen_fd, 1);
    printf("listen on port ... %d\n", port);
    tx_fd = accept(listen_fd, (struct sockaddr*)NULL, NULL);
    if(tx_fd < 0) {
        printf("create raw TCP port failed\n");
    } else {
        printf("connection established\n");

    }
    rx_fd = tx_fd;

    return (0 <= tx_fd) ? 0 : -1;
}

static void close_port(void)
{
    if (0 <= tx_fd)
        close(tx_fd);
    if (0 <= rx_fd)
        close(rx_fd);
}

int modem_xfer_tx(uint8_t c)
{
    int res;

    if (tx_error_rate && (own_rand() % tx_error_rate) == 0) {
        printf(" ** %s: TX error injected\n", __func__);
        c = (uint8_t)own_rand();
    }

    res = write(tx_fd, &c, 1);
    if (res < 0) {
        return -errno;
    }

    return 1;
}

int modem_xfer_rx(uint8_t *c, int timeout_ms)
{
    int res;
    fd_set set;
    struct timeval tv;
    FD_ZERO(&set);
    FD_SET(rx_fd, &set);
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    res = select(rx_fd + 1, &set, NULL, NULL, &tv);
    if(res < 0) {
        printf("select() failed (errno=%d)\n", errno);
        return -errno;
    }
    if(res == 0) {
        /* timeout occured */
        return 0;
    }

    res = read(rx_fd, c, 1);
    if (res < 0) {
        return -errno;
    }

    if (rx_error_rate && (own_rand() % rx_error_rate) == 0) {
        printf(" ** %s: RX error injected\n", __func__);
        *c = (uint8_t)own_rand();
    }

    return 1;
}

int modem_xfer_save(char *file_name, uint32_t offset, uint8_t *buf, uint16_t size)
{
    int res;
    char tmp[12];

    memcpy(tmp, file_name, sizeof(tmp));
    tmp[sizeof(tmp) - 1] = '\0';
    printf(" %11s %4u bytes at %6lu 0x%06lx\n", tmp, size, (unsigned long)offset,
           (unsigned long)offset);

    int fd = open(file_name, O_RDWR | O_CREAT, 0664);
    if (fd < 0) {
        printf(" %s: open('%s') failed (errno=%d)\n", __func__, file_name, errno);
        return -errno;
    }
    off_t pos = lseek(fd, offset, SEEK_SET);
    if (pos != offset) {
        printf(" %s: lseek() failed (%lu != %lu)\n", __func__, (unsigned long)pos,
               (unsigned long)offset);
        res = -EIO;
        goto close_return;
    }
    if (buf == 0 && size == 0) {
        if (ftruncate(fd, offset) != 0) {
            res = -EIO;
            goto close_return;
        }
    } else {
        if (write(fd, buf, size) != size) {
            res = -EIO;
            goto close_return;
        }
    }

    res = 0;

 close_return:
    close(fd);

    return res;
}

void modem_xfer_printf(int log_level, const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    vprintf(format, ap);
    va_end (ap);
}

int main(int ac, char *av[])
{
    uint8_t buf[MODEM_XFER_BUF_SIZE];
    int i;
    int port = -1;
    char *send_files[8];
    int num_send_files = 0;
    struct stat statbuf;
    char *p;

    for (i = 1; i < ac; i++) {
        if (av[i][0] == '-') {
            if (strcmp(av[i], "-p") == 0 || strcmp(av[i], "--port") == 0) {
                p = &av[i][0];
                if (i + 1 < ac) {
                    port = strtol(av[i + 1], &p, 0);
                }
                if (*p != '\0') {
                    printf("--port option requires network port number argument\n");
                    exit(1);
                }
                i++;
            } else
            if (strcmp(av[i], "-r") == 0 || strcmp(av[i], "--random-seed") == 0) {
                p = &av[i][0];
                if (i + 1 < ac) {
                    prev_random = strtol(av[i + 1], &p, 0);
                }
                if (*p != '\0') {
                    printf("--random-seed option requires a integer argument\n");
                    exit(1);
                }
                i++;
            } else {
                printf("unknown option %s\n", av[i]);
                exit(1);
            }
        } else {
            if (sizeof(send_files)/sizeof(*send_files) <= num_send_files) {
                printf("can't handle %s, too many files\n", av[i]);
                exit(1);
            }
            if (stat(av[i], &statbuf) != 0) {
                printf("can't get status of %s\n", av[i]);
                exit(1);
            }
            if ((statbuf.st_mode & S_IFMT) != S_IFREG) {
                printf("%s is not a regular file\n", av[i]);
                exit(1);
            }
            send_files[num_send_files++] = av[i];
        }
    }

    if (0 <= port) {
        if (open_socket(port) != 0) {
            printf("open_socket() failed\n");
            exit(1);
        }
    } else
    if (open_fifo() != 0) {
        printf("open_fifo() failed\n");
        exit(1);
    }

    if (num_send_files == 0) {
        tx_error_rate = 100;
        rx_error_rate = 500;
        if (ymodem_receive(buf) != 0) {
            printf("ymodem_receive() failed\n");
        }
    } else {
        ymodem_context ctx;
        uint8_t buf[MODEM_XFER_BUF_SIZE];
        int fd;
        int res;

        tx_error_rate = 500;
        rx_error_rate = 100;
        ymodem_send_init(&ctx, buf);
        for (i = 0; i < num_send_files; i++) {
            if (stat(send_files[i], &statbuf) != 0) {
                printf("can't get status of %s\n", av[i]);
                ymodem_send_cancel(&ctx);
                exit(1);
            }
            fd = open(send_files[i], O_RDONLY);
            if (fd < 0) {
                printf("can't open file %s\n", send_files[i]);
                ymodem_send_cancel(&ctx);
                exit(1);
            }
            char *file_name = strrchr(send_files[i], '/');
            if (file_name != NULL) {
                file_name++;
            } else {
                file_name = send_files[i];
            }
            res = ymodem_send_header(&ctx, file_name, (uint32_t)statbuf.st_size);
            if (res != MODEM_XFER_RES_OK) {
                printf("ymodem_send_header() failed, %d\n", res);
                exit(1);
            }
            uint32_t xfer_size = 0;
            while (xfer_size < (uint32_t)statbuf.st_size) {
                int n = read(fd, buf, MODEM_XFER_BUF_SIZE);
                if (n != MODEM_XFER_BUF_SIZE && xfer_size + n != (uint32_t)statbuf.st_size) {
                    printf("read(%s) failed at %lu/%lu\n", send_files[i], (unsigned long)xfer_size,
                           (unsigned long)statbuf.st_size);
                    ymodem_send_cancel(&ctx);
                    exit(1);
                }
                res = ymodem_send_block(&ctx);
                if (res != MODEM_XFER_RES_OK) {
                    printf("ymodem_send_block() failed, %d\n", res);
                    exit(1);
                }
                xfer_size += MODEM_XFER_BUF_SIZE;
            }
        }
        res = ymodem_send_end(&ctx);
        if (res != MODEM_XFER_RES_OK) {
            printf("ymodem_send_end() failed, %d\n", res);
        }
    }
    close_port();

    return 0;
}

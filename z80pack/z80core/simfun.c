/*
 * Z80SIM  -  a Z80-CPU simulator
 *
 * Copyright (C) 1987-2024 Udo Munk
 * Copyright (C) 2021 David McNaughton
 */

/*
 *	This module contains some commonly used functions.
 *	Some functions use the POSIX API available on workstations running
 *	OSX, Linux, BSD, Windows and so on, it is not usable on bare metal
 *	machines.
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "sim.h"
#include "simdefs.h"
#include "simglb.h"
#include "simmem.h"
#include "simport.h"
#include "simfun.h"

/* #define LOG_LOCAL_LEVEL LOG_DEBUG */
#include "log.h"
static const char *TAG = "func";

#define BUFSIZE	256		/* buffer size for file I/O */

static bool load_mos(char *fn, WORD start, int size);
static bool load_hex(char *fn, WORD start, int size);

/*
 *	Sleep for time microseconds, 999999 max
 */
void sleep_for_us(unsigned long time)
{
	struct timespec timer, rem;
	int err;

	timer.tv_sec = 0;
	timer.tv_nsec = 1000L * time;

again:
	if (nanosleep(&timer, &rem) == -1) {
		if ((err = errno) == EINTR) {
			/* interrupted, resume with the remaining time */
			if (rem.tv_nsec > 0L) {
				timer = rem;
				goto again;
			}
		} else {
			/* some error */
			LOGD(TAG, "sleep_us(%ld) %s", time, strerror(err));
			// cpu_error = IOERROR;
			// cpu_state = STOPPED;
		}
	}
}

/*
 *	Sleep for time milliseconds, 999 max
 */
void sleep_for_ms(unsigned time)
{
	struct timespec timer, rem;
	int err;

	timer.tv_sec = 0;
	timer.tv_nsec = 1000000L * time;

again:
	if (nanosleep(&timer, &rem) == -1) {
		if ((err = errno) == EINTR) {
			/* interrupted, resume with the remaining time */
			if (rem.tv_nsec > 0L) {
				timer = rem;
				goto again;
			}
		} else {
			/* some error */
			LOGD(TAG, "sleep_ms(%d) %s", time, strerror(err));
			// cpu_error = IOERROR;
			// cpu_state = STOPPED;
		}
	}
}

/*
 *	returns monotonic time in microseconds
 */
uint64_t get_clock_us(void)
{
	struct timeval tv;
	uint64_t t;
	static uint64_t old_t;

	gettimeofday(&tv, NULL);
	t = (uint64_t) tv.tv_sec * 1000000 + (uint64_t) tv.tv_usec;
	if (t < old_t) {
		LOGD(TAG, "get_clock_us() time jumped backwards %"
		     PRIu64 " > %" PRIu64, old_t, t);
		t = old_t;
	} else
		old_t = t;
	return t;
}

#ifdef WANT_ICE
/*
 *	Read an ICE command line from stdin.
 *	Sets buf to an empty string on EOF and signal interruptions.
 */
bool get_cmdline(char *buf, int len)
{
	bool res = true;

	if (fgets(buf, len, stdin) == NULL) {
		putchar('\n');
		if (isatty(fileno(stdin))) {
			clearerr(stdin);
			*buf = '\0';
		} else
			res = false;
	}
	return res;
}
#endif

/*
 *	Read a file into the memory of the emulated CPU.
 *	The following file formats are supported:
 *
 *		binary images with Mostek header
 *		Intel HEX
 *
 *	size == 0	don't care where it is loaded
 *	size < 0	try to load it at address "start"
 *	size > 0	check it is loaded in memory range
 *			"start" and "start + size - 1"
 */
bool load_file(char *fn, WORD start, int size)
{
	BYTE d;
	int fd, n;

	if (strlen(fn) == 0) {
		LOGE(TAG, "no input file given");
		return false;
	}

	if ((fd = open(fn, O_RDONLY)) == -1) {
		LOGE(TAG, "can't open file %s", fn);
		return false;
	}

	n = read(fd, (char *) &d, 1);	/* read first byte of file */
	close(fd);
	if (n != 1) {
		LOGE(TAG, "invalid file %s", fn);
		return false;
	}

	if (size > 0)
		LOGD(TAG, "LOAD in Range: %04Xh - %04Xh",
		     start, start + size - 1);

	if (d == 0xff) {		/* Mostek header ? */
		return load_mos(fn, start, size);
	} else {
		return load_hex(fn, start, size);
	}
}

/*
 *	Loader for binary images with Mostek header.
 *	Format of the first 3 bytes:
 *
 *	0xff ll lh
 *
 *	ll = load address low
 *	lh = load address high
 */
static bool load_mos(char *fn, WORD start, int size)
{
	register int i;
	int c, c2;
	FILE *fp;
	int laddr, count;

	if ((fp = fopen(fn, "r")) == NULL) {
		LOGE(TAG, "can't open file %s", fn);
		return false;
	}

	/* read load address */
	if ((c = getc(fp)) == EOF || c != 0xff
	    || (c = getc(fp)) == EOF || (c2 = getc(fp)) == EOF) {
		LOGE(TAG, "invalid Mostek file %s", fn);
		fclose(fp);
		return false;
	}
	laddr = (c2 << 8) | c;

	if (size < 0)
		laddr = start;
	if (size > 0 && laddr < start) {
		LOGW(TAG, "tried to load Mostek file outside "
		     "expected address range. Address: %04X", laddr);
		fclose(fp);
		return false;
	}

	count = 0;
	for (i = laddr; i < 65536; i++) {
		if ((c = getc(fp)) != EOF) {
			if (size > 0 && i >= (start + size)) {
				LOGW(TAG, "tried to load Mostek file outside "
				     "expected address range. "
				     "Address: %04X", i);
				fclose(fp);
				return false;
			}
			count++;
			putmem(i, (BYTE) c);
		} else {
			break;
		}
	}

	fclose(fp);

	PC = laddr;

	LOG(TAG, "Loader statistics for file %s:\r\n", fn);
	LOG(TAG, "START : %04XH\r\n", laddr);
	LOG(TAG, "END   : %04XH\r\n", laddr + count - 1);
	LOG(TAG, "PC    : %04XH\r\n", PC);
	LOG(TAG, "LOADED: %04XH (%d)\r\n\r\n", count, count);

	return true;
}

/*
 *	Loader for Intel HEX
 */
static bool load_hex(char *fn, WORD start, int size)
{
	register char *s;
	register BYTE *p;
	register int i;
	FILE *fp;
	char inbuf[BUFSIZE];
	BYTE outbuf[BUFSIZE / 2];
	char *s0;
	int count, n;
	int addr = 0;
	int saddr = 0xffff;
	int eaddr = 0;
	int chksum;

	if ((fp = fopen(fn, "r")) == NULL) {
		LOGE(TAG, "can't open file %s", fn);
		return false;
	}

	while (fgets(inbuf, BUFSIZE, fp) != NULL) {
		s = inbuf;
		while (isspace((unsigned char) *s))
			s++;
		if (*s != ':')
			continue;
		s0 = s++;

		p = outbuf;
		n = 0;
		chksum = 0;
		while (*s != '\r' && *s != '\n' && *s != '\0') {
			if (!(*s >= '0' && *s <= '9')
			    && !(*s >= 'A' && *s <= 'F')) {
				LOGE(TAG, "invalid character in "
				     "HEX record %s", s0);
				fclose(fp);
				return false;
			}
			*p = (*s <= '9' ? *s - '0' : *s - 'A' + 10) << 4;
			s++;
			if (*s == '\r' || *s == '\n' || *s == '\0') {
				LOGE(TAG, "odd number of characters in "
				     "HEX record %s", s0);
				fclose(fp);
				return false;
			} else if (!(*s >= '0' && *s <= '9')
				   && !(*s >= 'A' && *s <= 'F')) {
				LOGE(TAG, "invalid character in "
				     "HEX record %s", s0);
				fclose(fp);
				return false;
			}
			*p |= (*s <= '9' ? *s - '0' : *s - 'A' + 10);
			s++;
			chksum += *p++;
			n++;
		}
		if (n < 5) {
			LOGE(TAG, "invalid HEX record %s", s0);
			fclose(fp);
			return false;
		}
		if ((chksum & 255) != 0) {
			LOGE(TAG, "invalid checksum in HEX record %s", s0);
			fclose(fp);
			return false;
		}

		p = outbuf;
		count = *p++;
		if (count + 5 != n) {
			LOGE(TAG, "invalid count in HEX record %s", s0);
			fclose(fp);
			return false;
		}
		addr = *p++;
		addr = (addr << 8) | *p++;
		if (*p++ == 1)
			break;

		if (size > 0) {
			if (addr < start
			    || (addr + count - 1) >= (start + size)) {
				LOGW(TAG, "tried to load HEX record outside "
				     "expected address range. "
				     "Address: %04X-%04X",
				     addr, addr + count - 1);
				fclose(fp);
				return false;
			}
		}

		if (addr < saddr)
			saddr = addr;
		if (addr >= eaddr)
			eaddr = addr + count - 1;
		for (i = 0; i < count; i++)
			putmem(addr + i, *p++);
		addr = 0;
	}

	fclose(fp);
	if (saddr > eaddr)
		saddr = eaddr = count = 0;
	else
		count = eaddr - saddr + 1;
	PC = (addr != 0 ? addr : saddr);
	LOG(TAG, "Loader statistics for file %s:\r\n", fn);
	LOG(TAG, "START : %04XH\r\n", saddr);
	LOG(TAG, "END   : %04XH\r\n", eaddr);
	LOG(TAG, "PC    : %04XH\r\n", PC);
	LOG(TAG, "LOADED: %04XH (%d)\r\n\r\n", count & 0xffff, count & 0xffff);

	return true;
}

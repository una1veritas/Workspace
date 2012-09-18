/**
  ******************************************************************************
  * @file    lib_std/UTIL/src/term_io.c
  * @author  Martin Thomas, ChaN, Yasuo Kawachi
  * @version V1.0.0
  * @date    04/15/2009
  * @brief   Main program body
  ******************************************************************************
  * @copy
  *
  * This library is made by Martin Thomas and Chan. Yasuo Kawachi made small
  * modification to it.
  *
  * Copyright 2008-2009 Martin Thomas, Chan. and Yasuo Kawachi All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *  1. Redistributions of source code must retain the above copyright notice,
  *  this list of conditions and the following disclaimer.
  *  2. Redistributions in binary form must reproduce the above copyright notice,
  *  this list of conditions and the following disclaimer in the documentation
  *  and/or other materials provided with the distribution.
  *  3. Neither the name of the copyright holders nor the names of contributors
  *  may be used to endorse or promote products derived from this software
  *  without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY YASUO KAWACHI "AS IS" AND ANY EXPRESS OR IMPLIE  D
  * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
  * EVENT SHALL YASUO KAWACHI OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  */

#include <stdarg.h>
#include "term_io.h"

/**
  * @brief Convert ASCII string to integral value
  * @param srt: string to be converted. this is a pointer to a pointer of string.
  * @param res: pointer to string to contain converted result
  * @retval : converting result. 1 is PASSED. 0 is FAILED
  */
int xatoi (char **str, long *res)
{
	DWORD val;
	BYTE c, radix, s = 0;

// decide signedness
	while ((c = **str) == ' ') (*str)++;
	if (c == '-') {
		s = 1;
		c = *(++(*str));
	}
// decide radix(decimal, hexadecimal...)
	if (c == '0') {
		c = *(++(*str));
		if (c <= ' ') {
			*res = 0; return 1;
		}
		if (c == 'x') {
			radix = 16;
			c = *(++(*str));
		} else {
			if (c == 'b') {
				radix = 2;
				c = *(++(*str));
			} else {
				if ((c >= '0')&&(c <= '9'))
					radix = 8;
				else
					return 0;
			}
		}
	} else {
		if ((c < '1')||(c > '9'))
			return 0;
		radix = 10;
	}
// parse string
	val = 0;
	while (c > ' ') {
		if (c >= 'a') c -= 0x20;
		c -= '0';
		if (c >= 17) {
			c -= 7;
			if (c <= 9) return 0;
		}
		if (c >= radix) return 0;
		val = val * radix + c;
		c = *(++(*str));
	}
	if (s) val = -val;
	*res = val;
	return 1;
}

/**
  * @brief print function for teminal. if the byte is LF, it prints CR+LF
  * @param n : byte to be printed
  * @retval : None
  */
void xputc (char c)
{
	if (c == '\n') comm_put('\r');
	comm_put(c);
}

/**
  * @brief Print string constant using xputc function
  * @param srt: string to be printed.
  * @retval : None
  */
void xputs (const char* str)
{
	while (*str)
		xputc(*str++);
}

/**
  * @brief print integral argument
  * @param val : integral value to be printed
  * @param radix : radix when printed
  * @param len : length of columns printed
  * @retval : None
  */
void xitoa (long val, int radix, int len)
{
	BYTE c, r, sgn = 0, pad = ' ';
	BYTE s[20], i = 0;
	DWORD v;


	if (radix < 0) {
		radix = -radix;
		if (val < 0) {
			val = -val;
			sgn = '-';
		}
	}
	v = val;
	r = radix;
	if (len < 0) {
		len = -len;
		pad = '0';
	}
	if (len > 20) return;
	do {
		c = (BYTE)(v % r);
		if (c >= 10) c += 7;
		c += '0';
		s[i++] = c;
		v /= r;
	} while (v);
	if (sgn) s[i++] = sgn;
	while (i < len)
		s[i++] = pad;
	do
		xputc(s[--i]);
	while (i);
}

/**
  * @brief famous printf function for terminal
  * @param str : string to be printed
  * @param successive arguments : embedded value in string. refer printf manual
  * @retval : None
  */
void xprintf (const char* str, ...)
{
	va_list arp;
	int d, r, w, s, l;


	va_start(arp, str);

	while ((d = *str++) != 0) {
		if (d != '%') {
			xputc(d); continue;
		}
		d = *str++; w = r = s = l = 0;
		if (d == '0') {
			d = *str++; s = 1;
		}
		while ((d >= '0')&&(d <= '9')) {
			w += w * 10 + (d - '0');
			d = *str++;
		}
		if (s) w = -w;
		if (d == 'l') {
			l = 1;
			d = *str++;
		}
		if (!d) break;
		if (d == 's') {
			xputs(va_arg(arp, char*));
			continue;
		}
		if (d == 'c') {
			xputc((char)va_arg(arp, int));
			continue;
		}
		if (d == 'u') r = 10;
		if (d == 'd') r = -10;
		if (d == 'X') r = 16;
		if (d == 'b') r = 2;
		if (!r) break;
		if (l) {
			xitoa((long)va_arg(arp, long), r, w);
		} else {
			if (r > 0)
				xitoa((unsigned long)va_arg(arp, int), r, w);
			else
				xitoa((long)va_arg(arp, int), r, w);
		}
	}

	va_end(arp);
}

/**
  * @brief print dump data. address, hex data, ASCII
  * @param buff : successive integral  to be printed
  * @param ofs : address
  * @param cnt : number to be printed
  * @retval : None
  */
void put_dump (const BYTE *buff, DWORD ofs, int cnt)
{
	BYTE n;


	xprintf("%08lX ", ofs);
	for(n = 0; n < cnt; n++)
		xprintf(" %02X", buff[n]);
	xputc(' ');
	for(n = 0; n < cnt; n++) {
		if ((buff[n] < 0x20)||(buff[n] >= 0x7F))
			xputc('.');
		else
			xputc(buff[n]);
	}
	xputc('\n');
}

/**
  * @brief read string from console untill return is put
  * @param buff : array to contain data
  * @param len : maximum number able to be input
  * @retval : None
  */
void get_line (char *buff, int len)
{
	char c;
	int idx = 0;

	for (;;) {
		c = xgetc();
		if (c == '\r') break;
		if ((c == '\b') && idx) {
			idx--; xputc(c);
			xputc(' '); xputc(c); // added by mthomas for Eclipse Terminal plug-in
		}
		if (((BYTE)c >= ' ') && (idx < len - 1)) {
				buff[idx++] = c; xputc(c);
		}
	}
	buff[idx] = 0;
	xputc('\n');
}

int get_line_r (char *buff, int len, int* idx)
{
	char c;
	int retval = 0;
	int myidx;

	if ( xavail() ) {
		myidx = *idx;
		c = xgetc();
		if (c == '\r') {
			buff[myidx] = 0;
			xputc('\n');
			retval = 1;
		} else {
			if ((c == '\b') && myidx) {
				myidx--; xputc(c);
				xputc(' '); xputc(c); // added by mthomas for Eclipse Terminal plug-in
			}
			if (((BYTE)c >= ' ') && (myidx < len - 1)) {
					buff[myidx++] = c; xputc(c);
			}
		}
		*idx = myidx;
	}

	return retval;
}

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
 *
 *
 * Modified and added by Akihito Honda(Aki.h @akih_san)
 *  https://twitter.com/akih_san
 *  https://github.com/akih-san
 *
 * Date. 2024.3.28 
 */

#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include "../drivers/utils.h"

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
#if 0
int util_stricmp(const char *a, const char *b)
{
  int ua, ub;
  do {
      ua = toupper((unsigned char)*a++);
      ub = toupper((unsigned char)*b++);
   } while (ua == ub && ua != '\0');
   return ua - ub;
}
#endif
// calc number of leap year from 1980 (Western calendar)
//
uint16_t chk_leap(uint16_t year) {

	uint16_t n;

	n = 0;
	if( (year%4 == 0) && (year%100 > 0) ) n=1;
	if( year%400==0 ) n=1;
	return(n);
}

static uint16_t leaps(uint16_t year) {

	uint16_t y, n;

	for(n=0,y=1980;y <=year;y++){
		if( chk_leap( y ) ) n++;
	}
	return(n);
}

const uint16_t mtod[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

static uint16_t jan_to_days(uint16_t month, uint16_t day) {
	
	uint16_t i, d;
	
	for( d=0, i=0; i<month-1; i++ ) d += mtod[i];
	d += day;
	return( d );
}

uint16_t days_from_1980(uint16_t year, uint16_t month, uint16_t day) {
	uint16_t d1, d2;
	
	d1 = leaps(year);
	if ( month == 2 && day == 29 ) d1--;
	d2 = jan_to_days(month, day);
	if (d2 <= 59 && chk_leap( year ) ) d1--;	// except d2 <= 2/28 && year == leap year

	return((year - 1980)*365 + d1 + d2 - 1 );		// except base day (1980/1/1)
}

uint8_t cnv_bcd(uint8_t bval) {
	union {
		struct {
			uint8_t hex;
			uint8_t bcd;
		} conv ;
		uint16_t buf ;
	} convbcd ;

	uint8_t bitcnt;

	convbcd.buf = 0;
	convbcd.conv.hex = bval;

	for (bitcnt = 0 ; bitcnt < 8 ; bitcnt++) {
		if (((convbcd.conv.bcd & 0x0f) + 0x03) >= 0x08) convbcd.conv.bcd += 0x03;
		if (((convbcd.conv.bcd & 0xf0) + 0x30) >= 0x80) convbcd.conv.bcd += 0x30;
		convbcd.buf <<= 1;
	}
	return convbcd.conv.bcd;
}

uint8_t cnv_byte(uint8_t bval) {

	uint8_t convbin ;

	convbin = ((bval & 0xf0) >> 4) * 10 + (bval & 0x0f) ;
	return convbin ;
}

void cnv_ymd(uint16_t n_date, uint16_t *year, uint16_t *month, uint16_t *date ) {
	uint16_t remain_date;
	uint16_t y, m, d, leaps;

	remain_date = n_date + 1;
	y = 1980;
	for(;;) {
		leaps = chk_leap( y );
		d = 365 + leaps;
		if ( remain_date <= d ) break;
		remain_date -= d;
		y++;
	}
	if (y >= 2000) y-= 2000;
	else y-=1900;

	*year = y;
	m = 0;
	for(;;) {
		d = mtod[m];
		if ( m == 1 ) d += leaps;
		if ( remain_date <= d ) break;
		remain_date -= d;
		m++;
	};
	*month = ++m;
	*date = remain_date;
}

/*
 * Copyright (c) 2016, Takashi TOYOSHIMA <toyoshim@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * - Neither the name of the authors nor the names of its contributors may be
 *   used to endorse or promote products derived from this software with out
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUE
 * NTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#include <avr/io.h>

#include "sdcard.h"
#include "uart.h"

#ifdef SDCARD_DEBUG
#define DEBUGMSG(x) uart_puts(x)
#else
#define DEBUGMSG(x)
#endif

#define PORT PORTB
#define DDR  DDRB
#define PIN  PINB
#define P_CS _BV(PB0)
#define P_CK _BV(PB1)
#define P_DI _BV(PB2)
//#define P_PU _BV(PC4)
#define P_DO _BV(PB3)

static unsigned char
buffer[512];

static unsigned short
crc = 0xffff;

static unsigned long
cur_blk = 0;

static void
sd_out
(char c)
{
  int i;
  for (i = 7; i >= 0; i--) {
    char out = (0 == (c & (1 << i)))? 0: P_DI;
    PORT = (PORT & ~(P_CK | P_DI)) | out;
    PORT |=  P_CK;
  }
}

static int
sd_busy
(char f)
{
  unsigned long timeout = 0xffff;
  for (; 0 != timeout; timeout--) {
    char c;
    PORT &= ~P_CK;
    c = PIN;
    PORT |=  P_CK;
    if ((f & P_DO) == (c & P_DO)) return 0;
  }
  return -1;
}

static char
sd_in
(char n)
{
  int i;
  int rc = 0;
  for (i = 0; i < n; i++) {
    char c;
    PORT &= ~P_CK;
    c = PIN;
    PORT |= P_CK;
    rc <<= 1;
    if (0 != (c & P_DO)) rc |= 1;
  }
  return rc;
}

static char
sd_cmd
(char cmd, char arg0, char arg1, char arg2, char arg3, char crc)
{
  PORT = (PORT | P_DI | P_CK) & ~P_CS;
  sd_out(cmd);
  sd_out(arg0);
  sd_out(arg1);
  sd_out(arg2);
  sd_out(arg3);
  sd_out(crc);
  PORT |= P_DI;
  if (sd_busy(0) < 0) return -1;
  return sd_in(7);
}

void
sdcard_init
(void)
{
  /*
   * PC2: CK out
   * PC3: DI out
   * PC4: DO in
   * PC5: CS out
   */
  // Port Settings
  DDR  &= ~P_DO;
  DDR  |=  (P_CK | P_DI | P_CS);
  PORT &= ~(P_CK | P_DI | P_CS);
//  PORT |=  P_PU;
  cur_blk = 0;
}

int sdcard_open(void)
{
	DEBUGMSG("sdcard_open: send 80 clocks\n");

	// native mode to SPI mode
	// at least 1ms after the power-on

	// hold DI, CS high and send at least 74 clocks,
  PORT |= P_DI | P_CS;
  int i;
  for (i = 0; i < 80; i++) {
    PORT &= ~P_CK;
    PORT |=  P_CK;
  }

  DEBUGMSG("sdcard_open: issue CMD0\n");

  // now card is waiting for commands (but not in SPI mode)
  char rc;
  // CS low, CMD0 (software reset), CRC is required
  rc = sd_cmd(0x40, 0x00, 0x00, 0x00, 0x00, 0x95);
  if (1 != rc) return -1;

  // now card is in SPI mode (CRC off in default)
  // in idle state, returns R1 response

  DEBUGMSG("sdcard_open: initializing card CMD41\n");

  // initializing card
  // in-idle-state of the response will be cleared
  // if initialization has been finished.
  // it takes up to hundreds milli secs.
  // cmd1
  unsigned short timeout = 0xffff;
  for (; timeout != 0; timeout--) {
    rc = sd_cmd(0x41, 0x00, 0x00, 0x00, 0x00, 0x00);
    if (0 == rc) break;
    if (-1 == rc) return -2;
  }
  if (0 != rc) return -3;

  DEBUGMSG("sdcard_open: finished.\n");

  PORT &= ~P_CK;
  return 0;
}

int
sdcard_fetch
(unsigned long blk_addr)
{
  // cmd17
  char rc = sd_cmd(0x51, blk_addr >> 24, blk_addr >> 16, blk_addr >> 8, blk_addr, 0x00);
  if (0 != rc) return -1;
  if (sd_busy(0) < 0) return -2;
  int i;
  for (i = 0; i < 512; i++) buffer[i] = sd_in(8);
  rc = sd_in(8);
  crc = (rc << 8) | sd_in(8);

  // XXX: rc check

  PORT &= ~P_CK;

  cur_blk = blk_addr;
  return 0;
}

int
sdcard_store
(unsigned long blk_addr)
{
  // cmd24
  char rc = sd_cmd(0x58, blk_addr >> 24, blk_addr >> 16, blk_addr >> 8, blk_addr, 0x00);
  if (0 != rc) return -1;
  sd_out(0xff); // blank 1B
  sd_out(0xfe); // Data Token
  int i;
  for (i = 0; i < 512; i++) sd_out(buffer[i]); // Data Block
  sd_out(0xff); // CRC dummy 1/2
  sd_out(0xff); // CRC dummy 2/2
  if (sd_busy(0) < 0) return -3;
  rc = sd_in(4);
  if (sd_busy(~0) < 0) return -4;

  // XXX: rc check
  // 0 0101: accepted
  // 0 1011: CRC error
  // 0 1101: write error

  PORT &= ~P_CK;

  return rc; // test result code
}

unsigned short
sdcard_crc
(void)
{
  return crc;
}

int
sdcard_flush
(void)
{
  return sdcard_store(cur_blk);
}

void *
sdcard_buffer
(void)
{
  return buffer;
}

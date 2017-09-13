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

#include <inttypes.h>
#include "sram.h"

#include <avr/io.h>

/*
 * 128k x 8bit (A0 -- A16)
 * M68AF127B
 * AS6C1008
 */

/* SRAM I/F Port/Pin definitions */
#define ADDRL_DIR DDRA
#define ADDRL PORTA
#define ADDRL_MASK 0xff
#define ADDRH_DIR DDRC
#define ADDRH PORTC
#define ADDRH_MASK 0xff
#define ADDRX_DIR DDRD
#define ADDRX PORTD
#define ADDRX_MASK (1<<PD7)

#define DATA_DIR DDRA
#define DATA_OUT  PORTA
#define DATA_IN  PINA

#define CONTROL_DIR DDRG
#define CONTROL PORTG
#define SRAM_OE (1<<PG1)
#define SRAM_WE (1<<PG0)
#define SRAM_ALE (1<<PG2)
#define CONTROL_CS_DIR DDRL
#define CONTROL_CS PORTL
#define SRAM_CS (1<<PL6)

#define bitset(port, bv)   (port) |= (bv)
#define bitclear(port, bv) (port) &= ~(bv)

void sram_init() {
  ADDRL_DIR = ADDRL_MASK;
  ADDRH_DIR = ADDRH_MASK; // fixed to be OUTPUT
  ADDRX_DIR |= ADDRX_MASK; // fixed to be OUTPUT
  CONTROL_DIR |= ( SRAM_WE | SRAM_OE | SRAM_ALE );
  CONTROL |= ( SRAM_WE | SRAM_OE | SRAM_ALE);
  CONTROL_CS_DIR |= SRAM_CS;
  CONTROL_CS |= SRAM_CS;
}

inline void addr_out(unsigned short addr) {
  CONTROL |= SRAM_ALE;    // transparent
  ADDRL_DIR = ADDRL_MASK;  // set to OUTPUT
  ADDRL = addr & ADDRL_MASK;
  CONTROL &= ~SRAM_ALE;  // Latch on this edge
  addr >>= 8;
  ADDRH = addr & ADDRH_MASK; // ADDRH is always OUTPUT
}

inline void sram_bank(uint8_t bk) {
  ADDRX &= ~ADDRX_MASK;
  if ( bk & 1 )
    ADDRX |= ADDRX_MASK;
}

unsigned char sram_read(unsigned short addr) {
  unsigned char val;
  CONTROL_CS &= ~SRAM_CS;
  addr_out(addr);
  DATA_OUT = 0xff; // clear the pull-ups
  DATA_DIR = 0x00;
  CONTROL &= ~SRAM_OE;
  CONTROL |= SRAM_OE; // valid data remains while
  val = DATA_IN;
  CONTROL_CS |= SRAM_CS;
  return val;
}

void sram_write(unsigned short addr, unsigned char data) {
  CONTROL_CS &= ~SRAM_CS;
  addr_out(addr);  // A16 is always 0
  DATA_OUT = data;
  CONTROL &= ~SRAM_WE;
  CONTROL |= SRAM_WE;
  CONTROL_CS |= SRAM_CS;
}

/* sram_avr.c end */

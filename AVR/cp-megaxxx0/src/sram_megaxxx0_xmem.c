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
 *
 * AD0-AD8 address latch HC573
 *
 */

/* SRAM I/F Port/Pin definitions */
/*
#define ADDRL_DIR DDRA
#define ADDRL PORTA
#define ADDRL_MASK 0xff
#define ADDRH_DIR DDRC
#define ADDRH PORTC
#define ADDRH_MASK 0xff

#define DATA_DIR DDRA
#define DATA_OUT  PORTA
#define DATA_IN  PINA

#define CONTROL_DIR DDRG
#define CONTROL PORTG
#define SRAM_OE (1<<PG1)
#define SRAM_WE (1<<PG0)
#define SRAM_ALE (1<<PG2)
*/
#define ADDRX_DIR DDRD
#define ADDRX PORTD
#define ADDRX_MASK (1<<PD7)

#define CONTROL_CS_DIR DDRL
#define CONTROL_CS PORTL
#define SRAM_CS (1<<PL6)


#define bitset(port, bv)   (port) |= (bv)
#define bitclear(port, bv) (port) &= ~(bv)

uint16_t * wordptr = ((uint16_t *) 0x2200);
uint8_t * byteptr = ((uint8_t *) 0x2200);

void sram_init() {

	XMCRB=0; // need all 64K. no pins released
	XMCRA=1<<SRE; // enable xmem, no wait states

	CONTROL_CS_DIR |= SRAM_CS;
	CONTROL_CS &= ~SRAM_CS;

	ADDRX_DIR |= ADDRX_MASK;
	ADDRX &= ~ADDRX_MASK;
}

inline void sram_bank(uint8_t bk) {
  ADDRX &= ~ADDRX_MASK;
  if ( bk & 1 )
    ADDRX |= ADDRX_MASK;
}

unsigned char sram_read(unsigned short addr) {
	unsigned char val;
	if ( addr < 0x4000 )
		sram_bank(1);
	else
		sram_bank(0);
	val = byteptr[addr];
	return val;
}

void sram_write(unsigned short addr, unsigned char data) {
	if ( addr < 0x4000 )
		sram_bank(1);
	else
		sram_bank(0);
	byteptr[addr] = data;
}

/* sram_avr.c end */

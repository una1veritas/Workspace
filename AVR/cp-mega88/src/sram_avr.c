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

#include "sram.h"

#include <avr/io.h>

void
sram_init
(void)
{
  /*
   * PB1: SRAM WE
   * PB2: SRAM CS
   * PD3: SRAM A16
   * PB3: SRAM OE
   * PB4: LATCH_CLK for Address Low
   * PB5: LATCH_CLK for Address High
   * PC0-3: Address / Data
   * PD4-7: Address / Data
   */
  // Port Settings
  //   All ports are output
  DDRB    =  (_BV(DDB1) | _BV(DDB2) | _BV(DDB3) | _BV(DDB4) | _BV(DDB5));
  PORTB   = 0xff;
  DDRC    = ( _BV(DDB0) | _BV(DDB1) | _BV(DDB2) | _BV(DDB3) | _BV(DDB4) );
  PORTC   = 0; // &= ~( _BV(DDB0) | _BV(DDB1) | _BV(DDB2) | _BV(DDB3) | _BV(DDB4) )
  DDRD    = (_BV(DDB3) | _BV(DDB4) | _BV(DDB5) | _BV(DDB6) |_BV(DD7) );
  PORTD   = 0;

}

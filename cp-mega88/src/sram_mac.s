/*
 * Copyright (c) 2010, Takashi TOYOSHIMA <toyoshim@gmail.com>
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

  /* Original:
   * PB0: /W
   * PB1: E2
   * PB2: A16
   * PB3: N/A (input)
   * PB4: CLK for FF on Address Low
   * PB5: CLK for FF on Address High
   * PD*: Address / Data

.equ	DOUT,11
.equ	DDIR,10
.equ	DIN,9
.equ	CTRL,5

.equ	W_X,0
.equ	SEL,1
.equ	A16,2
.equ	C_L,4
.equ	C_H,5

.equ	C1,(1 << C_L)
.equ	C2,(1 << C_H)
.equ	RD,((1 << W_X) | (1 << SEL))
.equ	WR,(1 << SEL)

.equ	DATA_IN,0
.equ	DATA_OUT,0xff
   */

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

.equ 	DOUT,   0x08
.equ	DOUT_H, 0x0B
.equ 	DDIR, 	0x07
.equ	DDIR_H,	0x0A
.equ 	DIN, 	0x06
.equ 	DIN_H,	0x09
.equ 	CTRL,	0x05

.equ	W_X, 1
.equ	SEL, 2
.equ 	O_X, 3
.equ	A16,3
.equ	C_L,4
.equ	C_H,5

.equ	C1,(1 << C_L)
.equ	C2,(1 << C_H)
.equ	RD,((1 << O_X) | (1 << SEL))
.equ	WR,((1 << W_X) | (1 << SEL))
.equ 	CS, (1<<SEL)

.equ	DATA_IN,0
.equ	DATA_OUT,0xff
.equ 	DATA_HIGH, 0xf0
.equ 	DATA_LOW, 0x0f

.macro _sram_read adr_l=r24, adr_h=r25, ret=r24, work=r23
	out DOUT, \adr_l/* output address low   */
	in	\work, DOUT_H
	andi \work, 0x0f
	andi \adr_l, 0xf0
	or 	\adr_l, \work
	out DOUT_H, \adr_l
	ldi \work, C1
	out CTRL, \work	/* assert CLK for Latch */
	out DOUT, \adr_h/* output address high  */
	ldi \work, C2
	out CTRL, \work	/* assert CLK for Latch */
	out DDIR, r1	/* data port as input   */
	ldi \work, RD
	out CTRL, \work	/* assert /W and E2     */
	out DOUT, r1	/* disable pull-ups     */
	ldi \work, DATA_OUT
	out CTRL, r1	/* negate signals       */
	in \ret, DIN	/* read data            */
	out DDIR, \work	/* data port as output  */
.endm

.macro _sram_write adr_l=r24, adr_h=r25, data=r22, work=r23
	out DOUT, \adr_l/* output address low   */
	ldi \work, C1
	out CTRL, \work	/* assert CLK for Latch */
	out DOUT, \adr_h/* output address high  */
	ldi \work, C2
	out CTRL, \work	/* assert CLK for Latch */
	out DOUT, \data	/* output data          */
	ldi \work, WR
	out CTRL, \work	/* assert E2            */
	out CTRL, r1	/* negate signals       */
.endm

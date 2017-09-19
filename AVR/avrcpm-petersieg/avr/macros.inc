;    Commonly used macros
;
;    Copyright (C) 2010 Leo C.
;
;    This file is part of avrcpm.
;
;    avrcpm is free software: you can redistribute it and/or modify it
;    under the terms of the GNU General Public License as published by
;    the Free Software Foundation, either version 3 of the License, or
;    (at your option) any later version.
;
;    avrcpm is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with avrcpm.  If not, see <http://www.gnu.org/licenses/>.
;
;    $Id: macros.inc 169 2012-03-10 13:43:34Z leo $
;

;------------------------------------------------
; 
.macro	outm8
.if	@0 > 0x3f
	sts	@0,@1
.else
	out	@0,@1
.endif
.endm

;------------------------------------------------
; 
.macro	inm8
.if	@1 > 0x3f
	lds	@0,@1
.else
	in	@0,@1
.endif
.endm

;------------------------------------------------
; 
.macro	sbism8
.if	@0 > 0x1f
	in	_tmp0,@0
	sbrs	_tmp0,@1
.else
	sbis	@0,@1
.endif
.endm

;------------------------------------------------
; load 16 bit constant to register pair

.macro	ldiw
	ldi    @0l, low(@1)  
	ldi    @0h, high(@1)
.endm

;------------------------------------------------
; load 16 bit direct from data space

.macro	ldsw
	lds    @0l, @1 
	lds    @0h, @1+1
.endm

;------------------------------------------------
; store 16 bit direct to data space

.macro	stsw
	sts    @0,  @1l 
	sts    @0+1,@1h
.endm

;------------------------------------------------
; add 16 bit constant to register pair

.macro	addiw
	subi    @0l, low(-@1)  
	sbci    @0h, high(-@1)
.endm

;------------------------------------------------
; sub 16 bit constant from register pair

.macro	subiw
	subi    @0l, low(@1)  
	sbci    @0h, high(@1)
.endm

;------------------------------------------------
; Move single bit between two registers
;
;	bmov	dstreg,dstbit,srcreg.srcbit

.macro	bmov
	bst	@2,@3
	bld	@0,@1
.endm

;------------------------------------------------
; 
;
;
.macro  INTERRUPT
  .set pos_ = PC
  .org	@0			; vector address
  .if abs(pos_ - PC) > 2048
  	jmp     pos_
  .else
 	rjmp    pos_		; jump to handler
  .endif
  .org	pos_			; restore PC
.endm

;------------------------------------------------
; 
;
.macro  ljmp
  .if FLASHEND > 0x0fff
    .ifdef @0
      .if abs(PC - @0) > 2047
  	jmp	@0
      .else
      	rjmp	@0
      .endif
    .else
  	jmp	@0
    .endif
  .else
 	rjmp	@0
  .endif
.endm

;------------------------------------------------
;
;
.macro  lcall
  .if FLASHEND > 0x0fff
    .ifdef @0
      .if abs(PC - @0) > 2047
  	call	@0
      .else
      	rcall	@0
      .endif
    .else
  	call	@0
    .endif
  .else
 	rcall	@0
  .endif
.endm

;------------------------------------------------
; Print string.
;	printstring "String"

.macro	printstring
  .if FLASHEND > 0x0fff
	call	printstr
  .else
 	rcall	printstr
  .endif
  .if strlen(@0) % 2
    .db @0,0
  .else
    .db @0,0,0
  .endif
.endm

;------------------------------------------------
; Print newline
;	print cr, lf

.macro	printnewline
  lcall	printstr
  .db 13,0
.endm


;------------------------------------------------
; Print a Z80 flag
;	print_zflag F
;	  where F is the flag to print

.macro print_zflag

.set S_ = 'S'
.set Z_ = 'Z'
.set H_ = 'H'
.set P_ = 'P'
.set N_ = 'N'
.set C_ = 'C'

	ldi	temp, ' '
	sbrc	z_flags,ZFL_@0
	 ldi	temp, @0_
	rcall	uartputc
.endm

;------------------------------------------------
;	db_version VMAJOR, VMINOR

.macro	db_version

  .set	maj1_	= @0 / 10
  .set	maj0_	= @0 % 10
  .set	min1_	= @1 / 10
  .set	min0_	= @1 % 10

  .if maj1_
    .if min1_
      .db maj1_+'0',maj0_+'0','.',min1_+'0',min0_+'0',0
    .else
      .db maj1_+'0',maj0_+'0','.',          min0_+'0',0,0
    .endif
  .else
    .if min1_
      .db           maj0_+'0','.',min1_+'0',min0_+'0',0,0
    .else
      .db           maj0_+'0','.',          min0_+'0',0
    .endif
  .endif
.endm

; vim:set ts=8 noet nowrap


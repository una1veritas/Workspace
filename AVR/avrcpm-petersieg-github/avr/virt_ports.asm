;    Virtual Ports for the BIOS Interaction
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
;    $Id: virt_ports.asm 181 2012-03-19 19:36:07Z leo $
;


;
;   Port        Direction  Function
;hex	dez
;-------------------------------------------------------------------------
;00  	0 	in	- Con status. 
;			  Returns 0xFF if the UART has a byte, 0 otherwise.
;01  	1 	in/out	- Console input, aka UDR. / Console Output
;02  	2 	out	- Console Output (deprecated)
;03  	3	in	- "UART" status: bit 0 = rx, bit 1 = tx
;04  	4	in	- "UART" data register, no wait
;
;0D,0E	13,14	in/out	- Set address of Bios Controll Block
;0F  	15	in/out	- Disk select
;10,11  16,17 	in/out	- Track select
;12,13 	18,19 	in/out	- Sector select
;14,15 	20,21 	in/out	- Write addr
;   	 	
;16  	22 	out	- Trigger disk i/o operations
;			  Bit 7 = 1: Read sector
;			  Bit 6 = 1: Write sector
;			  Bit 5 = 1: BIOS WBOOT
;			  Bit 4 = 1: BIOS Home
;			  Only one of bits 4..7  may be set.
;			  If Write function (bit 6=1):
;			   Bits 0..2: 0 - write to allocated
;				      1 - write to directory
;				      2 - write unallocated
;				      3 - write to directory			 
;
;16  	22	in	- Result of last read/write operation.
;			  0x00 = ok, 0xff = error (--> Bad Sector)
;
;40  	64-71	in/out	- Timer/Clock control.	
;46


; ---------------------------------------------- Start of Code Segment
	.cseg
vport_tbl:
	.db	00,1		;Port 0, length 1
	.dw	conStatus	;	in
	.dw	dbgOut		;	out
	.db	01,1
	.dw	uartgetc
	.dw	uartputc
	.db	02,1		;Port 2 (old console output)
	.dw	uartgetc	; filler
	.dw	uartputc	; deprecated
	.db	03,1
	.dw	uartstat
	.dw	vport_out_dummy
	.db	04,1
	.dw	uartin
	.dw	uartout

	.db	13,9		; Port 13-21, (lenth 9)
	.dw	dsk_param_get
	.dw	dsk_param_set
	.db	22,1
	.dw	dskErrorRet
	.dw	dskDoIt

	.db	TIMERPORT,7
	.dw	clockget
	.dw	clockput

	.db	DEBUGPORT,1
	.dw	dbg_stat
	.dw	dbg_ctrl
	.db	0,0		; Stop mark

;---------------------------------------------------------------------

;Called with port in temp2 and value in temp.
portWrite:
	set
	rjmp	vprw_start

;Called with port in temp2. Should return value in temp.
portRead:
	clt

vprw_start:
	push	yh
	push	yl
.if PORT_DEBUG > 1
	tst	temp2
	brne	dvp_1		;don't debug console status
	brts	dvp_1
	rjmp	conStatus
dvp_1:
	printnewline
	brts	dvp_11
	printstring	"Port In:  "
	rjmp	dvp_12
dvp_11:	
	printstring	"Port Out: "
dvp_12:
	push	temp
	mov	temp,temp2
	rcall	printhex
	pop	temp
.endif
	ldiw	z,vport_tbl*2

vprw_loop:
	lpm	_tmp0,z+
	lpm	temp4,z+	;length
	cpi	temp4,0
	breq	vprw_exit	;no more ports

	mov	temp3,temp2	
	sub	temp3,_tmp0	;base port
	brcs	vprw_next	;port # too high
	cp	temp3,temp4     ;may be in range
	brcs	vprw_found	;
vprw_next:			;port # not in range, test next block.
	adiw	z,4
	rjmp	vprw_loop
vprw_found:
	brtc	PC+2
	adiw	z,2
	lpm	_tmp0,z+
	lpm	_tmp1,z+
	movw	z,_tmp0

.if PORT_DEBUG > 1
	push	temp2
	push	temp
	printstring ", exec: "
	movw	temp,z
	rcall	printhexw
	printstring ", rel port: "
	mov	temp,temp3
	rcall	printhex
	pop	temp
	pop	temp2
	printstring ", val: "
	brts	dvp_2
	icall
	rcall	printhex
	printstring " "
	pop	yl
	pop	yh
	ret
dvp_2:
	rcall	printhex
	printstring " "
				; relative port # in temp3
	icall
	pop	yl
	pop	yh
	ret
.else
	icall
	pop	yl
	pop	yh
	ret
.endif

vprw_exit:
				; trap for nonexistent port?
.if PORT_DEBUG > 1
	printstring ", not found!"
.endif	
vport_in_dummy:
	ldi	temp,0xff
vport_out_dummy:
	pop	yl
	pop	yh
	ret
	

uartstat:
	clr	temp
	lds	temp2,rxcount
	cpse	temp2,_0
	 sbr	temp,0x01		
	lds	temp2,txcount
	cpi	temp2,TXBUFSIZE
	breq	uartst_1
	 sbr	temp,0x02
uartst_1:
	ret

uartin:
	clr	temp
	lds	temp2,rxcount
	cpse	temp2,_0
	 rjmp	uartgetc
	ret

uartout:
	lds	temp2,txcount
	cpi	temp2,TXBUFSIZE
	breq	uartout_1
	rjmp uartputc
uartout_1:
	ret


conStatus:
	lds	temp,rxcount
	cpse	temp,_0
	 ldi	temp,0xff
	ret


dbgOut:
	printnewline
	printstring "Debug: "
	rcall printhex
	ret


dbg_stat:
	ldi	temp,0
	ret

dbg_ctrl:
	bmov	intstat,i_trace, temp,0
	ret



;---------------------------------------------------------------------
; vim:set ts=8 noet nowrap


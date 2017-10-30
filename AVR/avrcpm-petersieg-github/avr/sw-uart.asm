; Serial interface using the ATmega8/88 USART. 
; This is part of the Z80-CP/M emulator written by Sprite_tm.
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
;    $Id: sw-uart.asm 174 2012-03-13 18:52:08Z leo $
;

#define SSER_BIT_TC	(F_CPU+BAUD/2) / BAUD 

#define RXBUFMASK  	RXBUFSIZE-1
#define TXBUFMASK  	TXBUFSIZE-1

	.dseg
	
srx_state:
	.byte	1
srx_char_to:
	.byte	1
srx_dr:
	.byte	1
;srx_lastedge:
;	.byte	2
stx_bitcount:
	.byte	1
stx_dr:
	.byte	1
rxcount:
	.byte	1
rxidx_w:
	.byte	1
rxidx_r:
	.byte	1
txcount:
	.byte	1
txidx_w:
	.byte	1
txidx_r:
	.byte	1
rxfifo:
	.byte	RXBUFSIZE
txfifo:
	.byte	TXBUFSIZE


	.cseg

; Init 
uart_init:

; - Init clock/timer system and serial port

; Init timer 1 as 
; - Soft UART TX (OC1A/OCR1A).
; - Soft UART RX (ICP1/ICR1).
; - 1ms System timer is already configured at this point.

	ldi	temp,(1<<COM1A1)|(1<<COM1A0)	;OC1A high on compare match (UART TX)
	outm8	TCCR1A,temp

	ldi	temp,(1<<ICF1)			;clear pending int
	out	TIFR1,temp
	inm8	temp,TIMSK1
	ori	temp,(1<<ICIE1)			;Enable input capture int.  (UART RX)
	outm8	TIMSK1,temp
	ret
	
;------------------------------------------------------------------

	.cseg

; Timer/Counter1 Input Capture interrupt
	
	INTERRUPT ICP1addr
	
	push	temp
	in	temp,sreg
	push	temp
	push	zh
	push	zl
	inm8	zl,ICR1L
	inm8	zh,ICR1H
	push	temp2
	ldi	temp2,(1<<ICES1)
	inm8	temp,TCCR1B
	eor	temp,temp2			;toggle edge
	outm8	TCCR1B,temp
	ldi	temp,(1<<ICF1)			;clear pending int
	out	TIFR1,temp
	
#if 0
	lds	temp,srx_state
	subi	temp,-'0'
	rcall	uartputc
	lds	temp,srx_dr
	rcall	printhex
#endif	
	lds	temp,srx_state
	cpi	temp,0
	brne	srxi_S1		

; State 0: Wait for start bit

;	sts	srx_lastedge,zl			;save beginning of start bit
;	sts	srx_lastedge+1,zh
	movw	srx_lastedgel,zl
	sts	srx_dr,_0
	ldi	temp,1
	sts	srx_state,temp
	ldi	temp,2
	sts	srx_char_to,temp
	sbis	P_RXD-2,RXD			;RXD still low?
	rjmp	srxi_end2
	ldi	zl,(1<<ICNC1)|(1<<CS10)		;next edge is falling edge
	outm8	TCCR1B,zl
	ldi	zh,(1<<ICF1)			;clear pending int
	out	TIFR1,zh
	sts	srx_state,_0
	sts	srx_char_to,_0
	rjmp	srxi_end2

srxi_S1:
	cpi	temp,1
	brne	srxi_S2

; State 1: Check start bit (and collect 0-bits)

;	lds	temp,srx_lastedge
;	lds	temp2,srx_lastedge+1
;	sts	srx_lastedge,zl
;	sts	srx_lastedge+1,zh

	movw	temp,srx_lastedgel
	movw	srx_lastedgel,zl

	sub	zl,temp
	sbc	zh,temp2
	subi	zl,low ((SSER_BIT_TC+1)/2)
	sbci	zh,high((SSER_BIT_TC+1)/2)
	brcs	srxi_sberr

;	mov	temp,zh	
;	rcall	printhex
;	mov	temp,zl
;	rcall	printhex

	ldi	temp,0x80
srxi_1l:
	subi	zl,low(SSER_BIT_TC)
	sbci	zh,high(SSER_BIT_TC)
	brcs	srxi_1be
	lsr	temp
	brcc	srxi_1l

	subi	zl,low(SSER_BIT_TC)		; stop bit?
	sbci	zh,high(SSER_BIT_TC)
	brcc	srxi_1fe
	rjmp	srxi_complete0			; ok, x00 (^@) received
srxi_1fe:
	sts	srx_char_to,_0			; no stop bit --> framing error --> break
	sts	srx_state,_0
	sbr	intstat,(1<<i_break)
	rjmp	srxi_end2

srxi_1be:
	sts	srx_dr,temp
	ldi	temp,2
	sts	srx_state,temp
	rjmp	srxi_end2

srxi_sberr:
	ldi	temp,(1<<ICNC1)|(1<<CS10)	;next edge is falling edge
	outm8	TCCR1B,temp
	ldi	temp,(1<<ICF1)		;clear pending int
	out	TIFR1,temp
	sts	srx_state,_0		;next state
#if 1
	ldi	temp,'?'
	rcall	uartputc
	subi	zl,low (-(SSER_BIT_TC+1)/2)
	sbci	zh,high(-(SSER_BIT_TC+1)/2)
	mov	temp,zh
	rcall	printhex
	mov	temp,zl
	rcall	printhex
#endif
	rjmp	srxi_end2

srxi_S2:
	cpi	temp,2
	brne	srxi_S3

; State 2: collect 1-bits

;	lds	temp,srx_lastedge
;	lds	temp2,srx_lastedge+1
;	sts	srx_lastedge,zl
;	sts	srx_lastedge+1,zh

	movw	temp,srx_lastedgel
	movw	srx_lastedgel,zl

	sub	zl,temp
	sbc	zh,temp2
	subi	zl,low ((SSER_BIT_TC+1)/2)
	sbci	zh,high((SSER_BIT_TC+1)/2)

	lds	temp,srx_dr
srxi_2l:
	sec				;one more 1 bit
	ror	temp
	brcs	srxi_complete1		;8 bits recieved
	subi	zl,low(SSER_BIT_TC)
	sbci	zh,high(SSER_BIT_TC)
	brcc	srxi_2l
	
	sts	srx_dr,temp
	ldi	temp,3
	sts	srx_state,temp
	rjmp	srxi_end2
	
srxi_complete1:
	ldi	temp2,1			;We are in start bit now.
	sts	srx_state,temp2
	ldi	temp2,2
	sts	srx_char_to,temp2
	rjmp	srxi_complete
	
srxi_S3:
	cpi	temp,3
	brne	srxi_S4

; State 3: collect 0-bits

;	lds	temp,srx_lastedge
;	lds	temp2,srx_lastedge+1
;	sts	srx_lastedge,zl
;	sts	srx_lastedge+1,zh

	movw	temp,srx_lastedgel
	movw	srx_lastedgel,zl

	sub	zl,temp
	sbc	zh,temp2
	subi	zl,low ((SSER_BIT_TC+1)/2)
	sbci	zh,high((SSER_BIT_TC+1)/2)

	lds	temp,srx_dr
srxi_3l:
					;one more 0 bit
	lsr	temp
	brcs	srxi_complete0		;8 bits recieved
	subi	zl,low(SSER_BIT_TC)
	sbci	zh,high(SSER_BIT_TC)
	brcc	srxi_3l
	
	sts	srx_dr,temp
	ldi	temp,2
	sts	srx_state,temp
	rjmp	srxi_end2

srxi_S4:
	ldi	zl,(1<<ICNC1)|(1<<CS10)	;next edge is falling edge
	outm8	TCCR1B,zl
	ldi	zl,(1<<ICF1)		;clear pending int
	sts	srx_state,_0		;next state
	rjmp	srxi_end2

srxi_complete0:	
	sts	srx_char_to,_0		;clear timeout
	sts	srx_state,_0		;next state
srxi_complete:
#if 0
	ldi	zl,(1<<ICNC1)|(1<<CS10)	;next edge is falling edge
	outm8	TCCR1B,zl
	ldi	zl,(1<<ICF1)		;clear pending int
	out	TIFR1,zl
#endif

; Save received character in a circular buffer. Do nothing if buffer overflows.

	lds	zh,rxcount		;if rxcount < RXBUFSIZE
	cpi	zh,RXBUFSIZE		;   (room for at least 1 char?)
	brsh	srxi_ov			; 
	inc	zh			;
	sts	rxcount,zh		;   rxcount++

	ldi	zl,low(rxfifo)		;  
	lds	zh,rxidx_w		;
	add	zl,zh			;
	inc	zh			;
	andi	zh,RXBUFMASK		;
	sts	rxidx_w,zh		;   rxidx_w = ++rxidx_w % RXBUFSIZE
	ldi	zh,high(rxfifo)		;
	brcc	PC+2			;
	inc	zh			;
	st	z,temp			;   rxfifo[rxidx_w] = char
srxi_ov:					;endif

srxi_end2:
	pop	temp2
srxi_end: 
	pop	zl
	pop	zh
	pop	temp
	out	sreg,temp
	pop	temp
	reti


;----------------------------------------------------------------------

	.cseg

; Timer/Counter1 Compare Match A interrupt
	
	INTERRUPT OC1Aaddr
	
	push	temp
	in	temp,sreg
	push	temp
	push	zh

	inm8	temp,OCR1AL
	inm8	zh,OCR1AH
	subi	temp,low(-SSER_BIT_TC)
	sbci	zh,high(-SSER_BIT_TC)
	outm8	OCR1AH,zh
	outm8	OCR1AL,temp
	lds	temp,stx_bitcount
	tst	temp
	breq	stxi_nxtchar
	
	dec	temp
	sts	stx_bitcount,temp
	ldi	zh,9				;Start bit?
	cp	temp,zh
	ldi	zh,(1<<COM1A1)
	breq	stxi_0
	lds	temp,stx_dr
	sbrs	temp,0
	ldi	zh,(1<<COM1A1)|(1<<COM1A0)
	lsr	temp
	sts	stx_dr,temp
stxi_0:
	outm8	TCCR1A,zh
	pop	zh
	pop	temp
	out	sreg,temp
	pop	temp
	reti

; more characters?
stxi_nxtchar:
	lds	temp,txcount		;if txcount != 0
	tst	temp			;
	breq	stxi_dis
; get next char
	push	zl
	dec	temp			;
	sts	txcount,temp		;   --txcount
	ldi	zl,low(txfifo)		;  
	ldi	zh,high(txfifo)		;
	lds	temp,txidx_r		;
	add	zl,temp			;
	brcc	PC+2			;
	inc	zh			;
	inc	temp			;
	andi	temp,TXBUFMASK		;
	sts	txidx_r,temp		;
	ld	temp,z
	com	temp
	sts	stx_dr,temp
	ldi	temp,10
	sts	stx_bitcount,temp
	pop	zl
	pop	zh
	pop	temp
	out	sreg,temp
	pop	temp
	reti

; disable transmitter
stxi_dis:
	ldi	temp,(1<<COM1A1)|(1<<COM1A0)
	outm8	TCCR1A,temp
	inm8	temp,TIMSK1
	andi	temp,~(1<<OCIE1A)
	outm8	TIMSK1,temp
	pop	zh
	pop	temp
	out	sreg,temp
	pop	temp
	reti
;------------------------------------------------------------------

srx_to:
#if 0
	ldi	zl,(1<<ICNC1)|(1<<CS10)	;next edge is falling edge
	outm8	TCCR1B,zl
	ldi	zl,(1<<ICF1)		;clear pending int
	out	TIFR1,zl
#endif
	sts	srx_state,_0		;next state
	push	temp

#if 0
	ldi	temp,'|'
	rcall	uartputc
#endif
	lds	temp,srx_dr		;only 0 if timeout after leading edge of start bit.
	tst	temp
	brne	srxto_store
	sbr	intstat,(1<<i_break)
	rjmp	srxto_ov

srxto_store:
	mov	zl,temp
	com	zl
	andi	zl,0x80
srxto_l:
	lsr	temp
	or	temp,zl
	brcc	srxto_l
	
; Save received character in a circular buffer. Do nothing if buffer overflows.

	lds	zh,rxcount		;if rxcount < RXBUFSIZE
	cpi	zh,RXBUFSIZE		;   (room for at least 1 char?)
	brsh	srxto_ov			; 
	inc	zh			;
	sts	rxcount,zh		;   rxcount++

	ldi	zl,low(rxfifo)		;  
	lds	zh,rxidx_w		;
	add	zl,zh			;
	inc	zh			;
	andi	zh,RXBUFMASK		;
	sts	rxidx_w,zh		;   rxidx_w = ++rxidx_w % RXBUFSIZE
	ldi	zh,high(rxfifo)		;
	brcc	PC+2			;
	inc	zh			;
	st	z,temp			;   rxfifo[rxidx_w] = char
srxto_ov:					;endif
	
	pop	temp
	ret
	
	
;Fetches a char from the buffer to temp. If none available, waits till one is.

uartgetc:
	lds	temp,rxcount		;Number of characters in buffer
	tst	temp
	breq	uartgetc		;Wait for char
	
	push	zh
	push	zl
	ldi	zl,low(rxfifo)
	ldi	zh,high(rxfifo)
	lds	temp,rxidx_r
	add	zl,temp
	brcc	PC+2
	inc	zh
	inc	temp
	andi	temp,RXBUFMASK
	sts	rxidx_r,temp
	cli
	lds	temp,rxcount
	dec	temp
	sts	rxcount,temp
	sei
	ld	temp,z		;don't forget to get the char
	pop	zl
	pop	zh
	ret

;Sends a char from temp to the soft uart. 

uartputc:
	push	zh
	push	zl
	in	zl,sreg
	push	zl
	push	temp
sputc_l:
	lds	temp,txcount		;do {
	cpi	temp,TXBUFSIZE		;
	brsh	sputc_l			;} while (txcount >= TXBUFSIZE)

	ldi	zl,low(txfifo)		;  
	ldi	zh,high(txfifo)		;
	lds	temp,txidx_w		;
	add	zl,temp			;
	brcc	PC+2			;
	inc	zh			;
	inc	temp			;
	andi	temp,TXBUFMASK		;
	sts	txidx_w,temp		;   txidx_w = ++txidx_w % TXBUFSIZE
	pop	temp			;
	st	z,temp			;   txfifo[txidx_w] = char
	cli
	lds	zh,txcount
	inc	zh
	sts	txcount,zh
	dec	zh
	brne	sputc_e
; Enable transmitter
	inm8	zh,TIMSK1
	ori	zh,(1<<OCIE1A)
	outm8	TIMSK1,zh
sputc_e:
	pop	zl
	out	sreg,zl
	pop	zl
	pop	zh
	ret


; Wait, till tx buffer is empty.

uart_wait_empty:
	push	temp
uwe_loop:
	lds	temp,txcount
	tst	temp
	brne	uwe_loop
	pop	temp
	ret


; vim:set ts=8 noet nowrap


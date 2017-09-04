.include 	"m328pdef.inc"

.equ     clock = 16000000          ;clock frequency

.cseg
;reset and interrupt vectors
.org 	0x0000
		rjmp 	reset

; the end of interrupt vectors

.org 	0x0100
reset:
; set stack pointer SPL/SPH
		ldi     r16, low(RAMEND)
		out     SPL, r16
		ldi     r16, high(RAMEND)
		out 	SPH, r16

setup:
		ldi 	r16, 0xff;
		out 	DDRD, r16
		clr 	r17
		out 	PORTD, r17

		sts 	UCSR0B, r17 ; clear USART configurated by the bootloader

main:
		clr 	r0
		clr 	r1
		dec 	r1

loop:
		out 	PORTD, r0
		eor 	r0, r1
		ldi 	r16, low(400000 & 0xffff)
		ldi 	r17, high(400000 & 0xffff)
		ldi 	r18, low(400000>>16)
		call wait24

		rjmp loop


wait24:
		subi 	r16, $1
		sbci 	r17, $0
		sbci 	r18, $0
		brcs 	wait24_over
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		nop
		rjmp 	wait24
wait24_over:
		ret

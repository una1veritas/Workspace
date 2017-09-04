 
.include "m8def.inc"
 
.org 0x0000
 
main:
	ldi	r16, 0xFF
	out	DDRD, r16
	ldi r16, 0x00
	out DDRB, r16
	out	PORTD, r16
	ldi	r16, 0x01
loop:
	out	PORTD, r16
	rcall wait
	in	r17, PINB
	sbrc r17, 4
	ror	r16
	rjmp	loop    

wait:
	ldi	r19, 0x6f
wait_loop1:
	ldi 	r18, 0xff
wait_loop0:
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	dec r18
	brne wait_loop0
	dec r19
	brne wait_loop1
	ret

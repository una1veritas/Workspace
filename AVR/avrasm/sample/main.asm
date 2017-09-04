.include "m88def.inc"
.device = ATmega88

.org  0x0000

main:
sp_init:
	ldi	  r16, low(RAMEND)
	ldi	  r17, high(RAMEND)
	out	  SPL, r16
	out	  SPH, r17
	
setup:
	ldi	  r16, (1<<PB1 | 1<<PB5)
	out	  DDRB, r16
	clr	  r16
	out	  PORTB, r16
	ldi	  r16, (1<<PD6)
	out	  DDRD, r16
	ldi	  r16, (1<<PD6)
	out	  PORTD, r16 

	ldi	  r16, 0x01
loop:
	out	  PORTB, r16
	call  wait
	in	  r17, PIND
	sbrc  r17, PD6
	ror	  r16
	rjmp  loop    

wait:
	ldi	  r19, 0xa0
wait_loop1:
	ldi	  r18, 0xff
wait_loop0:
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	dec	  r18
	brne  wait_loop0
	dec	  r19
	brne  wait_loop1
	ret

#include<avr/io.h>
 
.section .data
.byte leftright, empty

.section .text
.global main
 
main:
	ldi	r16, 0xFF
	out	_SFR_IO_ADDR(DDRD), r16
	cbi _SFR_IO_ADDR(DDRB), 4
	sbi _SFR_IO_ADDR(PORTB), 4
	clr r0
	out	_SFR_IO_ADDR(PORTD), r0
	sts leftright, r0
	inc r0
loop:
	out	_SFR_IO_ADDR(PORTD), r0
	call wait
	call wait
	in	r17, _SFR_IO_ADDR(PINB)
	sbrs r17, 4
	rjmp	loop
	sbrc r0, 7
	sts leftright, 1
	sbrc r0, 0
	sts leftright, 0
	lds  r16, leftright
	cpi  r16, 1
	breq rright
	rol  r0
	rjmp loop
rright:
	ror r0
	rjmp loop


wait:
	ldi	r19, 100
wait_loop1:
	ldi 	r18, 250
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
	dec r18
	brne wait_loop0
	dec r19
	brne wait_loop1
	ret


.end

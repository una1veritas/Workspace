#include<avr/io.h>
 
 
.global main
 
main:
	ldi	r16, 0xFF
	out	_SFR_IO_ADDR(DDRD), r16
	ldi 	r16, 0x00
	out	_SFR_IO_ADDR(PORTD), r16
	out 	_SFR_IO_ADDR(DDRB), r16
	
	ldi	r16, 0x01
loop:
	out	_SFR_IO_ADDR(PORTD), r16
	call 	wait
	in  	r17, _SFR_IO_ADDR(PINB)
	sbrc 	r17, 5
	ror	r16
	rjmp	loop    

wait:
	ldi	r19, 0x3f
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

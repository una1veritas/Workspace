#include<avr/io.h>
 
 
.global main
 
main:
	ldi	r16, 0xFF
	out	_SFR_IO_ADDR(DDRD), r16
	ldi	r16, 'Q'
loop:
	out	_SFR_IO_ADDR(PORTD), r16
	rjmp loop
	
.end

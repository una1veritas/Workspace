#include<avr/io.h>
 
 
.global main
 
main:
	ldi	r16, 0xFF
	out	_SFR_IO_ADDR(DDRD), r16
	ldi	r16, 0b00000001
	out _SFR_IO_ADDR(DDRB), r16
loop:
	in  r16, _SFR_IO_ADDR(PORTB)
	out	_SFR_IO_ADDR(PORTD), r16
	rjmp loop
	
.end

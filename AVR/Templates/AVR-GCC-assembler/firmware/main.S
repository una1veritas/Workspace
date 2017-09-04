#include <avr/io.h> 
 
.global main
 
main:
	ldi	r16, 0xFF
	out	_SFR_IO_ADDR(DDRD), r16
	ldi r16, 0x03
	out _SFR_IO_ADDR(DDRB), r16
	out	_SFR_IO_ADDR(PORTD), r16
	ldi r16, 0xfc
	out	_SFR_IO_ADDR(PORTB), r16
	ldi	r16, 0x01
	mov r0, r16
	
loop:
	out	_SFR_IO_ADDR(PORTD), r0
	out	_SFR_IO_ADDR(PORTB), r1
	call 	wait
	in	r17, _SFR_IO_ADDR(PINB)
	sbrs 	r17, 4
	rjmp	loop
	rol r0
	rol r1
	sbrs 	r1, 2
	rjmp 	loop
	clr r1
	sec
	rjmp 	loop

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

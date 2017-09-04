
.include	"m328pdef.inc"

.org 	0x00

setup:
		ldi	r16, LOW(RAMEND)
		out	SPL, r16
		ldi	r16, HIGH(RAMEND)
		out SPH, r16

		ldi r16, 0b100000
		out DDRB, r16

;		ldi r17, 10

loop:
		ldi r16, 1<<PB5 ; Arduino PIN d13
		out	PORTB, r16
		rcall wait

		ldi r16, 0
		out PORTB, r16
		rcall wait
		
		rjmp loop


wait:
		ldi		r20, 50
wait_1:
		rcall 	wait01sec
		dec		r20
		brne 	wait_1
		ret

wait01sec:
		push 	r20
		ldi 	r20, 100
wait01sec_1:
		rcall 	idle_250
		dec 	r20
		brne 	wait01sec_1

		pop		r20
		ret


idle_250:
		push 	r20
		ldi		r20, 250
wait1msec_1:
		nop
		dec 	r20
		brne 	wait1msec_1

		pop 	r20
		ret

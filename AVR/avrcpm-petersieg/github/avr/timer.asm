;    Timer module
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
;    $Id: timer.asm 139 2010-10-08 10:50:03Z leo $
;

	.dseg

delay_timer1:
	.byte	1
delay_timer2:
	.byte	1
timer_base:
timer_ms:
	.byte	2
timer_s:
	.byte	4
; don't change order here, clock put/get depends on it.
cntms_out:		; register for ms
	.byte	2
utime_io:		; register for uptime. 
	.byte	4	
cnt_1ms:
	.byte	2
uptime:
	.byte	4
timer_top:
.equ timer_size	= timer_top - timer_base
	
.equ clkofs	= cnt_1ms-cntms_out
.equ timerofs	= cnt_1ms-timer_ms
 
	.cseg	

; ------------- system timer 1ms ---------------


; Timer/Counter1 Compare Match B interrupt
	
	INTERRUPT OC1Baddr
	
	push    zl
	in      zl,SREG
	push    zl
	push	zh
	inm8	zl,OCR1BL
	inm8	zh,OCR1BH
	addiw	z,F_CPU/1000
	outm8	OCR1BH,zh
	outm8	OCR1BL,zl
	
#if DRAM_8BIT	/* Implies software uart */
	lds	zl,srx_char_to
	subi	zl,1
	brcs	syscl0
	sts	srx_char_to,zl
	brne	syscl0
	rcall	srx_to
syscl0:
#endif
	lds	zl,delay_timer1
	subi	zl,1
	brcs	syscl_t1n
	sts	delay_timer1,zl
syscl_t1n:	
	lds	zl,delay_timer2
	subi	zl,1
	brcs	syscl_t2n
	sts	delay_timer2,zl
syscl_t2n:	
	lds     zl,cnt_1ms
	lds     zh,cnt_1ms+1
	adiw	z,1
	
	sts	cnt_1ms,zl
	sts	cnt_1ms+1,zh
	cpi	zl,low(1000)
	ldi	zl,high(1000)		;doesn't change flags
	cpc	zh,zl
	brlo	syscl_end
	
	sts	cnt_1ms,_0
	sts	cnt_1ms+1,_0

	lds	zl,uptime+0
	inc	zl
	sts	uptime+0,zl
	brne	syscl_end
	lds	zl,uptime+1
	inc	zl
	sts	uptime+1,zl
	brne	syscl_end
	lds	zl,uptime+2
	inc	zl
	sts	uptime+2,zl
	brne	syscl_end
	lds	zl,uptime+3
	inc	zl
	sts	uptime+3,zl
	
syscl_end:
	pop	zh
	pop     zl
	out     SREG,zl
	pop     zl
	reti

; ----------------------------------------------
; 	delay
;
; wait for temp ms
;

delay_ms:
	sts	delay_timer1,temp
dly_loop:
	lds	temp,delay_timer1
	cpi	temp,0
	brne	dly_loop
	ret

; ----------------------------------------------
; 

clockget:
	ldi	temp,0xFF
	subi	temp2,TIMER_MSECS
	brcs	clkget_end		;Port number in range?
	ldiw	z,cntms_out
	breq	clkget_copy		;lowest byte requestet, latch clock
	cpi	temp2,6
	brsh	clkget_end		;Port number to high?
	
	add	zl,temp2
	brcc	PC+2
	 inc	zh
	ld	temp,z
clkget_end:
	ret
	
	
		
clkget_copy:
	ldi	temp2,6
	cli
clkget_l:
	ldd	temp,z+clkofs
	st	z+,temp
	dec	temp2
	brne	clkget_l
	sei
	lds	temp,cntms_out
					;req. byte in temp
	ret

clockput:
	subi	temp2,TIMERPORT
	brcs	clkput_end		;Port number in range?
	brne	clkput_1
	
	; clock control

	cpi	temp,starttimercmd
	breq	timer_start
	cpi	temp,quitTimerCmd
	breq	timer_quit
	cpi	temp,printTimerCmd
	breq	timer_print
	cpi	temp,uptimeCmd
	brne	cp_ex
	rjmp	uptime_print
cp_ex:
	ret	
	
timer_quit:
	rcall	timer_print
	rjmp	timer_start

clkput_1:
	dec	temp2
	ldiw	z,cntms_out
	breq	clkput_copy		;lowest byte requestet, latch clock
	cpi	temp2,6
	brsh	clkput_end		;Port number to high?
	
	add	zl,temp2
	 brcc	PC+2
	inc	zh
	st	z,temp
clkput_end:
	ret
		
clkput_copy:
	st	z,temp
	adiw	z,5
	ldi	temp2,6
	cli
clkput_l:
	ldd	temp,z+clkofs
	st	z+,temp
	dec	temp2
	brne	clkput_l
	sei
	ret

; start/reset timer
;
timer_start:
	ldiw	z,timer_ms
	ldi	temp2,6
	cli
ts_loop:
	ldd	temp,z+timerofs
	st	z+,temp
	dec	temp2
	brne	ts_loop
	sei
	ret


; print timer
;
	
timer_print:
	push	yh
	push	yl
	ldiw	z,timer_ms

; put ms on stack (16 bit)

	cli
	ldd	yl,z+timerofs
	ld	temp2,z+
	sub	yl,temp2
	ldd	yh,z+timerofs
	ld	temp2,z+
	sbc	yh,temp2
	brsh	tp_s
	
	addiw	y,1000
	sec	
tp_s:
	push	yh
	push	yl

	ldd	temp,z+timerofs
	ld	yl,z+
	sbc	temp,yl

	ldd	temp2,z+timerofs
	ld	yh,z+
	sbc	temp2,yh

	ldd	temp3,z+timerofs
	ld	yl,z+
	sbc	temp3,yl

	sei
	ldd	temp4,z+timerofs
	ld	yh,z+
	sbc	temp4,yh
	
	printnewline
	printstring "Timer running. Elapsed: "
	rcall	print_ultoa

	printstring "."
	pop	temp
	pop	temp2
	ldi	temp3,0
	ldi	temp4,0
	rcall	print_ultoa
	printstring "s."

	pop	yl
	pop	yh
	ret
	
uptime_print:

	ldiw	z,cnt_1ms
	
	cli
	ld	temp,z+
	push	temp
	ld	temp,z+
	push	temp
	
	ld	temp,z+
	ld	temp2,z+
	ld	temp3,z+
	sei
	ld	temp4,z+
	
	printnewline
	printstring "Uptime: "
	
	rcall	print_ultoa
	printstring ","

	ldi	temp3,0
	ldi	temp4,0
	pop	temp2
	pop	temp
	rcall print_ultoa
	printstring "s."

	ret

; vim:set ts=8 noet nowrap


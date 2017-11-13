;    Print and Debug functions
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
;    $Id: utils.asm 181 2012-03-19 19:36:07Z leo $
;


	.cseg


;Print a unsigned lonng value to the uart
; temp4:temp3:temp2:temp = value

print_ultoa:
	push	yh
	push	yl
	push	z_flags
	push	temp4
	push	temp3
	push	temp2
	push	temp
				
	clr	yl		;yl = stack level

ultoa1:	ldi	z_flags, 32	;yh = temp4:temp % 10
	clr	yh		;temp4:temp /= 10
ultoa2:	lsl	temp	
	rol	temp2	
	rol	temp3	
	rol	temp4	
	rol	yh	
	cpi	yh,10	
	brcs	ultoa3	
	subi	yh,10	
	inc	temp
ultoa3:	dec	z_flags	
	brne	ultoa2
	cpi	yh, 10	;yh is a numeral digit '0'-'9'
	subi	yh, -'0'
	push	yh		;Stack it
	inc	yl	
	cp	temp,_0		;Repeat until temp4:temp gets zero
	cpc	temp2,_0
	cpc	temp3,_0
	cpc	temp4,_0
	brne	ultoa1	
	
	ldi	temp, '0'
ultoa5:	cpi	yl,3		; at least 3 digits (ms)
	brge	ultoa6
	push	temp	
	inc	yl
	rjmp	ultoa5

ultoa6:	pop	temp		;Flush stacked digits
	rcall	uartputc
	dec	yl	
	brne	ultoa6	

	pop	temp
	pop	temp2
	pop	temp3
	pop	temp4
	pop	z_flags
	pop	yl
	pop	yh
	ret


;Prints temp2:temp in hex to the uart
printhexw:
	push	temp
	mov	temp,temp2
	rcall	printhex
	pop	temp
	;fall thru

;Prints temp in hex to the uart
printhex:
	swap temp
	rcall printhexn
	swap temp	
	;fall thru

;Prints the lower nibble
printhexn:
	push temp
	andi temp,0xf
	cpi temp,0xA
	brlo printhexn_isno
	subi temp,-7
printhexn_isno:
	subi temp,-'0'
	rcall uartputc
	pop temp
	ret


; Prints a single space

dbg_printspace:
	ldi	temp,' '
	rjmp	uartputc

; Prints 16 bytes pointed to by Z in hex.

dbg_hexdump_line:			;Address in z
	push	temp2
	push	temp
	printnewline
	movw	temp,z			;Print address
	rcall	printhexw
	ldi	temp2,16		;16 byte per line
	rcall	dbg_printspace
dbg_hdl1:
	cpi	temp2,8
	brne	PC+2
	rcall	dbg_printspace
	
	rcall	dbg_printspace
	ld	temp,z+
	rcall	printhex
	dec	temp2
	brne	dbg_hdl1
	sbiw	z,16
	
	rcall	dbg_printspace
	rcall	dbg_printspace
	ldi	temp2,16
dbg_hdl2:
	ld	temp,z+
	cpi	temp,' '
	brlo	dbg_hdpd
	cpi	temp,0x7F
	brlo	dbg_hdp
dbg_hdpd:
	ldi	temp,'.'
dbg_hdp:
	rcall	uartputc
	dec	temp2
	brne	dbg_hdl2
	sbiw	z,16
	rcall	dbg_printspace
	pop	temp
	pop	temp2
	ret
	

;Prints the zero-terminated string following the call statement. 

printstr:
	push	zh		;SP+5
	push	zl		;   4
	push	yh		;   3
	push	yl		;   2
	push	temp		;   1
	in	yh,sph
	in	yl,spl
	ldd	zl,y+7		;SP+7  == "return adr." == String adr.
	ldd	zh,y+6		;SP+6

	lsl zl			;word to byte conv.
	rol zh
printstr_loop:
	lpm temp,z+
	cpi temp,0
	breq printstr_end
	rcall uartputc
	cpi temp,13
	brne printstr_loop
	ldi temp,10
	rcall uartputc
	rjmp printstr_loop

printstr_end:
	adiw zl,1		;rounding up
	lsr zh			;byte to word conv.
	ror zl

	std	y+7,zl
	std	y+6,zh
	pop	temp
	pop	yl
	pop	yh
	pop	zl
	pop	zh
	ret
	
; --------------- Debugging stuff ---------------
; Print a line with the 8080/Z80 registers

; TODO: y:

printregs:
	mov	temp,z_flags
	rcall	printflags
	printstring "  A ="
	mov	temp,z_a
	rcall	printhex	
	printstring " BC ="
	ldd	temp2,y+oz_b
	ldd	temp,y+oz_c
	rcall	printhexw
	printstring " DE ="
	ldd	temp2,y+oz_d
	ldd	temp,y+oz_e
	rcall	printhexw
	printstring " HL ="
	ldd	temp2,y+oz_h
	ldd	temp,y+oz_l
	rcall	printhexw
	printstring " SP="
	movw	temp, z_spl
	rcall	printhexw
	printstring " PC="
	movw	temp, z_pcl
	rcall	printhexw
	printstring "       "
	movw 	xl,z_pcl
	mem_read
	rcall	printhex
	printstring " "
	adiw	x,1
	mem_read
	rcall	printhex
	printstring " "
	adiw	x,1
	mem_read
	rcall	printhex
	printstring " "

#if EM_Z80
	ldd	temp,y+oz_f2
	rcall	printflags
	printstring "  A'="
	ldd	temp,y+oz_a2
	rcall	printhex	
	printstring " BC'="
	ldd	temp2,y+oz_b2
	ldd	temp,y+oz_c2
	rcall	printhexw
	printstring " DE'="
	ldd	temp2,y+oz_d2
	ldd	temp,y+oz_e2
	rcall	printhexw
	printstring " HL'="
	ldd	temp2,y+oz_h2
	ldd	temp,y+oz_l2
	rcall	printhexw
	printstring " IX="
	ldd	temp2,y+oz_xh
	ldd	temp,y+oz_xl
	rcall	printhexw
	printstring " IY="
	ldd	temp2,y+oz_yh
	ldd	temp,y+oz_yl
	rcall	printhexw
	printstring " I="
	ldd	temp,y+oz_i
	rcall	printhex	

	printstring "       "
#endif
	ret


#if EM_Z80
zflags_to_ch:
	.db	"SZ H VNC",0,0
#else	
zflags_to_ch:
	.db	"SZ H PNC",0,0
#endif
	
printflags:
	push	temp2
	mov	temp2,temp
	printnewline
	push	zl
	push	zh
	ldiw	z,zflags_to_ch*2
pr_zfl_next:
	lpm	temp,z+
	tst	temp
	breq	pr_zfl_end
	cpi	temp,' '			; Test if no flag
	breq	pr_zfl_noflag
	sbrs	temp2,7			; 
	 ldi	temp,' '			; Flag not set
	rcall	uartputc
pr_zfl_noflag:
	rol	temp2
	rjmp	pr_zfl_next
pr_zfl_end:
	pop	zh
	pop	zl	
	pop	temp2
	ret

; vim:set ts=8 noet nowrap


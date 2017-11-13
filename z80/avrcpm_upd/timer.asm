; This is a utility program to control the timer of the AVR CP/M emulator.
; This file is stolen from the SIMH AltairZ80 Simulator.
;
; changed to assemble with z80asm
; 
;	.Z80				; mnemonics only

TIMERCTL:	equ 040h

printStringCmd:	equ	09h
bdos:		equ	5
cr:			equ	13
lf:			equ	10
cmdLine:		equ	80h
startTimerCmd:	equ	1
quitTimerCmd:	equ	2
printTimerCmd:	equ	15
uptimeCmd:	equ	16
benchCmd:	equ	32

;	aseg
	org	100h

	jp	start

usage:	db	'Usage: TIMER S|P|B|Q|U',cr,lf
	db	'  S = (Re)Start the timer',cr,lf
	db	'  P = Print elapsed time since last S or Q command',cr,lf
	db	'  B = Print elapsed time for simple benchmark loop. wait < 60s',cr,lf
	db    '         ATmega88 with 20MHz D0 = 035,999s',cr,lf
	db	'  Q = Print the timer, then restart it',cr,lf
	db	'  U = Print uptime',cr,lf,'$',1AH

start:	ld	a,(cmdLine)		; get number of characters on command line
	or	a
	jp	z,pusage		; print usage, if command line empty
	ld	a,(cmdLine+2)	; get first character
	ld	hl,table		; <HL> points to (letter, command)^3
	ld	b,tabsize		; nr elements in table
again:	cp	(hl)		; compare command line letter with table entry
	inc	hl			; point to command
	jp	z,found		; if found
	inc	hl			; otherwise proceed to next entry
	dec	b			; decrement loop counter
	jp	nz,again		; try next character
pusage:	ld	de,usage	; address of usage text
	ld	c,printStringCmd	; CP/M command for print
	jp	bdos			; print it, get ret from bdos
found:	ld	a,(hl)	; get TIMER command
        cp     a,benchCmd
	jp     z,bench
	out	(TIMERCTL),a	; send to TIMER port
	ret
bench:
        ld      a,startTimerCmd
	out	(TIMERCTL),a	; start
; loop starts here
	ld      c,10
l1:    ld      b,0
l2:    ld      a,0
l3:	dec   a
	jp      nz,l3		; 256 x
        dec    b
	jp      nz,l2		; 256 x
	dec    c
	jp      nz,l1		; 10 x
; loop ends here
        ld      a,printTimerCmd
	out	(TIMERCTL),a	; print elapsed time
	ret				; and done

table:	db	'S',startTimerCmd
	db	'P',printTimerCmd
	db	'B',benchCmd
	db	'Q',quitTimerCmd
	db	'U',uptimeCmd
tabsize: equ	($-table)/2

timend:	equ	$
	ds	0300h-timend		; fill remainder with zeroes

;	end

; vim:set ts=8 noet nowrap


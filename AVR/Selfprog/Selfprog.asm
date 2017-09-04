.include 	"tn2313def.inc"

.macro 	ldiwz
		ldi 	zl, low(@0)
		ldi 	zh, high(@0)
.endmacro

.macro 	ldiwy
		ldi 	yl, low(@0)
		ldi 	yh, high(@0)
.endmacro

.macro 	ldiaz
		ldi 	zl, low(@0<<1)
		ldi 	zh, high(@0<<1)
.endmacro

.macro   addi 
   subi   @0, -@1      ;subtract the negative of an immediate value 
.endmacro

.macro	outi
		ldi		r16, @1
		out		@0, r16
.endmacro



.org 0x0000

.def	TL = r16
.def	TH = r17
.def	WL = r24
.def	WH = r25

reset:
		ldi		r16, low(RAMEND)	;set spl

start:		
		ldi		TL, 32
		ldiwy	LOW(Buffer)
		ldiwz	0x0000
loopy:
		lpm		WL, Z+
		lpm		WH, Z+
		st		Y+, WL
		st		Y+, WH
		subi		tl, 2
		brne	loopy
endofloopy:

		ldiwz	0x0000
		ldi		TH, 32
		outi	SPMCSR, (1<<CTPB)
		ldi		YL, LOW(Buffer)
		ldi		YH, HIGH(Buffer)
page_loading:
		ld		r0, Y+
		ld		r1, Y+
		outi	SPMCSR, (1<<SELFPRGEN)
		spm
		adiw	ZH:ZL, 2
		dec		TH
		dec		TH
		brne	page_loading


page_erasing:
		ldiaz	PAGESIZE*4
		outi	SPMCSR, (1<<PGERS)|(1<<SELFPRGEN)
		spm
		;
		nop
		nop
		nop
		nop

page_write:
		ldiaz	PAGESIZE*4
		outi	SPMCSR, (1<<PGWRT)|(1<<SELFPRGEN)
		spm
		nop
		nop
		nop
		nop

		rjmp      pc 

.org   PAGESIZE*4 
.db "mosi mosi kameyo kamesan yo", 0


.dseg
Buffer:	
.byte PAGESIZE<<1


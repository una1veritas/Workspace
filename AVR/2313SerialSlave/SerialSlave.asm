;
;
.include 	"tn2313def.inc"

.macro 	ldiw
		ldi 	@0H, high(@1)
		ldi 	@0L, low(@1)
.endmacro

.macro 	ldsw
		lds 	@0L, @1
		lds 	@0H, @1+1
.endmacro

; Create an Add Immediate by subtracting the negative 
; 
.macro   addi 
   subi   @0, -@1      ;subtract the negative of an immediate value 
.endmacro 

.macro	outi
		ldi		r16, @1
		out		@0, r16
.endmacro


.def 	buf0 = r18
.def 	buf1 = r19
.def 	buf2 = r20
.def 	buf3 = r21

.def 	tl = r16
.def 	th = r17
.def	wl = r24
.def	wh = r25

.macro 	reserve
		sts 	reg, r16
		;
		in		r16, SREG
		sts 	s_SREG, r16
		in  	r16, SPL
		sts 	s_SPL, r16
		;
		sts 	reg+1, r17
		sts 	reg+2, r18
		sts 	reg+3, r19
		sts 	reg+4, r20
		sts 	reg+5, r21
		sts 	reg+6, r22
		sts 	reg+7, r23
		sts 	reg+8, r24
		sts 	reg+9, r25
		sts 	reg+10, r26
		sts 	reg+11, r27
		sts 	reg+12, r28
		sts 	reg+13, r29
		sts 	reg+14, r30
		sts 	reg+15, r31
.endmacro

.macro 	restore
		lds		r17, reg+1
		lds 	r18, reg+2
		lds 	r19, reg+3
		lds 	r20, reg+4
		lds 	r21, reg+5
		lds 	r22, reg+6
		lds 	r23, reg+7
		lds 	r24, reg+8
		lds 	r25, reg+9
		lds 	r26, reg+10
		lds 	r27, reg+11
		lds 	r28, reg+12
		lds 	r29, reg+13
		lds 	r30, reg+14
		lds 	r31, reg+15
		;
		lds 	r16, s_SREG
		out		SREG, r16
		lds		r16, reg
.endmacro

.macro 	TX_char
		ldi		r16, @0
		rcall	USART_Transmit
.endmacro

.macro 	TX_r16
		rcall	USART_Transmit
.endmacro

.equ	CR = $0d

.equ     clock = 8000000          ;clock frequency
.equ     baudrate = 9600          ;choose a baudrate
;
.equ     ubrrval = (clock/(16*baudrate))-1



.cseg
;reset and interrupt vectors
.org 	0x0000	; RESET
		rjmp 	reset

.org	0x001	; INT0
		rjmp	ISR_INT0

.org	0x002
interrupt1:

.org 	0x0007
		rjmp 	ISR_USART0_RX
; the end of interrupt vectors

.org 	0x0013
user_prog:
;
;***************
main:
		ldi 	r16, 0xff
		out 	DDRB, r16
		sbi		DDRD, 5
		ldi		r16, 100
		sts		HOWMANYTIMES, r16
;		clr		r0
;		ldi		r16, $1
;		mov		r1, r16
;		ldi		r16, LOW(1000)
;		ldi		r17, HIGH(1000)
;loop_init:
;		clr		r2
;		clr		r3
loop:
;		cp		r3, r17
;		brne	loop_do
;		cp		r2, r16
;		brne	loop_do
;		rjmp	loop_end
;loop_do:
		out 	PORTB, r1
		sbi		PORTD, 5
		rcall 	wait
		cbi		PORTD, 5
		rcall 	wait
		;
;		add		r2, r1
;		adc		r3, r0
		rjmp 	loop
;loop_end:
;		; goes into forever_loop
;		rjmp	PC

wait:
		ldi 	r20, LOW(10000)
		ldi 	r21, HIGH(10000)
outer_loop:
		rcall 	wait01
		subi 	r20, 1
		sbci	r21, 0
		brne	outer_loop
		ret

wait01:
		lds 	r19, HOWMANYTIMES
inner_loop:
		nop
		nop
		nop
		dec 	r19
		brne 	inner_loop
		ret


.dseg
.org 	0x60
HOWMANYTIMES:	.byte	1
;
;************************
.cseg
.org 	0x0280
reset:
		outi	SPL, LOW(STACKBOTTOM-1)
;		ldi 	r16, (0<<PB7)|(1<<PB6)|(0<<PB5)
;		out 	DDRB, r16
init:
		rcall 	USART_Init
;		in		r16, MCUCR
;		andi	r16, ~((1<<ISC00) | (1<<ISC00))
;		out		MCUCR, r16
;		cbi		DDRD, 2
;		sbi		PORTD, 2
;		in		r16, GIMSK
;		ori		r16, (1<<INT0)
;		out		GIMSK, r16
		sei
		rjmp 	user_prog

;
;***********************
ISR_INT0:
		sbis 	UCSRA, RXC
		rjmp	ISR_INT0
		reti

;
;***********************
ISR_USART0_RX:
		reserve
		rcall 	assume_PC
		rcall	RX_word
		;
		mov 	r16, buf2
		cpi 	r16, 'W'
		brne	PC+2
		rjmp	write_progmem
		cpi 	r16, 'S'
		brne	PC+2
		rjmp	store_ram
		cpi 	r16, 'R'
		breq	read_progmem
		cpi 	r16, 'J'
		brne	PC+2
		rjmp	jump_into
		cpi 	r16, 'T'
		brne	PC+2
		rjmp	show_status

;skip_write:

		; default command
load_ram:
;		ldiw	Z, HexPrefix<<1
;		rcall	TX_message
		mov 	zh, buf1
		mov		zl, buf0
;		andi	zl, $f0
		mov		tl, zh
		rcall	TX_r16hexdec
		mov		r16, zl
		rcall	TX_r16hexdec
		TX_char	' '
;		ldi		th, $10
load_ram_dump:
		ld		tl, Z+
		rcall	TX_r16hexdec
;		TX_char	' '
;		dec		th
;		brne	load_ram_dump
		rjmp	output_end

read_progmem:
;		ldiw	Z, HexPrefix<<1
;		rcall	TX_message
		mov 	zh, buf1
		mov		zl, buf0
;		andi	zl, 0xf0
		mov		tl, zh
		rcall	TX_r16hexdec
		mov		tl, zl
		rcall	TX_r16hexdec
		TX_char	' '
		lsl 	zl
		rol 	zh
;		ldi		th, $10
read_progmem_loop:
		lpm 	tl, Z+
		mov		buf3, tl
		lpm 	tl, Z+
		rcall	TX_r16hexdec
		mov 	tl, buf3
		rcall	TX_r16hexdec
;		TX_char ' '
;		dec		th
;		brne	read_progmem_loop

		rjmp	output_end

store_ram:
		mov 	zl, buf0
		mov		zh, buf1
		rcall	RX_word
;		ldiw	Z, HexPrefix<<1
;		rcall	TX_message
		mov		tl, zl
		rcall	TX_r16hexdec
		TX_char	' '
;		mov 	tl, buf0 ; only low byte
		st  	Z, buf0
;		TX_char	'$'
		mov		tl, buf0
		rcall	TX_r16hexdec
		rjmp	output_end

write_progmem:
		mov		yh, buf1
		mov		yl, buf0
		rcall	RX_word
		; YH:YL is the address for the instruction to be updated.
		; buf1:buf0 holds new instruction word
;		ldiw	Z, HexPrefix<<1
;		rcall	TX_message
		mov		tl, yh
		rcall	TX_r16hexdec
		mov		tl, yl
		rcall	TX_r16hexdec
		TX_char	' '
		lsl		yl
		rol		yh
		mov		xl, yl
		andi	xl, $0f<<1
		; shift for address expr.
		; YH:YL, aword holds position of instruction (address<<1)
		mov		tl, buf1
		rcall	TX_r16hexdec
		mov		tl, buf0
		rcall	TX_r16hexdec
;		TX_char	CR

		
		mov		th, yh
		mov		tl, yl
		subi	tl, LOW(0x0013)
		sbci	th, HIGH(0x0013)
		brcc	PC+2

		rjmp	output_end
		mov		th, yh
		mov		tl, yl
		subi	tl, LOW(0x0280)
		sbci	th, HIGH(0x0280)
		brcs	PC+2
		rjmp	output_end

;
; fill the prog mem page with the contents of progmem
; except new value at vw
		outi	SPMCSR, (1<<CTPB) 	; clear page
		andi	yl, $ff & ($f0<<1)
		ldiw	Z, 0x0000	; the address in the page buffer

		mov		wl, r0
		mov		wh, r1
page_loading:
		mov		th, zl
		andi	th, $0f<<1
		cp		th, xl
		brne	load_progmem_val
load_modified:
		mov		r0, buf0
		mov		r1, buf1
		adiw	Y, 2
		rjmp	write_page_buffer
load_progmem_val:
		push	zh
		push	zl
		mov		zl, yl
		mov		zh, yh
		lpm		r0, Z+
		lpm		r1, Z+
		pop		zl
		pop		zh
		adiw	Y, 2
write_page_buffer:
		outi	SPMCSR, (1<<SELFPRGEN)
		spm
		adiw	ZH:ZL, 2
		cpi		zl, ($0f<<1)
		brne	page_loading
		;
		mov		r0, wl
		mov		r1, wh
;
page_erasing:
		mov		zl, yl
		mov		zh, yh
		andi	zl, $ff & ($f0<<1)
		;
		outi   	SPMCSR, (1<<PGERS)|(1<<SPMEN) 
		spm 
		nop
		nop

page_write:	; write page to flash
		outi	SPMCSR, (1<<PGWRT)|(1<<SELFPRGEN)
		spm

		ldi		tl, LOW(1536)
		ldi		th, HIGH(1536)
wait_for_page_write_completion:
		subi	tl, 1
		sbci	th, 0
		brne	wait_for_page_write_completion
		;
		rjmp 	output_end


jump_into:
		sts		s_PC, buf0
		sts		s_PC+1, buf1
		rjmp	output_end

show_status:
		lds		r16, s_PC+1
		rcall	TX_r16hexdec
		lds		r16, s_PC
		rcall	TX_r16hexdec
		TX_char	' '
		lds		r16, s_SPL
		rcall	TX_r16hexdec
		TX_char	' '
		lds		r16, s_SREG
		rcall	TX_r16hexdec
		rjmp	output_end


output_end:
		TX_char $0d

		lds		zl, s_SPL
		adiw	zl, 1
		lds		r16, s_PC+1
		st		Z+, r16
		lds		r16, s_PC
		st		Z, r16
		restore
		reti

assume_PC:
		clr 	zh
		lds 	zl, s_SPL
		adiw	zl,1
		ld  	r16, Z+
		sts 	s_PC+1, r16
		ld  	r16, Z
		sts 	s_PC, r16
		ret


USART_Init:
		ldi r17, HIGH(ubrrval)
		ldi r16, LOW(ubrrval)
		; Set baud rate
		out UBRRH, r17
		out UBRRL, r16
		; Enable receiver and transmitter, receiver interrupt
		ldi r16, (1<<RXEN)|(1<<TXEN)|(0<<UCSZ2) | (1<<RXCIE)
		out UCSRB,r16
		; Set frame format: 8data, 2stop bit
		ldi r16, (1<<USBS)|(3<<UCSZ0)
		out UCSRC,r16
		ret

USART_Transmit:
		; Wait for empty transmit buffer
		sbis 	UCSRA,UDRE
		rjmp 	USART_Transmit
		; Put data (r16) into buffer, sends the data
		out 	UDR,r16
		ret

USART_Receive:
		; Wait for data to be received
		sbis 	UCSRA, RXC
		rjmp 	USART_Receive
		; Get and return received data from buffer
		in 		r16, UDR
		ret

USART_Flush:
		sbis 	UCSRA, RXC
		ret
		in 		r16, UDR
		rjmp 	USART_Flush

RX_string:
		rcall 	USART_Receive
		st  	Z+, r16
		subi	r16, $2e
		brne 	RX_string
		ldi 	r16, $00
		st  	-Z, r16 ; place the terminal char
		ret
;RX_string end.

RX_word:
		ldi 	r16, $30
		mov 	buf3, r16
		mov 	buf2, r16
		mov 	buf1, r16
		mov 	buf0, r16
RX_word_receive:
		rcall 	USART_Receive
		cpi 	r16, $30
		brcs	RX_word_end
		cpi 	r16, $3a
		brcs	RX_word_cont
		cpi 	r16, $41
		brcs	RX_word_end
		cpi 	r16, $47
		brcs	RX_word_cont
		cpi 	r16, $61
		brcs	RX_word_end
		cpi 	r16, $67
		brcs	RX_word_cont
		breq	RX_word_end
RX_word_cont:
		mov   	buf3, buf2
		mov   	buf2, buf1
		mov   	buf1, buf0
		mov   	buf0, r16
		rjmp	RX_word_receive
RX_word_end:
; restore the delimiter
		push 	r16
RX_word_xtoi:
		mov 	r16, buf3
		rcall	hex_to_nibble
		mov 	r17, r16
		swap	r17
		mov 	r16, buf2
		rcall	hex_to_nibble
		or  	r17, r16
		mov 	buf2, r17
		mov 	r16, buf1
		rcall	hex_to_nibble
		mov 	r17, r16
		swap	r17
		mov 	r16, buf0
		rcall	hex_to_nibble
		or  	r16, r17
		mov 	buf0, r16
		mov		buf1, buf2
RX_word_exit:
		pop 	buf2
		ret
;
;
hex_to_nibble:
		subi	r16, $30
		cpi 	r16, $10
		brcs 	hex_to_nibble_exit
		andi	r16, $17
		subi	r16, 7
hex_to_nibble_exit:
		ret
;hex_to_nibble end.

nibble_to_hexdec:
		andi	r16, $0f
		cpi 	r16, 10
		brcs 	PC+2
		subi 	r16, -$7  ; addi r16, $7
;nibble_to_hexdec_2:
		subi 	r16, -$30 ; addi r16, $30
		ret

TX_r16hexdec:
		push	r16
		swap	r16
		rcall	nibble_to_hexdec  ;ntohex
		rcall 	USART_Transmit
		pop 	r16
		rcall	nibble_to_hexdec
		rcall 	USART_Transmit
		ret
;TX_bytehex end.


TX_message:
		lpm  	r16, Z+
		cpi 	r16, $00
		brne 	PC+2
		ret
		rcall 	USART_Transmit
		rjmp 	TX_message
;TX_message end.


;**************************
.dseg
.org 	0x00c8
STACKBOTTOM:

s_PC:		.byte	2
s_SREG: 	.byte 	1
s_SPL: 		.byte 	1

reg:		.byte	16


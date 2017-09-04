;2002 Richard Cappels All Rights Reserved email projects@cappels.org
;HOME

;*********************************
;2313 monitor program; should work on any AVR with UART
;*********************************

.include "2313def.inc"     

.equ     clock = 8000000          ;clock frequency
.equ     baudrate = 9600          ;choose a baudrate

.equ     baudconstant = (clock/(16*baudrate))-1

.def	temp = r16	;general purpose variable
.def	inchar = r17     ;char destined to go out the uart
.def	outchar = r18     ;char coming in from the uart
.def	inbytel = r19     ;lower byte for asci-hex conversion
.def	inbyteh = r20     ;higher byte for asci-hex conversion
.def	currentadd = r21  ;pointer to write address for first page of memory	

.org     $00
		rjmp     reset          ;reset handle

reset:  rjmp     init          ;start init

init:     
	ldi     temp,low(ramend)
	out     spl,temp     	;set spl
	ldi     temp,baudconstant     
	out     ubrr,temp     	;load baudrate
	rcall	TypeGreeting	
	ldi	currentadd,$00
;     	rcall	establish_address_send_data	;estsablish current address
	
loop:		;*****command interpretation loop****
	ldi	outchar,$3A	;send prompt (colon char) to terminal
	rcall	rs_send
	ldi	outchar,$20
	rcall	rs_send
	ldi	outchar,$20
	rcall	rs_send	
	rcall	rs_rec		;get char from terminal and
				;interpret char
				
	cpi	inchar,$2F	;if / decrement current address and display
	breq	decrementanddisplay
	cpi	inchar,$3F	;if ? then display meny
	breq	domenu
	cpi	inchar,$20	;if <space>, increment current address and dispaly
	breq	incaddress
	cpi	inchar,$0D	;if carriage return, read data at current address
	breq	readdata
	cpi	inchar,$21	;if ! then use watchdog timer to reset the
	breq	pull_the_plug
	cpi	inchar,$25	;if % (percent sign) monitor memory in binary
	breq	monitorbinary
	
	andi	inchar,$DF	;make upper-case ascii
	
	cpi	inchar,$52	;if R or r, read data at current address
	breq	readdata	
	cpi	inchar,$57	;if W or w, write data at current address
	breq	writedata
	cpi	inchar,$41	;if A or a, set new current address
	breq	setaddress	
	cpi	inchar,$51	;if Q or q, restart monitor
	breq	reset	
	cpi	inchar,$4D	;if M or m, monitor memory location
	breq	monitormemory
	rjmp	loop		;keep going


decrementanddisplay:	;decrement current address and display current memory contens
	dec	currentadd
	rcall	typeaddressdata
	rjmp 	loop        
     
monitorbinary:	;send contents of memory as binary until char received from terminal
	
		
	rcall	crlf
	rcall	sendaddress	; write adress and a space char
	ldi     ZH,0 		; set Z register pair to point to current address
	mov     ZL,currentadd    	
	ld	inbytel,Z	; get contents of (correntadd) into inbytel		
	rcall 	sendbinarybyte	;send that byte to terminal
	ldi	outchar,$20	;send space char to terminal
	rcall	rs_send	
		
		
	sbi     ucr,rxen     ;set reciver bit.
	sbis     usr,rxc          ;repeat until char
     	rjmp     monitorbinary
	rjmp 	loop	     	;exit when char received
	
	
domenu:
	rcall	TypeGreeting
	rjmp 	loop

pull_the_plug:			;enable watchdog timer and wait for hardware reset
	ldi     ZH,high(2*resetmessage)     ; Load high part of byte address into ZH
	ldi     ZL,low(2*resetmessage)     ; Load low part of byte address into ZL
	rcall 	sendstring
	wdr
	ldi	temp,$08
	out	wdtcr,temp
wait_for_reset:
	rjmp	wait_for_reset


readdata:			;read data from current location to terminal
	rcall	typeaddressdata
	rjmp	loop


writedata:
	ldi     ZH,high(2*newdatamessage)     ; Load high part of byte address into ZH
	ldi     ZL,low(2*newdatamessage)     ; Load low part of byte address into ZL
	rcall 	sendstring			;ask for new datat to enter
	rcall	recbyte	
	ldi	ZH,$00
	mov	ZL,currentadd	
	st	Z,inbytel
	rcall	typeaddressdata
	rjmp	loop	

incaddress:	;increment current address and display current memory contens
	inc	currentadd
	rcall	typeaddressdata
	rjmp 	loop        
     

setaddress:	;set current address
	rcall	establish_address_send_data
	rjmp loop
	     	
monitormemory:	;send contents of memory until char received from terminal
	
		
	rcall	crlf
	rcall	sendaddress	; write adress and a space char
	ldi     ZH,0 		; set Z register pair to point to current address
	mov     ZL,currentadd    	
	ld	inbytel,Z	; get contents of (correntadd) into inbytel		
	rcall 	sendbyte	;send that byte to terminal
	ldi	outchar,$20	;send space char to terminal
	rcall	rs_send	
	sbi     ucr,rxen     	;set reciver bit.
	sbis     usr,rxc  	;repeat until char
     	rjmp     monitormemory
	rjmp 	loop	     	;exit when char received	


     
rs_send:
     sbi     ucr,txen     ;set sender bit
     sbis     usr,udre     ;wait till register is cleared
     rjmp     rs_send     
     out     udr,outchar     ;send the byte
     cbi     ucr,txen     ;clear sender bit
     ret               ;go back
     
     
     
rs_rec:     
	sbi     ucr,rxen     ;set reciver bit...               
	sbis     usr,rxc          ;wait for a byte
	rjmp     rs_rec
	in     inchar,udr     ;read valuable
	cbi     ucr,rxen     ;clear register
	ret               ;go back
     

rs_rec_echo:               ;receive and echo char
	rcall 	rs_rec
	mov     outchar,inchar
     	rcall  	rs_send          ;send to comX               
     	ret

crlf:                    ;send carriage return and line feed.
     ldi     ZH,high(2*crlfmessage)     ; Load high part of byte address into ZH
     ldi     ZL,low(2*crlfmessage)     ; Load low part of byte address into ZL
     rcall     sendstring
     ret



sendstring:          ;call with location of string in Z
     lpm                    ; Load byte from program memory into r0
     tst     r0               ; Check if we've reached the end of the message
     breq     finishsendstering     ; If so, return
     mov     outchar,r0
     rcall     rs_send
     adiw     ZL,1               ; Increment Z registers
     rjmp     sendstring
finishsendstering:
     ret



sendline:          ;send a string terminated in cariage return and line feed
               ;call with location of start of string in Z               
     rcall     sendstring
     rcall     crlf
     ret

sendaddress:	;send address followed by space char
	mov 	inbytel,currentadd
	rcall	sendbyte
	ldi	outchar,$20
	rcall 	rs_send
	ret

TypeGreeting:
	rcall      crlf
	rcall      crlf
	ldi     ZH,high(2*hellomessage)     ; Load high part of byte address into ZH
	ldi     ZL,low(2*hellomessage)     ; Load low part of byte address into ZL
	rcall     sendline          ; sent it.
	ret


askforaddress:
     ldi     ZH,high(2*askaddressmessage)     ; Load high part of byte address into ZH
     ldi     ZL,low(2*askaddressmessage)     ; Load low part of byte address into ZL
     rcall     sendstring          ; sent it.
     ret




byte_to_asciihex:     ;convert byte in inbytel to ascii in inbyteh,nbytel
mov     inbyteh,inbytel
lsr     inbyteh          ;convert the high nybble to ascii byte
lsr     inbyteh
lsr     inbyteh
lsr      inbyteh
subi     inbyteh,$D0     ;add $30
cpi      inbyteh,$3A     
brlo     PC+2          ;If less than 9 skip next instruction
subi     inbyteh,$F9     ;add 8 to ASCII (if data greater than 9)
     ; byte in inbyteh represents upper nybble that was in inbytel at start

andi     inbytel,0b00001111     ;convert the lower nybble to ascii byte
subi      inbytel,$D0     ;add $30
cpi      inbytel,$3A     
brlo     PC+2          ;If less than 9 skip next instruction
subi     inbytel,$F9     ;add 8 to ASCII (if data greater than 9)
     ; byte in inbyteh represents upper nybble that was in inbytel at start
ret


asciihex_to_byte:     	;convert ascii in inbyteh,inbytel to byte in inbytel
sbrc     inbyteh,6     	;convert high byte
subi     inbyteh,$f7     ;add     inbyte,temp     ;if bit 6 is set, add $09
andi     inbyteh,$0F

sbrc     inbytel,6 	;convert low byte
subi     inbytel,$f7     ;add     inbyte,temp     ;if bit 6 is set, add $09
andi     inbytel,$0F

lsl     inbyteh     	;combine them
lsl     inbyteh
lsl     inbyteh
lsl     inbyteh     
or     inbytel,inbyteh
ret

sendbyte:      ;send byte contained in inbytel to terminal

	rcall 	byte_to_asciihex
     	mov     outchar,inbyteh
     	rcall 	rs_send
     	mov     outchar,inbytel
     	rcall  	rs_send
	ret
	
sendbinarybyte:
	ldi	temp,$08
stillsendingbinary:	
	ldi	outchar,$30
	rol	inbytel
	brcc	dontsendone
	ldi	outchar,$31
dontsendone:
     	rcall  	rs_send
     	dec 	temp
     	brne	stillsendingbinary    	
	ret



recbyte:    	; get ascii hex byte from terminal. Result in inbytel
	rcall 	rs_rec_echo          		;read from comX
	mov     inbyteh,inchar     		;put 1st char from uart into inbyteh for hex conversion     
	rcall  	rs_rec_echo
	mov     inbytel,inchar     		;put 2nd char from uart into inbytel for hex conversion
	rcall 	asciihex_to_byte ;convert ASCII to byte in inbytel
	ret


typeaddressdata:
	rcall	crlf		;send carriage return and life feed
	rcall	sendaddress	; write adress and a space char
	ldi     ZH,0 		; set Z register pair to point to current address
	mov     ZL,currentadd    	
	ld	inbytel,Z	; get contents of (correntadd) into inbytel		
	rcall 	sendbyte	;send that byte to terminal
	ldi	outchar,$20	;send space char to terminal
	rcall	rs_send	
	ret


establish_address_send_data:
	rcall	askforaddress	;ask for memory address
	rcall	recbyte  	;get a byte from terminal
	mov	currentadd,inbytel ;save received byte as current address
	rcall	typeaddressdata
	ret



hellomessage:
.db     "2313 monitor  2002.02.01 V              Dick Cappels"
.db	$0A,$0D
.db	"A = set address "
.db	$0A,$0D
.db	"W = write "
.db	$0A,$0D
.db  	"R or <CR> = show memory contents at current address "
.db	$0A,$0D
.db	"<space> = increment current address and display "
.db	$0A,$0D
.db	"/ = decrement current address and display "
.db	$0A,$0D
.db	"M = monitor data until key pressed"
.db	$0A,$0D
.db	"% = monitor data and show in binary "
.db	$0A,$0D
.db	"? = type opening screen."
.db	$0A,$0D
.db	"Q = restart firmware"
.db	$0A,$0D
.db	"! = reset chip"
.db	$0A,$0D
.db	"All address and data values are hexadecimal and are byte wide."
.db	$0A,$0D
.db      00,00

askaddressmessage:
.db	"Address:  "
.db	$00,00

crlfmessage:
.db     $0A,$0D
.db     00,00


newdatamessage:
.db	"new data: "
.db	$00,$00

resetmessage:
.db	"Hardware reset via watchdog initiated."
.db	$00,$00


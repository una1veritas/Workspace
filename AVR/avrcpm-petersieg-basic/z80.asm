;    Z80 emulator with CP/M support. The Z80-specific instructions themselves actually aren't
;    implemented yet, making this more of an i8080 emulator.
;    
;    Copyright (C) 2010 Sprite_tm
;
;    This program is free software: you can redistribute it and/or modify
;    it under the terms of the GNU General Public License as published by
;    the Free Software Foundation, either version 3 of the License, or
;    (at your option) any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with this program.  If not, see <http://www.gnu.org/licenses/>.
;
;    $Id$
;

;.nolist


#if defined atmega8
	.include "m8def.inc"
#elif defined atmega168
	.include "m168def.inc"
#elif defined atmega88
	.include "m88def.inc"
	;FUSE_H=0xDF
	;FUSE_L=0xF7
#elif defined atmega328p
                               /* default */
	.include "m328Pdef.inc"
#endif
.list
.listmac


#ifndef F_CPU
	#define F_CPU  16000000        /* system clock in Hz; defaults to 20MHz */
#endif
#ifndef BAUD
	#define BAUD   38400           /* console baud rate */
#endif
 
#define PARTID 0x52		/* Partition table id */
				/* http://www.win.tue.nl/~aeb/partitions/partition_types-1.html */
#define RAMDISKNR 'I'-'A'	/* Driveletter for first RAM disk */
#define RAMDISKCNT 1		/* Number of RAM disks */

#define UBRR_VAL  ((F_CPU+BAUD*8)/(BAUD*16)-1)  /* clever rounding */

#define	RXBUFSIZE 64		/* USART recieve buffer size. Must be power of 2 */
#define	TXBUFSIZE 64		/* USART transmit buffer size. Must be power of 2 */

#define DRAM_WAITSTATES 1	/* Number of additional clock cycles for dram read access */
#define REFR_RATE   64000       /* dram refresh rate in cycles/s. */
				/* Most drams need 1/15.6µs. */
#define REFR_PRE    8           /* timer prescale factor */
#define REFR_CS     0x02        /* timer clock select for 1/8  */
#define REFR_CNT    F_CPU / REFR_RATE / REFR_PRE



#define DRAM_WORD_ACCESS 0	/* experimental */

#define EM_Z80	0		/* we don't have any z80 instructions yet */

.equ MMC_DEBUG   = 0
.equ INS_DEBUG   = 0
.equ MEMTEST     = 1
.equ BOOTWAIT    = 1
.equ PORT_DEBUG  = 0
.equ DISK_DEBUG  = 0
.equ HOSTRW_DEBUG= 0
.equ MEMFILL_CB  = 1
.equ STACK_DBG   = 0
.equ PRINT_PC    = 0

;Port declarations

; Port D
.equ rxd    = 0
.equ txd    = 1
.equ ram_oe = 2
.equ ram_a8 = 3
.equ mmc_cs = 4
.equ ram_a5 = 5
.equ ram_a6 = 6
.equ ram_a7 = 7

.equ P_OE  = PORTD
.equ P_AH  = PORTD
.equ P_A8  = PORTD
.equ P_MMC_CS = PORTD
	             ; ram_a[7..5]
.equ RAM_AH_MASK = (1<<ram_a8)|(1<<ram_a7)|(1<<ram_a6)|(1<<ram_a5)
.equ PD_OUTPUT_MASK = (1<<mmc_cs) | (1<<ram_oe) | RAM_AH_MASK


;Port B
.equ ram_a4 =	0
.equ ram_a3 =	1
.equ ram_a2 =	2
.equ ram_a1 =	3
.equ mmc_mosi =	3
.equ ram_a0 =	4
.equ mmc_miso =	4
.equ ram_ras =	5
.equ mmc_sck =	5


.equ P_RAS = PORTB
.equ P_AL  = PORTB
			; ram_a[4..0]
.equ RAM_AL_MASK = (1<<ram_a4)|(1<<ram_a3)|(1<<ram_a2)|(1<<ram_a1)|(1<<ram_a0)
.equ PB_OUTPUT_MASK = (1<<ram_ras) | RAM_AL_MASK

;Port C
.equ ram_d0 =	0
.equ ram_d1 =	1
.equ ram_d2 =	2
.equ ram_d3 =	3
.equ ram_w  =	4
.equ ram_cas=	5

.equ P_DQ  = PORTC
.equ P_W   = PORTC
.equ P_CAS = PORTC
.equ RAM_DQ_MASK = (1<<ram_d3)|(1<<ram_d2)|(1<<ram_d1)|(1<<ram_d0)
.equ PC_OUTPUT_MASK = (1<<ram_cas)|(1<<ram_w)


;Flag bits in z_flags
.equ ZFL_S	=	7
.equ ZFL_Z	=	6
.equ ZFL_H	=	4
.equ ZFL_P	=	2
.equ ZFL_N	=	1
.equ ZFL_C	=	0

;Register definitions

.def	_tmp	= r0	;  0
.def	_0	= r1

.def	z_c	= r4
.def	z_b	= r5
.def	z_e	= r6
.def	z_d	= r7
.def	z_l	= r8
.def	z_h	= r9
.def	z_a	= r10

.def	insdecl	= r12	;
.def	insdech	= r13	;
.def	z_spl	= r14
.def	z_sph	= r15	;
.def	temp	= r16 	;
.def	temp2	= r17 	;
.def	temp3	= r18
.def	temp4	= r19
.def	z_flags	= r20	;
			;
.def	opl	= r22	;
.def	oph	= r23	;
.def	z_pcl	= r24	;
.def	z_pch	= r25	;
; xl 		;r26
; xh		;r27
; yl		;r28
; yh		;r29
; zl		;r30	;
; zh		;r31	;



#if defined __ATmega8__
.equ	flags	= TWBR
.equ	P_PUD	= SFIOR
#else
.equ	flags	= GPIOR0
.equ	P_PUD	= MCUCR
#endif

; Flags:
	.equ	hostact	= 7		;host active flag
	.equ	hostwrt	= 6		;host written flag
	.equ	rsflag	= 5		;read sector flag
	.equ	readop	= 4		;1 if read operation
	.equ	trace	= 0

; This is the base z80 port address for clock access
#define	TIMERPORT 0x40
#define TIMER_CTL   TIMERPORT
#define TIMER_MSECS TIMERPORT+1
#define TIMER_SECS  TIMER_MSECS+2

#define starttimercmd	1
#define quitTimerCmd	2
#define printTimerCmd	15
#define uptimeCmd	16

#if defined __ATmega8__
.equ	RXTXDR0	= UDR
.equ	UCSR0A	= UCSRA
.equ	 UDRE0	= UDRE
.equ	UCSR0B	= UCSRB
.equ	 RXCIE0	= RXCIE
.equ	 UDRIE0	= UDRIE
.equ	 RXEN0	= RXEN
.equ	 TXEN0	= TXEN
.equ	UCSR0C	= UCSRC
.equ	 UCSZ00	= UCSZ0
.equ	 UCSZ01	= UCSZ1
.equ	UBRR0H	= UBRRH
.equ	UBRR0L	= UBRRL
.equ	OCR2A	= OCR2
.equ	OC2Aaddr= OC2addr
.equ	TCCR2A	= TCCR2
.equ	TCCR2B	= TCCR2
.equ	TIMSK1	= TIMSK
.equ	TIMSK2	= TIMSK
.equ	OCIE2A	= OCIE2
.equ	TIFR1	= TIFR
.equ	ICIE1	= TICIE1
#else
.equ	RXTXDR0	= UDR0
#endif


;----------------------------------------
; 
.macro	outm8
.if	@0 > 0x3f
	sts	@0,@1
.else
	out	@0,@1
.endif
.endm

;----------------------------------------
; 
.macro	inm8
.if	@1 > 0x3f
	lds	@0,@1
.else
	in	@0,@1
.endif
.endm



; -------------------- DRAM ---------------

; DRAM_SETADDR val, low_and_mask, low_or_mask, high_and_mask, high_or_mask
.macro DRAM_SETADDR
	mov temp,@0
.if low(@1) != 0xff
	andi temp,@1
.endif
.if  low(@2) != 0
	ori temp, @2
.endif
	out P_AL,temp
	
	mov temp,@0
.if low(@3) != 0xff
	andi temp,@3
.endif
	ori temp, @4 | (1<<mmc_cs)
	out P_AH,temp
.endm

;----------------------------------------
; add wait states
;	dram_wait  number_of_cycles

.macro dram_wait
.if @0 > 1
	rjmp	PC+1
	dram_wait @0 - 2
.elif @0 > 0
	nop
	dram_wait @0 - 1
.endif
.endm


.cseg
.org 0
	rjmp start		; reset vector
.org OC2Aaddr
	rjmp refrint		; tim2cmpa
.org OC1Aaddr   		; Timer/Counter1 Compare Match A
    	rjmp sysclockint	; 1ms system timer
.org URXCaddr   
	rjmp rxint		; USART receive int.
.org UDREaddr
	rjmp txint		; USART transmit int.
	
.org INT_VECTORS_SIZE

start:
	ldi temp,low(RAMEND)	; top of memory
	out SPL,temp		; init stack pointer
	ldi temp,high(RAMEND)	; top of memory
	out SPH,temp		; init stack pointer

	clr	_0

; - Kill wdt
	wdr
	out MCUSR,_0

	ldi temp,(1<<WDCE) | (1<<WDE)
	outm8	WDTCSR,temp
	ldi temp,(1<<WDCE)
	outm8	WDTCSR,temp

; - Clear RAM

	ldi	zl,low(SRAM_START)
	ldi	zh,high(SRAM_START)
	ldi	temp2,high(ramtop)
clr_l:
	st	z+,_0
	cpi	zl,low(ramtop)
	cpc	zh,temp2
	brne	clr_l

; - Setup Ports
	ldi 	temp,(1<<PUD)		;disable pullups
	outm8	P_PUD,temp
	ldi	temp,0xFF
	out 	PORTD,temp		;all pins high
	out 	PORTB,temp
	out 	PORTC,temp
	out 	DDRD,temp		; all outputs
	out	DDRB,temp
	out	DDRC,temp

	outm8	TIMSK1,_0
	outm8	TIMSK2,_0
	outm8	TCCR2A,_0
	outm8	TCCR2B,_0


; - Init serial port

	ldi temp, (1<<TXEN0) | (1<<RXEN0) | (1<<RXCIE0)
	outm8 UCSR0B,temp
.ifdef URSEL
	ldi temp, (1<<URSEL) | (1<<UCSZ01) | (1<<UCSZ00)
.else
	ldi temp, (1<<UCSZ01) | (1<<UCSZ00)
.endif
	outm8 UCSR0C,temp
	ldi temp, high(UBRR_VAL)
	outm8 UBRR0H,temp
	ldi temp, low(UBRR_VAL)
	outm8 UBRR0L,temp

; Init clock/timer system

; Init timer 1 as 1 ms system clock tick.

	ldi	temp,high(F_CPU/1000)
	outm8	OCR1AH,temp
	ldi	temp,low(F_CPU/1000)
	outm8	OCR1AL,temp
	ldi	temp,(1<<WGM12) | (1<<CS10)	;CTC, clk/1
	outm8	TCCR1B,temp
	inm8	temp,TIMSK1
	ori	temp,(1<<OCIE1A)
	outm8	TIMSK1,temp

;Init timer2. Refresh-call should happen every (8ms/512)=312 cycles.

	ldi	temp,REFR_CNT*2			; 2 cycles per int
	outm8	OCR2A,temp
	inm8	temp,TCCR2A
	ori	temp,(1<<WGM21)			;CTC mode
	outm8	TCCR2A,temp
	inm8	temp,TCCR2B
	ori	temp,REFR_CS			;clk/REFR_PRE
	outm8	TCCR2B,temp
	inm8	temp,TIMSK2
	ori	temp, (1<<OCIE2A)
	outm8	TIMSK2,temp

	sei


.if BOOTWAIT
	ldi temp,10
	rcall delay_ms

.endif

	rcall printstr
	.db 13,"CPM on an AVR, v1.0",13,0


.if MEMTEST
	rcall printstr
	.db "Testing RAM: fill...",0,0

;Fill RAM
	ldi xl,0
	ldi xh,0
ramtestw:
	mov temp,xh
	eor temp,xl
	rcall memwritebyte
	adiw xl,1
	brcc ramtestw
	rcall printstr
	.db "wait...",0

	ldi	temp2,8
ramtestwl:
	ldi	temp,255
	rcall	delay_ms
	dec	temp2
	brne	ramtestwl

	rcall printstr
	.db "reread...",13,0,0

;re-read RAM
	ldi xl,0
	ldi xh,0
ramtestr:
	rcall memReadByte
	mov temp2,xh
	eor temp2,xl
	cp temp,temp2
	breq ramtestrok
	rcall printhex
	ldi temp,'<'
	rcall uartPutc
	mov temp,xh
	eor temp,xl
	rcall printhex
	ldi temp,'@'
	rcall uartPutc
	mov temp,xh
	rcall printhex
	mov temp,xl
	rcall printhex
	ldi temp,13
	rcall uartPutc
ramtestrok:
	adiw xl,1
	brcc ramtestr

.endif

.if MEMFILL_CB
	;Fill ram with cbs, which (for now) will trigger an invalid opcode error.
	ldi xl,0
	ldi xh,0
ramfillw:
	ldi temp,0xcb
	rcall memwritebyte
	adiw xl,1
	brcc ramfillw
.endif



;----------------------------------------------------------------------------

; Partition table offsets:
#define PART_TYPE   4
#define PART_START  8
#define PART_SIZE  12

	rcall printstr
	.db "Initing mmc...",13,0

boot_again:
	rcall mmcInit
;Load first sector from MMC (boot sector)

	ldi	yh,0			; Sector 0
	ldi	yl,0
	movw	x,y
	rcall	mmcReadSect

;Test, if it has a valid MBR

	ldi	yl,low(hostparttbl)
	ldi	yh,high(hostparttbl)
	ldi	zl,low(hostbuf+510-1)	;Point to last byte of partition table
	ldi	zh,high(hostbuf+510-1)

	ldi	opl,0			;opl holds number of found disks (paritions)
	ldd	temp,z+1		;MBR signature (0xAA55)  at and of sector?
	ldd	temp2,z+2
	ldi	temp4,0xAA
	cpi	temp,0x55		
	cpc	temp2,temp4
	breq	boot_part

;No MBR, no partition table ...
	inc	opl			;pretend we have one.
	ldi	temp,high((1<<16) * 128/512)
	std	y+0,_0			;start at beginning of card
	std	y+1,_0
	std	y+2,_0
	std	y+3,_0
	std	y+4,_0			;max CP/M 2.2 disk size
	std	y+5,temp
	std	y+6,_0
	std	y+7,_0
	rjmp	boot_ipl
		
;Search Partition Table for CP/M partitions
boot_part:
	sbiw	z,63			;Now at first byte of partition table
	ldi	oph,high(hostbuf+510)
boot_ploop:
	ldd	temp,z+PART_TYPE
	cpi	temp,PARTID
	brne	boot_nextp
	
; Found a CP/M partition
	
	ldd	temp,z+PART_START
	st	y+,temp
	ldd	temp2,z+PART_START+1
	st	y+,temp2
	ldd	temp3,z+PART_START+2
	st	y+,temp3
	ldd	temp4,z+PART_START+3
	st	y+,temp4
	
	rcall	printstr
	.db	"CP/M partition at: ",0
	rcall	print_ultoa
	rcall	printstr
	.db	", size: ",0,0
	ldd	temp,z+PART_SIZE
	st	y+,temp
	ldd	temp2,z+PART_SIZE+1
	st	y+,temp2
	ldd	temp3,z+PART_SIZE+2
	st	y+,temp3
	ldd	temp4,z+PART_SIZE+3
	st	y+,temp4
	lsr	temp4
	ror	temp3
	ror	temp2
	ror	temp
	rcall	print_ultoa
	rcall	printstr
	.db	"KB.",13,0,0
	
	inc	opl
	cpi	opl,MAXDISKS
	breq	boot_pend	
boot_nextp:
	adiw	zl,16
	cpi	zl,low(hostbuf+510)
	cpc	zh,oph
	brsh	boot_pend
	rjmp	boot_ploop
boot_pend:

; Read first sector of first CP/M partition

	lds	xl,hostparttbl
	lds	xh,hostparttbl+1
	lds	yl,hostparttbl+2
	lds	yh,hostparttbl+3
	rcall	mmcReadSect

boot_ipl:
	sts	ndisks,opl
	tst	opl
	brne	boot_ipl2
	rcall	printstr
	.db	"No bootable CP/M disk found! Please change MMC/SD-Card",13,0
	ldi	temp2,18
boot_wl:
	ldi	temp,255
	rcall	delay_ms
	dec	temp2
	brne	boot_wl
	rjmp	boot_again
	

boot_ipl2:

;First sector of disk or first CP/M partition is in hostbuf.

;Save to Z80 RAM (only 128 bytes because that's retro)
	ldi zl,low(hostbuf)
	ldi zh,high(hostbuf)
	ldi xh,0x20
	ldi xl,0x00
iplwriteloop:
	ld temp,z+
	rcall memWriteByte
	adiw xl,1
	cpi zl,low(hostbuf+128)
	brne iplwriteloop
	cpi zh,high(hostbuf+128)
	brne iplwriteloop
	rcall	dsk_boot_nommc		;init (de)blocking buffer


;Init z80
	ldi temp,0x00
	mov z_pcl,temp
	ldi temp,0x20
	mov z_pch,temp

	cbi	flags,trace
	rcall printstr
	.db 13,"Ok, CPU is live!",13,0,0

main:
	cbi	flags,trace
	cpi z_pch,1
	brlo notraceon
	cpi z_pch,$dc
	brsh notraceon
	sbi	flags,trace
notraceon:


.if PRINT_PC
	cpi z_pch,1
	brlo noprintpc
	cpi z_pch,0xdc
	brsh noprintpc

	rcall printstr
	.db 13,"PC=",0
	mov temp,z_pch
	rcall printhex
	mov temp,z_pcl
	rcall printhex
	ldi temp,' '
;	ldi temp,10
;	rcall uartputc
noprintpc:
.endif

	; *** Stage 1: Fetch next opcode
	movw xl,z_pcl
	rcall memReadByte


.if INS_DEBUG
	sbis	flags,trace
	rjmp	notrace1
	rcall printstr
	.db "PC=",0
	push temp
	mov temp,z_pch
	rcall printhex
	mov temp,z_pcl
	rcall printhex
	pop temp
	rcall printstr
	.db ", opcode=",0
	rcall printhex
notrace1:
.endif
	adiw z_pcl,1

	; *** Stage 2: Decode it using the ins_table.
	ldi zh,high(inst_table*2)
	mov zl,temp
	add zl,temp
	adc zh,_0
	lpm insdecl,Z+
	lpm insdech,Z

.if INS_DEBUG
	sbis	flags,trace
	rjmp	notrace2
	rcall printstr
	.db ", decoded=",0,0
	mov temp,insdech
	rcall printhex
	mov temp,insdecl
	rcall printhex
	rcall printstr
	.db ".",13,0,0
notrace2:
.endif

	; *** Stage 3: Fetch operand. Use the fetch jumptable for this.
	mov temp,insdecl
	andi temp,0x1F
	breq nofetch
	ldi zl,low(fetchjumps)
	ldi zh,high(fetchjumps)
	add zl,temp
	adc zh,_0
	icall

.if INS_DEBUG
	sbis	flags,trace
	rjmp	notrace3
	rcall printstr
	.db "pre: oph:l=",0
	mov temp,oph
	rcall printhex
	mov temp,opl
	rcall printhex
	rcall printstr
	.db " -- ",0,0
notrace3:
.endif

nofetch:
	; *** Stage 4: Execute operation :) Use the op jumptable for this.
	mov temp,insdech
	andi temp,0xFC
	breq nooper
	lsr temp
	lsr temp
	ldi zl,low(opjumps)
	ldi zh,high(opjumps)
	add zl,temp
	adc zh,_0
	icall

.if INS_DEBUG
	sbis	flags,trace
	rjmp	notrace4
	rcall printstr
	.db ",post:oph:l=",0,0
	mov temp,oph
	rcall printhex
	mov temp,opl
	rcall printhex
notrace4:
.endif

nooper:
	; *** Stage 5: Store operand. Use the store jumptable for this.
	swap insdecl
	swap insdech
	movw temp,insdecl
	andi temp,0x0E
	andi temp2,0x30
	or temp,temp2
	breq nostore
	lsr temp
	ldi zl,low(storejumps)
	ldi zh,high(storejumps)
	add zl,temp
	adc zh,_0
	icall

.if INS_DEBUG
	sbis	flags,trace
	rjmp	notrace5
	rcall printstr
	.db ", stored.",0
notrace5:
.endif

nostore:

.if INS_DEBUG
	sbis	flags,trace
	rjmp	notrace6
	rcall printstr
	.db 13,0
notrace6:
.endif

	;All done. Neeeext!
	rjmp main


; ----------------Virtual peripherial interface ------

;The hw is modelled to make writing a CPM BIOS easier.
;Ports:
;0 	- Con status. Returns 0xFF if the UART has a byte, 0 otherwise.
;1 	- Console input, aka UDR.
;2 	- Console output
;15	- Disk select
;16,17 	- Track select
;18 	- Sector select
;20 	- Write addr l
;21 	- Write addr h
;22 	- Trigger - write  to read, to write a sector using the above info;
;			, write to allocated/dirctory/unallocated

	.equ	READ_FUNC  = 7
	.equ	WRITE_FUNC = 6
	.equ	BOOT_FUNC  = 5
	.equ	HOME_FUNC  = 4



;*****************************************************
;*         CP/M to host disk constants               *
;*****************************************************
	.equ	MAXDISKS  = 4		;Max number of Disks (partitions)
	.equ	blksize = 1024		;CP/M allocation size
	.equ	hostsize = 512		;host disk sector size
;	.equ	hostspt = 20		;host disk sectors/trk
	.equ	hostblk = hostsize/128	;CP/M sects/host buff
;	.equ	CPMSPT = hostblk*hostspt;CP/M sectors/track
	.equ	CPMSPT = 26		;
	.equ	SECMSK = hostblk-1	;sector mask
	.equ	SECSHF = log2(hostblk)	;sector shift

;*****************************************************
;*        BDOS constants on entry to write           *
;*****************************************************
	.equ	WRALL = 0		;write to allocated
	.equ	WRDIR = 1		;write to directory
	.equ	WRUAL = 2		;write to unallocated
	.equ	WRTMSK= 3		;write type mask


	.dseg
ndisks:		.byte	1	;Number of CP/M disks

seekdsk:	.byte	1	;seek disk number
seektrk:	.byte	2	;seek track number
seeksec:	.byte	1	;seek sector number

hostparttbl:	.byte	8*MAXDISKS ; host partition table (start sector, sector count)
hostdsk:	.byte	1	;host disk number
hostlba:	.byte	3	;host sector number (relative to partition start)

unacnt:		.byte	1	;unalloc rec cnt
unadsk:		.byte	1	;last unalloc disk
unatrk:		.byte	2	;last unalloc track
unasec:		.byte	1	;last unalloc sector

erflag:		.byte	1	;error reporting
wrtype:		.byte	1	;write operation type
dmaadr:		.byte	2	;last dma address
hostbuf:	.byte	hostsize;host buffer (from/to SD-card)


	.cseg
	
;Called with port in temp2. Should return value in temp.
portRead:
	cpi temp2,0
	breq conStatus
	cpi temp2,1
	breq conInp

	cpi	temp2,15
	breq	dskDiskCheck
	cpi	temp2,22
	breq	dskErrorRet

	cpi	temp2,TIMER_MSECS
	brlo	pr_noclock
	cpi	temp2,TIMER_MSECS+6
	brsh	pr_noclock
	rjmp	clockget

pr_noclock:
	ldi	temp,0xFF
	ret

;Called with port in temp2 and value in temp.
portWrite:
	cpi temp2,0
	breq dbgOut
	cpi temp2,2
	breq conOut

	cpi temp2,15
	breq dskDiskSel
	cpi temp2,16
	breq dskTrackSel_l
	cpi temp2,17
	breq dskTrackSel_h
	cpi temp2,18
	breq dskSecSel
	cpi temp2,20
	breq dskDmaL
	cpi temp2,21
	breq dskDmaH

	cpi temp2,22
	breq dskDoIt
	
	cpi	temp2,TIMERPORT
	brlo	pw_noclock
	cpi	temp2,TIMER_MSECS+6
	brsh	pw_noclock
	rjmp	clockput

pw_noclock:
	ret


conStatus:

	lds temp,rxcount
	tst temp
	breq PC+2
	 ldi temp,0xff
	ret

conInp:
	rjmp uartGetc

dbgOut:
	rcall printstr
	.db "Debug: ",0
	rcall printhex
	rcall printstr
	.db 13,0
	ret

conOut:
	rjmp uartputc


dskDiskCheck:
	lds	temp,seekdsk
	lds	temp2,ndisks	;check if selected disk # is less then # of disks
	cp	temp,temp2
	brlt	dsk_dchend
	
	cpi	temp,RAMDISKNR
	brlt	dsk_dcher
	cpi	temp,RAMDISKNR+RAMDISKCNT
	brlt	dsk_dchend
dsk_dcher:
	ldi	temp,0xff		;error return
	ret
	
dsk_dchend:
	ldi	temp,0
	ret
	
dskErrorRet:
	lds	temp,erflag
	ret

dskDiskSel:
	sts seekdsk,temp
	ret

dskTrackSel_l:
	sts seektrk,temp
	sts seektrk+1,_0
	ret

dskTrackSel_h:
	sts seektrk+1,temp
	ret

dskSecSel:
	sts seeksec,temp
	ret

dskDmal:
	sts dmaadr,temp
	ret

dskDmah:
	sts dmaadr+1,temp
	ret

dskDoIt:
.if DISK_DEBUG
	push temp
	sbrc	temp,READ_FUNC
	 rjmp	dskdbgr
	sbrc	temp,WRITE_FUNC
	 rjmp	dskdbgw
	rjmp	dskdbge

dskdbgr:
	rcall printstr
	.db 13,"Disk read:  ",0
	rjmp	dskdbg1
dskdbgw:
	rcall printstr
	.db 13,"Disk write: ",0
dskdbg1:
	lds	temp,seekdsk
	subi	temp,-('A')
	rcall	uartputc
	rcall	printstr
	.db	": track ",0,0
	lds temp,seektrk+1
	rcall printhex
	lds temp,seektrk
	rcall printhex
	rcall printstr
	.db " sector ",0,0
	lds temp,seeksec
	rcall printhex
	rcall printstr
	.db " dma-addr ",0,0
	lds temp,dmaadr+1
	rcall printhex
	lds temp,dmaadr
	rcall printhex
	pop	temp
	push	temp
	sbrs	temp,WRITE_FUNC
	 rjmp	dskdbge
	rcall printstr
	.db " wrtype ",0,0
	andi	temp,3
	rcall printhex
dskdbge:
	pop temp
.endif
	;See what has to be done.
	sbrc	temp,READ_FUNC
	 rjmp	dsk_read
	sbrc	temp,WRITE_FUNC
	 rjmp	dsk_write
	sbrc	temp,HOME_FUNC
	 rjmp	dsk_home
	sbrc	temp,BOOT_FUNC
	 rjmp	dsk_boot

	rcall	printstr
	.db "DISK I/O: Invalid Function code: ",0
	rcall	printhex
	rjmp	haltinv

dsk_boot:
; TODO: Partition table must also be reread to make this work.
;	rcall mmcInit
dsk_boot_nommc:
	cbi	flags,hostact		;host buffer inactive
	sts	unacnt,_0		;clear unalloc count
	ret

dsk_home:
	sbis	flags,hostwrt		;check for pending write
	cbi	flags,hostact		;clear host active flag
	ret


dsk_read:

	sbi	flags,readop		;read operation
	;RAM disk?
	lds	temp2,seekdsk
	cpi	temp2,RAMDISKNR
	brlt	PC+2
	 rjmp	rdskDoIt

	sts	unacnt,_0
	sbi	flags,rsflag		;must read data
	ldi	temp,WRUAL		;write type
	sts	wrtype,temp		;treat as unalloc
	rjmp	dsk_rwoper		;to perform the read


dsk_write:
	;write the selected CP/M sector

	cbi	flags,readop		;not a read operation

	;RAM disk?
	lds	temp2,seekdsk
	cpi	temp2,RAMDISKNR
	brlt	PC+2
	 rjmp	rdskDoIt
	andi	temp,WRTMSK
	sts	wrtype,temp		;save write type

	cpi	temp,WRUAL		;write unallocated?
	brne	dsk_chkuna		;check for unalloc

;	write to unallocated, set parameters
	ldi	temp,blksize/128	;next unalloc recs
	sts	unacnt,temp
	lds	temp,seekdsk		;disk to seek
	sts	unadsk,temp		;unadsk = sekdsk
	lds	temp,seektrk
	sts	unatrk,temp		;unatrk = sectrk
	lds	temp,seektrk+1
	sts	unatrk+1,temp		;unatrk = sectrk
	lds	temp,seeksec
	sts	unasec,temp		;unasec = seksec
;
dsk_chkuna:
	;check for write to unallocated sector
	lds	temp,unacnt		;any unalloc remain?
	tst	temp
	breq	dsk_alloc		;skip if not

;	more unallocated records remain
	dec	temp			;unacnt = unacnt-1
	sts	unacnt,temp
	lds	temp,seekdsk		;same disk?
	lds	temp2,unadsk
	cp	temp,temp2		;seekdsk = unadsk?
	brne	dsk_alloc		;skip if not

;	disks are the same
	lds	temp,unatrk
	lds	temp2,unatrk+1
	lds	temp3,seektrk
	lds	temp4,seektrk+1
	cp	temp,temp3		;seektrk = unatrk?
	cpc	temp2,temp4
	brne	dsk_alloc		;skip if not

;	tracks are the same
	lds	temp,seeksec		;same sector?
	lds	temp2,unasec
	cp	temp,temp2		;seeksec = unasec?
	brne	dsk_alloc		;skip if not

;	match, move to next sector for future ref
	inc	temp2			;unasec = unasec+1
	sts	unasec,temp2
	cpi	temp2,CPMSPT		;end of track? (count CP/M sectors)
	brlo	dsk_noovf		;skip if no overflow

;	overflow to next track
	sts	unasec,_0		;unasec = 0
	lds	temp,unatrk
	lds	temp2,unatrk+1
	subi	temp, low(-1)		;unatrk = unatrk+1
	sbci	temp2,high(-1)
	sts	unatrk,temp
	sts	unatrk+1,temp2
;
dsk_noovf:
	cbi	flags,rsflag		;rsflag = 0
	rjmp	dsk_rwoper		;to perform the write
;
dsk_alloc:
	;not an unallocated record, requires pre-read
	sts	unacnt,_0		;unacnt = 0
	sbi	flags,rsflag		;rsflag = 1

;*****************************************************
;*	Common code for READ and WRITE follows       *
;*****************************************************

dsk_rwoper:
	;enter here to perform the read/write
.if DISK_DEBUG
	rcall	printstr
	.db	", flags: ",0
	in	temp,flags
	rcall	printhex
.endif
	sts	erflag,_0	;no errors (yet)

	;Convert track/sector to an LBA address (in 128byte blocks)

	lds	xl,seeksec		;
	ldi	xh,0			;
	ldi	yl,0			;
	lds	temp3,seektrk		;
	lds	temp4,seektrk+1		;
	ldi	temp,CPMSPT		;
	mul	temp3,temp		;
	add	xl,r0			;
	adc	xh,r1			;
	mul	temp4,temp		;
	add	xh,r0			;yl:xh:xl := sec + trk * SectorsPerTrack
	adc	yl,r1			;
	clr	_0

	mov	temp,xl
	andi	temp,SECMSK		;mask buffer number
	push	temp			;save for later

	;Convert from CP/M LBA blocks to host LBA blocks
	ldi temp,SECSHF
dsk_sh1:
	lsr	yl
	ror	xh
	ror	xl
	dec	temp
	brne	dsk_sh1
					;yl:xh:xl = host block to seek
;	active host sector?
	in	_tmp,flags		;host active flag
	sbi	flags,hostact		;always becomes 1
	sbrs	_tmp,hostact		;was it already?
	 rjmp	dsk_filhst		;fill host if not

;	host buffer active, same as seek buffer?
	lds	temp,seekdsk
	lds	temp2,hostdsk		;same disk?
	cp	temp,temp2		;seekdsk = hostdsk?
	brne	dsk_nomatch

;	same disk, same block?
	lds	temp,hostlba
	lds	temp2,hostlba+1
	lds	temp3,hostlba+2
	cp	xl,temp
	cpc	xh,temp2
	cpc	yl,temp3
	breq	dsk_match
;
dsk_nomatch:
	;proper disk, but not correct sector
	sbis	flags,hostwrt		;host written?
	 rjmp	dsk_filhst
	push	xl
	push	xh
	push	yl
	 rcall	dsk_writehost		;clear host buff
	pop	yl
	pop	xh
	pop	xl

dsk_filhst:
	;may have to fill the host buffer
	lds	temp,seekdsk
	sts	hostdsk,temp
	sts	hostlba,xl
	sts	hostlba+1,xh
	sts	hostlba+2,yl

	sbic	flags,rsflag		;need to read?
	 rcall	dsk_readhost		;yes, if 1
	cbi	flags,hostwrt		;no pending write

dsk_match:

	;copy data to or from buffer
	ldi	zl,low(hostbuf)
	ldi	zh,high(hostbuf)
	ldi	temp,128
	pop	temp2			;get buffer number (which part of hostbuf)
	mul	temp2,temp
	add	zl,r0			;offset in hostbuf
	adc	zh,r1
.if 0 ; DISK_DEBUG
	push	r0
	push	r1
	clr	_0
	rcall printstr
	.db "; host buf adr: ",0,0
	pop	temp
	rcall printhex
	pop	temp
	rcall printhex
.endif
	clr	_0

	lds	xl,dmaadr
	lds	xh,dmaadr+1
	ldi	temp3,128			;length of move
	sbic	flags,readop		;which way?
	 rjmp	dsk_rmove		;skip if read

;	mark write operation
	sbi	flags,hostwrt		;hostwrt = 1
dsk_wmove:
	rcall	memReadByte
	st	z+,temp
	adiw	xl,1
	dec	temp3
	brne dsk_wmove
	rjmp	dsk_rwmfin
	
dsk_rmove:
	ld	temp,z+
	rcall	memWriteByte
	adiw	xl,1
	dec	temp3
	brne	dsk_rmove
dsk_rwmfin:
;	data has been moved to/from host buffer
	lds	temp,wrtype	;write type
	cpi	temp,WRDIR	;to directory?
	breq	dsk_wdir
	ret			;no further processing
dsk_wdir:
;	clear host buffer for directory write
	lds	temp,erflag
	tst	temp		;errors?
	breq	dsk_wdir1
	ret			;skip if so
dsk_wdir1:
	rcall	dsk_writehost	;clear host buff
	cbi	flags,hostwrt	;buffer written
	ret

;*****************************************************

; hostdsk = host disk #,  (partition #)
; hostlba = host block #, relative to partition start 
; Read/Write "hostsize" bytes to/from hostbuf
	

dsk_hostparam:
	ldi	zl,low(hostparttbl)
	ldi	zh,high(hostparttbl)
	lds	temp,hostdsk
.if HOSTRW_DEBUG
	push	temp
	subi	temp,-('A')
	rcall	uartputc
	rcall	printstr
	.db	": ",0,0
	pop	temp
.endif

	lsl	temp		
	lsl	temp		
	lsl	temp		
	add	zl,temp		
	adc	zh,_0		

	lds	temp,hostlba
	lds	temp2,hostlba+1
	lds	temp3,hostlba+2

.if HOSTRW_DEBUG
	rcall	printstr
	.db	"lba: ",0
	clr	temp4
	rcall	print_ultoa
.endif

	ldd	xl,z+4
	ldd	xh,z+5
	ldd	yl,z+6
	
	cp	temp,xl
	cpc	temp2,xh
	cpc	temp3,yl
	brcs	dsk_hp1
	
.if HOSTRW_DEBUG
	rcall	printstr
	.db	", max: ",0
	push	temp4
	push	temp3
	push	temp2
	push	temp
	movw	temp,x
	mov	temp3,yl
	clr	temp4
	rcall	print_ultoa
	pop	temp
	pop	temp2
	pop	temp3
	pop	temp4
	rcall	printstr
	.db	" ",0
.endif
	
	clr	temp
	ret

dsk_hp1:
	ldd	xl,z+0
	ldd	xh,z+1
	ldd	yl,z+2
	ldd	yh,z+3

	add	xl,temp
	adc	xh,temp2
	adc	zl,temp3
	adc	zh,_0
.if HOSTRW_DEBUG
	rcall	printstr
	.db	", abs:",0,0
	push	temp4
	push	temp3
	push	temp2
	push	temp
	movw	temp,x
	movw	temp3,y
	rcall	print_ultoa
	pop	temp
	pop	temp2
	pop	temp3
	pop	temp4
	rcall	printstr
	.db	" ",0
.endif
	ori	temp,255
dsk_hpex:
	ret

;*****************************************************
;*	WRITEhost performs the physical write to     *
;*	the host disk, READhost reads the physical   *
;*	disk.					     *
;*****************************************************

dsk_writehost:
.if HOSTRW_DEBUG
	rcall	printstr
	.db	13,"host write ",0,0
.endif
	rcall	dsk_hostparam
	brne	dsk_wr1
	ldi	temp,255
	sts	erflag,temp
	ret
	
dsk_wr1:
	rcall	mmcWriteSect
	sts	erflag,_0
	ret

dsk_readhost:
.if HOSTRW_DEBUG
	rcall	printstr
	.db	13,"host read  ",0,0
.endif
	rcall	dsk_hostparam
	brne	dsk_rd1
	ldi	temp,255
	sts	erflag,temp
	ret
	
dsk_rd1:
	rcall	mmcReadSect
	sts	erflag,_0
	ret


;***************************************************************************
; ----------------- RAM disk -----------------

	.dseg
rdskbuf:
	.byte	128
	
	.cseg
;----------------------------------------------

rdsk_adr:
	ldi	xl,0
	lds	xh,seeksec
	lds	temp2,seektrk
	
	lsr	xh
	ror	xl			;Col 0..7
	
	mov	temp,temp2
	andi	temp,0x0f
	swap	temp
	or	xh,temp			;Row  0..7
	
	ldi	zl,low (rdskbuf)
	ldi	zh,high(rdskbuf)
	ldi	temp3,128
	DRAM_SETADDR xh, ~0,(1<<ram_ras), ~0,(1<<ram_a8)|(1<<ram_oe)
	cbi	P_RAS,ram_ras

.if DISK_DEBUG
	mov	temp,xh
	rcall	printhex
	rcall	printstr
	.db	" ",0
	mov	temp,xl
	rcall	printhex
	rcall	printstr
	.db	" ",0
.endif
	ret

;----------------------------------------------

rdskDoIt:
	sts	erflag,_0
	sbis	flags,readop
	 rjmp	rdsk_wr
	
.if DISK_DEBUG
	rcall	printstr
	.db	13,"rd-adr: ",0
.endif
	rcall	rdsk_adr
rdsk_rdl:
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_oe)), (1<<ram_a8)
	cbi	P_CAS,ram_cas
	cbi	P_A8,ram_a8
	inc	xl
	dram_wait DRAM_WAITSTATES	;
	in	temp,P_DQ-2		; PIN
	sbi	P_CAS,ram_cas

	cbi	P_CAS,ram_cas
	andi	temp,0x0f
	swap	temp
	dram_wait DRAM_WAITSTATES	;
	in	temp2,P_DQ-2	; PIN
	andi	temp2,0x0f
	or	temp,temp2

	sbi	P_OE,ram_oe
	sbi	P_CAS,ram_cas
	dec	temp3
	st	z+,temp
	brne	rdsk_rdl

	sbi	P_RAS,ram_ras
	ldi	zl,low (rdskbuf)
	ldi	zh,high(rdskbuf)
	lds	xl,dmaadr
	lds	xh,dmaadr+1
	ldi	temp3,128	
rdsk_rdstl:
	ld	temp,z+
	rcall	dram_write
	adiw	x,1
	dec	temp3
	brne	rdsk_rdstl
	ret
	

rdsk_wr:
.if DISK_DEBUG
	rcall	printstr
	.db	13,"wr-adr: ",0
.endif	
	lds	xl,dmaadr
	lds	xh,dmaadr+1
	ldi	zl,low (rdskbuf)
	ldi	zh,high(rdskbuf)
	ldi	temp3,128	
rdsk_wrldl:
	rcall	dram_read
	st	z+,temp
	adiw	x,1
	dec	temp3
	brne	rdsk_wrldl	

	ldi	temp2,RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out	DDRC,temp2
	rcall	rdsk_adr
rdsk_wrl:
	ld	temp,z+
	mov	temp2,temp
	andi	temp,RAM_DQ_MASK & ~(1<<ram_w)
	ori	temp,(1<<ram_cas)
	out	PORTC,temp
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_a8)),(1<<ram_oe)
	cbi	PORTC,ram_cas
	sbi	PORTD,ram_a8
	sbi	PORTC,ram_cas
	swap	temp2
	andi	temp2,RAM_DQ_MASK & ~(1<<ram_w)
	ori	temp2,(1<<ram_cas)
	out	PORTC,temp2
	cbi	PORTC,ram_cas
	inc	xl
	sbi	PORTC,ram_cas
	dec	temp3
	brne	rdsk_wrl

	sbi	P_RAS,ram_ras
	ldi	temp,~RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out	DDRC,temp
	out	PORTC,temp
	ret
	

;***************************************************************************

; ----------------- MMC/SD routines ------------------

mmcByteNoSend:
	ldi temp,0xff
mmcByte:

.if MMC_DEBUG
	rcall printstr
	.db "MMC: <--",0
	rcall printhex
.endif
	
	out SPDR,temp
mmcWrByteW:
	in temp,SPSR
	sbrs temp,7
	 rjmp mmcWrByteW
	in temp,SPDR

.if MMC_DEBUG
	rcall printstr
	.db ", -->",0
	rcall printhex
	rcall printstr
	.db ".",13,0
.endif
	ret


;Wait till the mmc answers with the response in temp2, or till a timeout happens.
mmcWaitResp:
	ldi zl,0
	ldi zh,0
mmcWaitResploop:
	rcall mmcByteNoSend
	cpi temp,0xff
	brne mmcWaitResploopEnd
	adiw zl,1
	cpi zh,255
	breq mmcWaitErr
	rjmp mmcWaitResploop
mmcWaitResploopEnd:
	ret


mmcWaitErr:
	mov temp,temp2
	rcall printhex
	rcall printstr
	.db ": Error: MMC resp timeout!",13,0
	rjmp resetAVR

mmcInit:
	ldi temp,0x53
	out SPCR,temp
	
	;Init start: send 80 clocks with cs disabled
	sbi P_MMC_CS,mmc_cs

;	ldi temp2,20
	ldi temp2,10     ; exactly 80 clocks
mmcInitLoop:
	mov temp,temp2
	rcall mmcByte
	dec temp2
	brne mmcInitLoop

	cbi P_MMC_CS,mmc_cs
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	sbi P_MMC_CS,mmc_cs
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	rcall mmcByteNoSend
	rcall mmcByteNoSend

	;Send init command
	cbi P_MMC_CS,mmc_cs
	ldi temp,0xff	;dummy
	rcall mmcByte
	ldi temp,0xff	;dummy
	rcall mmcByte
	ldi temp,0x40	;cmd
	rcall mmcByte
	ldi temp,0	;pxh
	rcall mmcByte
	ldi temp,0	;pxl
	rcall mmcByte
	ldi temp,0	;pyh
	rcall mmcByte
	ldi temp,0	;pyl
	rcall mmcByte
	ldi temp,0x95	;crc
	rcall mmcByte
	ldi temp,0xff	;return byte
	rcall mmcByte

	ldi temp2,0 			;Error Code 0
	rcall mmcWaitResp  		;Test on CMD0 is OK

	sbi P_MMC_CS,mmc_cs		;disable /CS
	rcall mmcByteNoSend


;Read OCR till card is ready
	ldi temp2,20			;repeat counter
mmcInitOcrLoop:	
	push temp2

	cbi P_MMC_CS,mmc_cs		;enable /CS
	ldi temp,0xff	;dummy
	rcall mmcByte
	ldi temp,0x41	;cmd
	rcall mmcByte
	ldi temp,0	;pxh
	rcall mmcByte
	ldi temp,0	;pxl
	rcall mmcByte
	ldi temp,0	;pyh
	rcall mmcByte
	ldi temp,0	;pyl
	rcall mmcByte
;	ldi temp,0x95			;crc
	ldi temp,0x01			;crc
	rcall mmcByte
	rcall mmcByteNoSend

	ldi temp2,1
	rcall mmcWaitResp		;wait until mmc-card send a byte <> 0xFF
							;the first answer must be 0x01 (Idle-Mode)
	cpi temp,0
	breq mmcInitOcrLoopDone ;second answer is 0x00 (Idle-Mode leave) CMD1 is OK

	sbi P_MMC_CS,mmc_cs		;disable /CS

;	rcall mmcByteNoSend     ;unnecessary

	ldi	temp,10
	rcall	delay_ms
	
	pop temp2
	dec temp2
	cpi temp2,0
	brne mmcInitOcrLoop		;repeat 

	ldi temp2,4  
	rjmp mmcWaitErr

mmcInitOcrLoopDone:
	pop temp2
	sbi P_MMC_CS,mmc_cs  		;disable /CS
	rcall mmcByteNoSend

	out SPCR,_0
	ret


;Call this with yh:yl:xh:xl = sector number
;
mmcReadSect:
	ldi temp,0x50
	out SPCR,temp

	cbi P_MMC_CS,mmc_cs
	rcall mmcByteNoSend
	ldi temp,0x51	;cmd (read sector)
	rcall mmcByte
	lsl	xl			;convert to byte address (*512)
	rol	xh
	rol 	yl
	mov	temp,yl
	rcall mmcByte
	mov temp,xh ;pxl
	rcall mmcByte
	mov temp,xl ;pyh
	rcall mmcByte
	ldi temp,0  ;pyl
	rcall mmcByte
	ldi temp,0x95	;crc
	rcall mmcByte
	ldi temp,0xff	;return byte
	rcall mmcByte

	;resp
	ldi temp2,2
	rcall mmcWaitResp

	;data token
	ldi temp2,3
	rcall mmcWaitResp

	;Read sector to AVR RAM
	ldi zl,low(hostbuf)
	ldi zh,high(hostbuf)
mmcreadloop:
	rcall mmcByteNoSend
	st z+,temp
	cpi zl,low(hostbuf+512)
	brne mmcreadloop
	cpi zh,high(hostbuf+512)
	brne mmcreadloop

	;CRC
	rcall mmcByteNoSend
	rcall mmcByteNoSend

	sbi P_MMC_CS,mmc_cs
	rcall mmcByteNoSend

	out SPCR,_0
	ret


;Call this with yh:yl:xh:xl = sector number
;
mmcWriteSect:
	ldi temp,0x50
	out SPCR,temp

	cbi P_MMC_CS,mmc_cs
	rcall mmcByteNoSend

	ldi temp,0x58	;cmd (write sector)
	rcall mmcByte
	lsl	xl			;convert to byte address (*512)
	rol	xh
	rol 	yl
	mov	temp,yl
	rcall mmcByte
	mov temp,xh ;pxl
	rcall mmcByte
	mov temp,xl ;pyh
	rcall mmcByte
	ldi temp,0  ;pyl
	rcall mmcByte
	ldi temp,0x95	;crc
	rcall mmcByte
	ldi temp,0xff	;return byte
	rcall mmcByte

	;resp
	ldi temp2,1
	rcall mmcWaitResp

	;Send data token
	ldi temp,0xfe
	rcall mmcByte

	;Write sector from AVR RAM
	ldi zl,low(hostbuf)
	ldi zh,high(hostbuf)
mmcwriteloop:
	ld temp,z+
	rcall mmcByte
	cpi zl,low(hostbuf+512)
	brne mmcwriteloop
	cpi zh,high(hostbuf+512)
	brne mmcwriteloop

	;CRC
	rcall mmcByteNoSend
	rcall mmcByteNoSend

	;Status. Ignored for now.
	rcall mmcByteNoSend

;Wait till the mmc has written everything
mmcwaitwritten:
	rcall mmcByteNoSend
	cpi temp,0xff
	brne mmcwaitwritten

	sbi P_MMC_CS,mmc_cs
	rcall mmcByteNoSend

	out SPCR,_0
	ret


;Set up wdt to time out after 1 sec.
resetAVR:
	cli
	ldi temp,(1<<WDCE)
	outm8 WDTCSR,temp
	ldi temp,(1<<WDCE) | (1<<WDE) | (110<<WDP0)
	outm8 WDTCSR,temp
resetwait:
	rjmp resetwait

; ------------------ DRAM routines -------------

;Loads the byte on address xh:xl into temp.
;must not alter xh:xl

dram_read:
	cli
	DRAM_SETADDR xh, ~0,(1<<ram_ras), ~(1<<ram_a8), (1<<ram_oe)
	cbi P_RAS,ram_ras
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_oe)), (1<<ram_a8)
	cbi P_CAS,ram_cas
	cbi P_A8,ram_a8
	dram_wait DRAM_WAITSTATES	;
	in  temp,P_DQ-2		; PIN
	sbi P_CAS,ram_cas

	cbi P_CAS,ram_cas
	andi temp,0x0f
	swap temp
	dram_wait DRAM_WAITSTATES	;
	in  temp2,P_DQ-2	; PIN
	andi temp2,0x0f
	or  temp,temp2

	sbi P_OE,ram_oe
	sbi P_CAS,ram_cas
	sbi P_RAS,ram_ras
	sei
	ret

#if DRAM_WORD_ACCESS
dram_read_w:
	cpi xl,255
	brne dram_read_w1
	
	rcall dram_read
	push temp
	adiw xl,1
	rcall dram_read
	mov temp2,temp
	pop temp
	ret	

dram_read_w1:
	cli
	DRAM_SETADDR xh, ~0,(1<<ram_ras), ~(1<<ram_a8),(1<<ram_oe)
	cbi P_RAS,ram_ras
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_oe)), (1<<ram_a8)
	cbi P_CAS,ram_cas
	cbi P_A8,ram_a8
	nop
	in  temp,P_DQ-2		; PIN
	sbi P_CAS,ram_cas
	cbi P_CAS,ram_cas
	andi temp,0x0f
	swap temp
	nop
	in  temp2,P_DQ-2	; PIN
	sbi P_CAS,ram_cas
	andi temp2,0x0f
	or  temp,temp2
	
;	push temp
	mov _wl,temp
	inc xl
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_oe)), (1<<ram_a8)
	cbi P_CAS,ram_cas
	cbi P_A8,ram_a8
	nop
	in  temp,P_DQ-2		; PIN
	sbi P_CAS,ram_cas
	cbi P_CAS,ram_cas
	andi temp,0x0f
	swap temp
	nop
	in  temp2,P_DQ-2	; PIN
	sbi P_CAS,ram_cas
	andi temp2,0x0f
	or  temp2,temp
;	pop temp
	mov temp,_wl

	sbi P_OE,ram_oe
	sbi P_RAS,ram_ras
	sei
	ret
#endif

;Writes the byte in temp to  xh:xl
;must not alter xh:xl

dram_write:
	cli
	ldi temp2,RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out DDRC,temp2

	mov  temp2,temp
	andi temp,RAM_DQ_MASK & ~(1<<ram_w)
	ori temp,(1<<ram_cas)
	out PORTC,temp
	DRAM_SETADDR xh, ~0,(1<<ram_ras), ~(1<<ram_a8),(1<<ram_oe)
	cbi P_RAS,ram_ras
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_a8)),(1<<ram_oe)
	cbi PORTC,ram_cas
	sbi PORTC,ram_cas

	sbi PORTD,ram_a8
	swap temp2

	andi temp2,RAM_DQ_MASK & ~(1<<ram_w)
	ori temp2,(1<<ram_cas)
	out PORTC,temp2

	cbi PORTC,ram_cas
	sbi P_RAS,ram_ras

	ldi temp,~RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out DDRC,temp
	out PORTC,temp
	sei
	ret

#if DRAM_WORD_ACCESS
dram_write_w:
	cpi xl,255
	brne dram_write_w1
	
	push temp2
	rcall dram_write
	pop temp
	adiw xl,1
	rcall dram_write
	ret	

dram_write_w1:
	cli
	push temp2
	ldi temp2,RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out DDRC,temp2

	mov  temp2,temp
	andi temp,RAM_DQ_MASK & ~(1<<ram_w)
	ori temp,(1<<ram_cas)
	out PORTC,temp

	DRAM_SETADDR xh, ~0,(1<<ram_ras), ~(1<<ram_a8),(1<<ram_oe)
	cbi P_RAS,ram_ras
	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_a8)),(1<<ram_oe)
	cbi PORTC,ram_cas
	sbi PORTC,ram_cas

	sbi PORTD,ram_a8
	swap temp2

	andi temp2,RAM_DQ_MASK & ~(1<<ram_w)
	ori temp2,(1<<ram_cas)
	out PORTC,temp2

	cbi PORTC,ram_cas
	sbi PORTC,ram_cas

	pop temp
	inc xl
	mov  temp2,temp
	andi temp,RAM_DQ_MASK & ~(1<<ram_w)
	ori temp,(1<<ram_cas)
	out PORTC,temp

	DRAM_SETADDR xl, ~(1<<ram_ras),0, ~((1<<ram_a8)),(1<<ram_oe)
	cbi PORTC,ram_cas
	sbi PORTC,ram_cas

	sbi PORTD,ram_a8
	swap temp2

	andi temp2,RAM_DQ_MASK & ~(1<<ram_w)
	ori temp2,(1<<ram_cas)
	out PORTC,temp2
	cbi PORTC,ram_cas

	sbi P_RAS,ram_ras

	ldi temp,~RAM_DQ_MASK | (1<<ram_w) | (1<<ram_cas)
	out DDRC,temp
	out PORTC,temp
	sei
	ret
#endif

; ****************************************************************************

; refresh interupt; exec 2 cbr cycles
refrint:			;4

	sbis P_RAS,ram_ras	;2
	reti
				;       CAS  RAS  
	cbi P_CAS,ram_cas	;2       1|   1|  
				;        1|   1|  
	cbi P_RAS,ram_ras	;2      |0    1|  
				;       |0    1|  
	nop			;1      |0   |0   
;	nop			;1      |0   |0   
	sbi P_RAS,ram_ras	;2      |0   |0   
				;       |0   |0   
	dram_wait DRAM_WAITSTATES-1	;
;	nop			;1      |0   |0   
	cbi P_RAS,ram_ras	;2      |0    1|  
				;       |0    1|  
	sbi P_CAS,ram_cas	;2      |0   |0   
				;       |0   |0   
	sbi P_RAS,ram_ras	;2       1|  |0   
				;        1|   1|  
	reti			;4  --> 21 cycles


; ------------- system timer 1ms ---------------
    .dseg

delay_timer:
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
	.equ timer_size = timer_top - timer_base
	
	.equ clkofs = cnt_1ms-cntms_out
	.equ timerofs = cnt_1ms-timer_ms
 
    .cseg	
sysclockint:
	push    zl
	in      zl,SREG
	push    zl
	push	zh
	
	lds	zl,delay_timer
	subi	zl,1
	brcs	syscl1
	sts	delay_timer,zl
syscl1:	
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

; wait for temp ms

delay_ms:
	sts	delay_timer,temp
dly_loop:
	lds	temp,delay_timer
	cpi	temp,0
	brne	dly_loop
	ret

; 

clockget:
	ldi	temp,0xFF
	subi	temp2,TIMER_MSECS
	brcs	clkget_end		;Port number in range?
	ldi	zl,low(cntms_out)
	ldi	zh,high(cntms_out)
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
	ldi	zl,low(cntms_out)
	ldi	zh,high(cntms_out)
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
	ldi	zl,low(timer_ms)
	ldi	zh,high(timer_ms)
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
	ldi	zl,low(timer_ms)
	ldi	zh,high(timer_ms)

; put ms on stack (16 bit)

	cli
	ldd	yl,z+timerofs
	ld	temp2,z+
	sub	yl,temp2
	ldd	yh,z+timerofs
	ld	temp2,z+
	sbc	yh,temp2
	brsh	tp_s
	
	subi	yl,low(-1000)
	sbci	yh,high(-1000)
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
	
	rcall printstr
	.db 13,"Timer running. Elapsed: ",0
	rcall	print_ultoa

	rcall printstr
	.db ".",0
	pop	temp
	pop	temp2
	ldi	temp3,0
	ldi	temp4,0
	rcall	print_ultoa
	rcall printstr
	.db "s.",0,0

	pop	yl
	pop	yh
	ret
	
uptime_print:

	ldi	zl,low(cnt_1ms)
	ldi	zh,high(cnt_1ms)
	
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
	
	rcall printstr
	.db 13,"Uptime: ",0
	
	rcall	print_ultoa
	rcall printstr
	.db ",",0

	ldi	temp3,0
	ldi	temp4,0
	pop	temp2
	pop	temp
	rcall print_ultoa
	rcall printstr
	.db "s.",0,0

	ret


	
; --------------- Debugging stuff ---------------

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


;Prints the lower nibble of temp in hex to the uart
printhexn:
	push temp
	andi temp,0xf
	cpi temp,0xA
	brlo printhexn_isno
	subi temp,-('A'-10)
	rcall uartputc
	pop temp
	ret
printhexn_isno:
	subi temp,-'0'
	rcall uartputc
	pop temp
	ret

;Prints temp in hex to the uart
printhex:
	swap temp
	rcall printhexn
	swap temp
	rcall printhexn
	ret

;Prints the zero-terminated string following the call statement. 

printstr:
	push	zh
	push	zl
	push	yh
	push	yl
	push	temp
	in	yh,sph
	in	yl,spl
	ldd	zl,y+7
	ldd	zh,y+6

	lsl zl
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
	adiw zl,1
	lsr zh
	ror zl

	std	y+7,zl
	std	y+6,zh
	pop	temp
	pop	yl
	pop	yh
	pop	zl
	pop	zh
	ret
	
; --------------- AVR HW <-> Z80 periph stuff ------------------

.equ memReadByte	=	dram_read
.equ memWriteByte	=	dram_write
#if DRAM_WORD_ACCESS
.equ memReadWord	=	dram_read_w
.equ memWriteWord	=	dram_write_w
#endif

; --------------------------------------------------------------

	.dseg
	
#define RXBUFMASK  RXBUFSIZE-1
#define TXBUFMASK  TXBUFSIZE-1

rxcount:
	.byte	1
rxidx_w:
	.byte	1
rxidx_r:
	.byte	1
txcount:
	.byte	1
txidx_w:
	.byte	1
txidx_r:
	.byte	1
rxfifo:
	.byte	RXBUFSIZE
txfifo:
	.byte	TXBUFSIZE

ramtop:	.byte	0	
	.cseg

; Save received character in a circular buffer. Do nothing if buffer overflows.

rxint:
	push	temp
	in	temp,sreg
	push	temp
	push	zh
	push	zl
	inm8	temp,RXTXDR0
	lds	zh,rxcount		;if rxcount < RXBUFSIZE
	cpi	zh,RXBUFSIZE		;   (room for at least 1 char?)
	brsh	rxi_ov			; 
	inc	zh			;
	sts	rxcount,zh		;   rxcount++

	ldi	zl,low(rxfifo)		;  
	lds	zh,rxidx_w		;
	add	zl,zh			;
	inc	zh			;
	andi	zh,RXBUFMASK		;
	sts	rxidx_w,zh		;   rxidx_w = ++rxidx_w % RXBUFSIZE
	ldi	zh,high(rxfifo)		;
	brcc	PC+2			;
	inc	zh			;
	st	z,temp			;   rxfifo[rxidx_w] = char
rxi_ov:					;endif
	pop	zl
	pop	zh
	pop	temp
	out	sreg,temp
	pop	temp
	reti


;Fetches a char from the buffer to temp. If none available, waits till one is.

uartgetc:
	lds	temp,rxcount		; Number of characters in buffer
	tst	temp
	breq	uartgetc		;Wait for char
	
	push	zh
	push	zl
	ldi	zl,low(rxfifo)
	ldi	zh,high(rxfifo)
	lds	temp,rxidx_r
	add	zl,temp
	brcc	PC+2
	inc	zh
	inc	temp
	andi	temp,RXBUFMASK
	sts	rxidx_r,temp
	cli
	lds	temp,rxcount
	dec	temp
	sts	rxcount,temp
	sei
	ld	temp,z		;don't forget to get the char
	pop	zl
	pop	zh
	ret

txint:	
	push	temp
	in	temp,sreg
	push	temp
	lds	temp,txcount		;if txcount != 0
	tst	temp			;
	breq	txi_e			; 

	dec	temp			;
	sts	txcount,temp		;   --txcount
	push	zh			;
	push	zl			;
	ldi	zl,low(txfifo)		;  
	ldi	zh,high(txfifo)		;
	lds	temp,txidx_r		;
	add	zl,temp			;
	brcc	PC+2			;
	inc	zh			;
	inc	temp			;
	andi	temp,TXBUFMASK		;
	sts	txidx_r,temp		;
	ld	temp,z
	outm8 RXTXDR0,temp
	pop	zl
	pop	zh
txi_e:					;endif
	lds	temp,txcount
	tst	temp
	brne	txi_x
	ldi temp, (1<<TXEN0) | (1<<RXEN0) | (1<<RXCIE0)
	outm8 UCSR0B,temp
txi_x:
	pop	temp
	out	sreg,temp
	pop	temp
	reti


;Sends a char from temp to the uart. 
uartputc:
	push	zh
	push	zl
	push	temp
putc_l:
	lds	temp,txcount		;do {
	cpi	temp,TXBUFSIZE		;
	brsh	putc_l			;} while (txcount >= TXBUFSIZE)

	ldi	zl,low(txfifo)		;  
	ldi	zh,high(txfifo)		;
	lds	temp,txidx_w		;
	add	zl,temp			;
	brcc	PC+2			;
	inc	zh			;
	inc	temp			;
	andi	temp,TXBUFMASK		;
	sts	txidx_w,temp		;   txidx_w = ++txidx_w % TXBUFSIZE
	pop	temp			;
	st	z,temp			;   txfifo[txidx_w] = char
	cli
	lds	zl,txcount
	inc	zl
	sts	txcount,zl
	ldi	zl, (1<<UDRIE0) | (1<<TXEN0) | (1<<RXEN0) | (1<<RXCIE0)
	outm8	UCSR0B,zl
	sei
	pop	zl
	pop	zh
	ret

; ------------ Fetch phase stuff -----------------

.equ FETCH_NOP	= (0<<0)
.equ FETCH_A	= (1<<0)
.equ FETCH_B	= (2<<0)
.equ FETCH_C	= (3<<0)
.equ FETCH_D	= (4<<0)
.equ FETCH_E	= (5<<0)
.equ FETCH_H	= (6<<0)
.equ FETCH_L	= (7<<0)
.equ FETCH_AF	= (8<<0)
.equ FETCH_BC	= (9<<0)
.equ FETCH_DE	= (10<<0)
.equ FETCH_HL	= (11<<0)
.equ FETCH_SP	= (12<<0)
.equ FETCH_MBC	= (13<<0)
.equ FETCH_MDE	= (14<<0)
.equ FETCH_MHL	= (15<<0)
.equ FETCH_MSP	= (16<<0)
.equ FETCH_DIR8	= (17<<0)
.equ FETCH_DIR16= (18<<0)
.equ FETCH_RST	= (19<<0)


;Jump table for fetch routines. Make sure to keep this in sync with the .equs!
fetchjumps:
	rjmp do_fetch_nop
	rjmp do_fetch_a
	rjmp do_fetch_b
	rjmp do_fetch_c
	rjmp do_fetch_d
	rjmp do_fetch_e
	rjmp do_fetch_h
	rjmp do_fetch_l
	rjmp do_fetch_af
	rjmp do_fetch_bc
	rjmp do_fetch_de
	rjmp do_fetch_hl
	rjmp do_fetch_sp
	rjmp do_fetch_mbc
	rjmp do_fetch_mde
	rjmp do_fetch_mhl
	rjmp do_fetch_msp
	rjmp do_fetch_dir8
	rjmp do_fetch_dir16
	rjmp do_fetch_rst

do_fetch_nop:
	ret

do_fetch_a:
	mov opl,z_a
	ret

do_fetch_b:
	mov opl,z_b
	ret

do_fetch_c:
	mov opl,z_c
	ret

do_fetch_d:
	mov opl,z_d
	ret

do_fetch_e:
	mov opl,z_e
	ret

do_fetch_h:
	mov opl,z_h
	ret

do_fetch_l:
	mov opl,z_l
	ret

do_fetch_af:
	mov opl,z_flags
	mov oph,z_a
	ret

do_fetch_bc:
	movw opl,z_c
	ret

do_fetch_de:
	movw opl,z_e
	ret

do_fetch_hl:
	movw opl,z_l
	ret

do_fetch_sp:
	movw opl,z_spl
	ret

do_fetch_mbc:
	movw xl,z_c
	rcall memReadByte
	mov opl,temp
	ret

do_fetch_mde:
	movw xl,z_e
	rcall memReadByte
	mov opl,temp
	ret

do_fetch_mhl:
	movw xl,z_l
	rcall memReadByte
	mov opl,temp
	ret

do_fetch_msp:
	movw xl,z_spl
#if DRAM_WORD_ACCESS
	rcall memReadWord
	movw opl,temp
#else
	rcall memReadByte
	mov opl,temp
	adiw xl,1
	rcall memReadByte
	mov oph,temp
#endif	
	ret

do_fetch_dir8:
	movw xl,z_pcl
	rcall memReadByte
	adiw z_pcl,1
	mov opl,temp
	ret

do_fetch_dir16:
	movw xl,z_pcl
#if DRAM_WORD_ACCESS
	rcall memReadWord
	movw opl,temp
#else
	rcall memReadByte
	mov opl,temp
	adiw xl,1
	rcall memReadByte
	mov oph,temp
#endif	
	adiw z_pcl,2
	ret

do_fetch_rst:
	movw xl,z_pcl
	subi xl,1
	sbci xh,0
	rcall memReadByte
	andi temp,0x38
	ldi oph,0
	mov opl,temp
	ret
	


; ------------ Store phase stuff -----------------

.equ STORE_NOP	= (0<<5)
.equ STORE_A	= (1<<5)
.equ STORE_B	= (2<<5)
.equ STORE_C	= (3<<5)
.equ STORE_D	= (4<<5)
.equ STORE_E	= (5<<5)
.equ STORE_H	= (6<<5)
.equ STORE_L	= (7<<5)
.equ STORE_AF	= (8<<5)
.equ STORE_BC	= (9<<5)
.equ STORE_DE	= (10<<5)
.equ STORE_HL	= (11<<5)
.equ STORE_SP	= (12<<5)
.equ STORE_PC	= (13<<5)
.equ STORE_MBC	= (14<<5)
.equ STORE_MDE	= (15<<5)
.equ STORE_MHL	= (16<<5)
.equ STORE_MSP	= (17<<5)
.equ STORE_RET	= (18<<5)
.equ STORE_CALL	= (19<<5)
.equ STORE_AM	= (20<<5)

;Jump table for store routines. Make sure to keep this in sync with the .equs!
storejumps:
	rjmp do_store_nop
	rjmp do_store_a
	rjmp do_store_b
	rjmp do_store_c
	rjmp do_store_d
	rjmp do_store_e
	rjmp do_store_h
	rjmp do_store_l
	rjmp do_store_af
	rjmp do_store_bc
	rjmp do_store_de
	rjmp do_store_hl
	rjmp do_store_sp
	rjmp do_store_pc
	rjmp do_store_mbc
	rjmp do_store_mde
	rjmp do_store_mhl
	rjmp do_store_msp
	rjmp do_store_ret
	rjmp do_store_call
	rjmp do_store_am


do_store_nop:
	ret

do_store_a:
	mov z_a,opl
	ret

do_store_b:
	mov z_b,opl
	ret

do_store_c:
	mov z_c,opl
	ret

do_store_d:
	mov z_d,opl
	ret

do_store_e:
	mov z_e,opl
	ret

do_store_h:
	mov z_h,opl
	ret

do_store_l:
	mov z_l,opl
	ret

do_store_af:
	mov z_a,oph
	mov z_flags,opl
	ret

do_store_bc:
	mov z_b,oph
	mov z_c,opl
	ret

do_store_de:
	mov z_d,oph
	mov z_e,opl
	ret

do_store_hl:
	mov z_h,oph
	mov z_l,opl
	ret

do_store_mbc:
	movw xl,z_c
	mov temp,opl
	rcall memWriteByte
	ret

do_store_mde:
	movw xl,z_e
	mov temp,opl
	rcall memWriteByte
	ret

do_store_mhl:
	movw xl,z_l
	mov temp,opl
	rcall memWriteByte
	ret

do_store_msp:
	movw xl,z_spl
#if DRAM_WORD_ACCESS
	movw temp,opl
	rcall memWriteWord
#else
	mov temp,opl
	rcall memWriteByte
	adiw xl,1
	mov temp,oph
	rcall memWriteByte
#endif
	ret

do_store_sp:
	movw z_spl,opl
	ret

do_store_pc:
	movw z_pcl,opl
	ret

do_store_ret:
	rcall do_op_pop16
	movw z_pcl,opl
	ret

do_store_call:
	push opl
	push oph
	movw opl,z_pcl
	rcall do_op_push16
	pop z_pch
	pop z_pcl
	ret

do_store_am:
	movw xl,opl
	mov temp,z_a
	rcall memWriteByte
	ret


; ------------ Operation phase stuff -----------------


.equ OP_NOP		= (0<<10)
.equ OP_INC		= (1<<10)
.equ OP_DEC		= (2<<10)
.equ OP_INC16	= (3<<10)
.equ OP_DEC16	= (4<<10)
.equ OP_RLC 	= (5<<10)
.equ OP_RRC 	= (6<<10)
.equ OP_RR	 	= (7<<10)
.equ OP_RL		= (8<<10)
.equ OP_ADDA	= (9<<10)
.equ OP_ADCA	= (10<<10)
.equ OP_SUBFA	= (11<<10)
.equ OP_SBCFA	= (12<<10)
.equ OP_ANDA	= (13<<10)
.equ OP_ORA		= (14<<10)
.equ OP_XORA	= (15<<10)
.equ OP_ADDHL	= (16<<10)
.equ OP_STHL	= (17<<10) ;store HL in fetched address
.equ OP_RMEM16	= (18<<10) ;read mem at fetched address
.equ OP_RMEM8	= (19<<10) ;read mem at fetched address
.equ OP_DA		= (20<<10)
.equ OP_SCF		= (21<<10)
.equ OP_CPL		= (22<<10)
.equ OP_CCF		= (23<<10)
.equ OP_POP16	= (24<<10)
.equ OP_PUSH16	= (25<<10)
.equ OP_IFNZ	= (26<<10)
.equ OP_IFZ		= (27<<10)
.equ OP_IFNC	= (28<<10)
.equ OP_IFC		= (29<<10)
.equ OP_IFPO	= (30<<10)
.equ OP_IFPE	= (31<<10)
.equ OP_IFP		= (32<<10)
.equ OP_IFM		= (33<<10)
.equ OP_OUTA	= (34<<10)
.equ OP_IN		= (35<<10)
.equ OP_EXHL	= (36<<10)
.equ OP_DI		= (37<<10)
.equ OP_EI		= (38<<10)
.equ OP_INV		= (39<<10)
.equ OP_CPFA	= (40<<10)
.equ OP_INCA	= (41<<10)
.equ OP_DECA	= (42<<10)

opjumps:
	rjmp do_op_nop
	rjmp do_op_inc
	rjmp do_op_dec
	rjmp do_op_inc16
	rjmp do_op_dec16
	rjmp do_op_rlc
	rjmp do_op_rrc
	rjmp do_op_rr
	rjmp do_op_rl
	rjmp do_op_adda
	rjmp do_op_adca
	rjmp do_op_subfa
	rjmp do_op_sbcfa
	rjmp do_op_anda
	rjmp do_op_ora
	rjmp do_op_xora
	rjmp do_op_addhl
	rjmp do_op_sthl
	rjmp do_op_rmem16
	rjmp do_op_rmem8
	rjmp do_op_da
	rjmp do_op_scf
	rjmp do_op_cpl
	rjmp do_op_ccf
	rjmp do_op_pop16
	rjmp do_op_push16
	rjmp do_op_ifnz
	rjmp do_op_ifz
	rjmp do_op_ifnc
	rjmp do_op_ifc
	rjmp do_op_ifpo
	rjmp do_op_ifpe
	rjmp do_op_ifp
	rjmp do_op_ifm
	rjmp do_op_outa
	rjmp do_op_in
	rjmp do_op_exhl
	rjmp do_op_di
	rjmp do_op_ei
	rjmp do_op_inv
	rjmp do_op_cpfa
	rjmp do_op_inca
	rjmp do_op_deca


;How the flags are supposed to work:
;7 ZFL_S - Sign flag (=MSBit of result)
;6 ZFL_Z - Zero flag. Is 1 when the result is 0
;4 ZFL_H - Half-carry (carry from bit 3 to 4)
;2 ZFL_P - Parity/2-complement Overflow
;1 ZFL_N - Subtract - set if last op was a subtract
;0 ZFL_C - Carry
;
;I sure hope I got the mapping between flags and instructions correct...

;----------------------------------------------------------------
;|                                                              |
;|                            Zilog                             |
;|                                                              |
;|                 ZZZZZZZ    88888      000                    |
;|                      Z    8     8    0   0                   |
;|                     Z     8     8   0   0 0                  |
;|                    Z       88888    0  0  0                  |
;|                   Z       8     8   0 0   0                  |
;|                  Z        8     8    0   0                   |
;|                 ZZZZZZZ    88888      000                    |
;|                                                              |
;|          Z80 MICROPROCESSOR Instruction Set Summary          |
;|                                                              |
;----------------------------------------------------------------
;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;|----------+------+---------------------+----------------------|
;|ADC A,s   |***V0*|Add with Carry       |A=A+s+CY              |
;|ADC HL,ss |**?V0*|Add with Carry       |HL=HL+ss+CY           |
;|ADD A,s   |***V0*|Add                  |A=A+s                 |
;|ADD HL,ss |--?-0*|Add                  |HL=HL+ss              |
;|ADD IX,pp |--?-0*|Add                  |IX=IX+pp              |
;|ADD IY,rr |--?-0*|Add                  |IY=IY+rr              |
;|AND s     |**1P00|Logical AND          |A=A&s                 |
;|BIT b,m   |?*1?0-|Test Bit             |m&{2^b}               |
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|CALL nn   |------|Unconditional Call   |-[SP]=PC,PC=nn        |
;|CCF       |--?-0*|Complement Carry Flag|CY=~CY                |
;|CP s      |***V1*|Compare              |A-s                   |
;|CPD       |****1-|Compare and Decrement|A-[HL],HL=HL-1,BC=BC-1|
;|CPDR      |****1-|Compare, Dec., Repeat|CPD till A=[HL]or BC=0|
;|CPI       |****1-|Compare and Increment|A-[HL],HL=HL+1,BC=BC-1|
;|CPIR      |****1-|Compare, Inc., Repeat|CPI till A=[HL]or BC=0|
;|CPL       |--1-1-|Complement           |A=~A                  |
;|DAA       |***P-*|Decimal Adjust Acc.  |A=BCD format          |
;|DEC s     |***V1-|Decrement            |s=s-1                 |
;|DEC xx    |------|Decrement            |xx=xx-1               |
;|DEC ss    |------|Decrement            |ss=ss-1               |
;|DI        |------|Disable Interrupts   |                      |
;|DJNZ e    |------|Dec., Jump Non-Zero  |B=B-1 till B=0        |
;|EI        |------|Enable Interrupts    |                      |
;|EX [SP],HL|------|Exchange             |[SP]<->HL             |
;|EX [SP],xx|------|Exchange             |[SP]<->xx             |
;|EX AF,AF' |------|Exchange             |AF<->AF'              |
;|EX DE,HL  |------|Exchange             |DE<->HL               |
;|EXX       |------|Exchange             |qq<->qq'   (except AF)|
;|HALT      |------|Halt                 |                      |
;|IM n      |------|Interrupt Mode       |             (n=0,1,2)|
;|IN A,[n]  |------|Input                |A=[n]                 |
;|IN r,[C]  |***P0-|Input                |r=[C]                 |
;|INC r     |***V0-|Increment            |r=r+1                 |
;|INC [HL]  |***V0-|Increment            |[HL]=[HL]+1           |
;|INC xx    |------|Increment            |xx=xx+1               |
;|INC [xx+d]|***V0-|Increment            |[xx+d]=[xx+d]+1       |
;|INC ss    |------|Increment            |ss=ss+1               |
;|IND       |?*??1-|Input and Decrement  |[HL]=[C],HL=HL-1,B=B-1|
;|INDR      |?1??1-|Input, Dec., Repeat  |IND till B=0          |
;|INI       |?*??1-|Input and Increment  |[HL]=[C],HL=HL+1,B=B-1|
;|INIR      |?1??1-|Input, Inc., Repeat  |INI till B=0          |
;|JP [HL]   |------|Unconditional Jump   |PC=[HL]               |
;|JP [xx]   |------|Unconditional Jump   |PC=[xx]               |
;|JP nn     |------|Unconditional Jump   |PC=nn                 |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|JR e      |------|Unconditional Jump   |PC=PC+e               |
;|JR cc,e   |------|Conditional Jump     |If cc JR(cc=C,NC,NZ,Z)|
;|LD dst,src|------|Load                 |dst=src               |
;|LD A,i    |**0*0-|Load                 |A=i            (i=I,R)|
;|LDD       |--0*0-|Load and Decrement   |[DE]=[HL],HL=HL-1,#   |
;|LDDR      |--000-|Load, Dec., Repeat   |LDD till BC=0         |
;|LDI       |--0*0-|Load and Increment   |[DE]=[HL],HL=HL+1,#   |
;|LDIR      |--000-|Load, Inc., Repeat   |LDI till BC=0         |
;|NEG       |***V1*|Negate               |A=-A                  |
;|NOP       |------|No Operation         |                      |
;|OR s      |**0P00|Logical inclusive OR |A=Avs                 |
;|OTDR      |?1??1-|Output, Dec., Repeat |OUTD till B=0         |
;|OTIR      |?1??1-|Output, Inc., Repeat |OUTI till B=0         |
;|OUT [C],r |------|Output               |[C]=r                 |
;|OUT [n],A |------|Output               |[n]=A                 |
;|OUTD      |?*??1-|Output and Decrement |[C]=[HL],HL=HL-1,B=B-1|
;|OUTI      |?*??1-|Output and Increment |[C]=[HL],HL=HL+1,B=B-1|
;|POP xx    |------|Pop                  |xx=[SP]+              |
;|POP qq    |------|Pop                  |qq=[SP]+              |
;|PUSH xx   |------|Push                 |-[SP]=xx              |
;|PUSH qq   |------|Push                 |-[SP]=qq              |
;|RES b,m   |------|Reset bit            |m=m&{~2^b}            |
;|RET       |------|Return               |PC=[SP]+              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;|RETI      |------|Return from Interrupt|PC=[SP]+              |
;|RETN      |------|Return from NMI      |PC=[SP]+              |
;|RL m      |**0P0*|Rotate Left          |m={CY,m}<-            |
;|RLA       |--0-0*|Rotate Left Acc.     |A={CY,A}<-            |
;|RLC m     |**0P0*|Rotate Left Circular |m=m<-                 |
;|RLCA      |--0-0*|Rotate Left Circular |A=A<-                 |
;|RLD       |**0P0-|Rotate Left 4 bits   |{A,[HL]}={A,[HL]}<- ##|
;|RR m      |**0P0*|Rotate Right         |m=->{CY,m}            |
;|RRA       |--0-0*|Rotate Right Acc.    |A=->{CY,A}            |
;|RRC m     |**0P0*|Rotate Right Circular|m=->m                 |
;|RRCA      |--0-0*|Rotate Right Circular|A=->A                 |
;|RRD       |**0P0-|Rotate Right 4 bits  |{A,[HL]}=->{A,[HL]} ##|
;|RST p     |------|Restart              | (p=0H,8H,10H,...,38H)|
;|SBC A,s   |***V1*|Subtract with Carry  |A=A-s-CY              |
;|SBC HL,ss |**?V1*|Subtract with Carry  |HL=HL-ss-CY           |
;|SCF       |--0-01|Set Carry Flag       |CY=1                  |
;|SET b,m   |------|Set bit              |m=mv{2^b}             |
;|SLA m     |**0P0*|Shift Left Arithmetic|m=m*2                 |
;|SRA m     |**0P0*|Shift Right Arith.   |m=m/2                 |
;|SRL m     |**0P0*|Shift Right Logical  |m=->{0,m,CY}          |
;|SUB s     |***V1*|Subtract             |A=A-s                 |
;|XOR s     |**0P00|Logical Exclusive OR |A=Axs                 |
;|----------+------+--------------------------------------------|
;| F        |-*01? |Flag unaffected/affected/reset/set/unknown  |
;| S        |S     |Sign flag (Bit 7)                           |
;| Z        | Z    |Zero flag (Bit 6)                           |
;| HC       |  H   |Half Carry flag (Bit 4)                     |
;| P/V      |   P  |Parity/Overflow flag (Bit 2, V=overflow)    |
;| N        |    N |Add/Subtract flag (Bit 1)                   |
;| CY       |     C|Carry flag (Bit 0)                          |
;|-----------------+--------------------------------------------|
;| n               |Immediate addressing                        |
;| nn              |Immediate extended addressing               |
;| e               |Relative addressing (PC=PC+2+offset)        |
;| [nn]            |Extended addressing                         |
;| [xx+d]          |Indexed addressing                          |
;| r               |Register addressing                         |
;| [rr]            |Register indirect addressing                |
;|                 |Implied addressing                          |
;| b               |Bit addressing                              |
;| p               |Modified page zero addressing (see RST)     |
;|-----------------+--------------------------------------------|
;|DEFB n(,...)     |Define Byte(s)                              |
;|DEFB 'str'(,...) |Define Byte ASCII string(s)                 |
;|DEFS nn          |Define Storage Block                        |
;|DEFW nn(,...)    |Define Word(s)                              |
;|-----------------+--------------------------------------------|
;| A  B  C  D  E   |Registers (8-bit)                           |
;| AF  BC  DE  HL  |Register pairs (16-bit)                     |
;| F               |Flag register (8-bit)                       |
;| I               |Interrupt page address register (8-bit)     |
;| IX IY           |Index registers (16-bit)                    |
;| PC              |Program Counter register (16-bit)           |
;| R               |Memory Refresh register                     |
;| SP              |Stack Pointer register (16-bit)             |
;|-----------------+--------------------------------------------|
;| b               |One bit (0 to 7)                            |
;| cc              |Condition (C,M,NC,NZ,P,PE,PO,Z)             |
;| d               |One-byte expression (-128 to +127)          |
;| dst             |Destination s, ss, [BC], [DE], [HL], [nn]   |
;| e               |One-byte expression (-126 to +129)          |
;| m               |Any register r, [HL] or [xx+d]              |
;| n               |One-byte expression (0 to 255)              |
;| nn              |Two-byte expression (0 to 65535)            |
;| pp              |Register pair BC, DE, IX or SP              |
;| qq              |Register pair AF, BC, DE or HL              |
;| qq'             |Alternative register pair AF, BC, DE or HL  |
;| r               |Register A, B, C, D, E, H or L              |
;| rr              |Register pair BC, DE, IY or SP              |
;| s               |Any register r, value n, [HL] or [xx+d]     |
;| src             |Source s, ss, [BC], [DE], [HL], nn, [nn]    |
;| ss              |Register pair BC, DE, HL or SP              |
;| xx              |Index register IX or IY                     |
;|-----------------+--------------------------------------------|
;| +  -  *  /  ^   |Add/subtract/multiply/divide/exponent       |
;| &  ~  v  x      |Logical AND/NOT/inclusive OR/exclusive OR   |
;| <-  ->          |Rotate left/right                           |
;| [ ]             |Indirect addressing                         |
;| [ ]+  -[ ]      |Indirect addressing auto-increment/decrement|
;| { }             |Combination of operands                     |
;| #               |Also BC=BC-1,DE=DE-1                        |
;| ##              |Only lower 4 bits of accumulator A used     |
;----------------------------------------------------------------


.equ AVR_T = SREG_T
.equ AVR_H = SREG_H
.equ AVR_S = SREG_S
.equ AVR_V = SREG_V
.equ AVR_N = SREG_N
.equ AVR_Z = SREG_Z
.equ AVR_C = SREG_C

;------------------------------------------------;
; Move single bit between two registers
;
;	bmov	dstreg,dstbit,srcreg.srcbit

.macro	bmov
	bst	@2,@3
	bld	@0,@1
.endm


;------------------------------------------------;
; Load table value from flash indexed by source reg.
;
;	ldpmx	dstreg,tablebase,indexreg
;
; (6 words, 8 cycles)

.macro	ldpmx
	ldi	zh,high(@1*2)	; table must be page aligned
	mov	zl,@2		       
	lpm	@0,z	
.endm
.macro	do_z80_flags_HP
#if EM_Z80
	bmov	z_flags, ZFL_P, temp, AVR_V
	bmov	z_flags, ZFL_H, temp, AVR_H
#endif
.endm

.macro	do_z80_flags_set_N
#if EM_Z80
	ori	z_flags, (1<<ZFL_N)       ; Negation auf 1
#endif
.endm

.macro	do_z80_flags_set_HN
#if EM_Z80
	ori 	z_flags,(1<<ZFL_N)|(1<<ZFL_H)
#endif
.endm

.macro	do_z80_flags_clear_N
#if EM_Z80
	andi	z_flags,~(1<<ZFL_N)
#endif
.endm

.macro	do_z80_flags_op_rotate
	; must not change avr carry flag!
#if EM_Z80
	andi   z_flags, ~( (1<<ZFL_H) | (1<<ZFL_N) | (1<<ZFL_C) )
#else
	andi   z_flags, ~( (1<<ZFL_C) )
#endif
.endm

.macro	do_z80_flags_op_and
#if EM_Z80
	ori	z_flags,(1<<ZFL_H)
#else
	ori	z_flags,(1<<ZFL_H)
#endif
.endm

.macro	do_z80_flags_op_or
#if EM_Z80
#endif
.endm


do_op_nop:
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|INC r     |***V0-|Increment            |r=r+1                 |
;|INC [HL]  |***V0-|Increment            |[HL]=[HL]+1           |
;|INC [xx+d]|***V0-|Increment            |[xx+d]=[xx+d]+1       |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|INC r     |**-P0-|Increment            |r=r+1                 |
;|INC [HL]  |**-P0-|Increment            |[HL]=[HL]+1           |
;
; 
do_op_inc:
	inc	opl
#if EM_Z80
	in	temp, sreg
#endif
	andi	z_flags,(1<<ZFL_H)|(1<<ZFL_C)	; preserve C-, and H-flag
	ldpmx	temp2, sz53p_tab, opl
	or	z_flags,temp2		;
	do_z80_flags_HP
	ret

do_op_inca:
	inc	z_a
#if EM_Z80
	in	temp, sreg
#endif
	andi	z_flags,(1<<ZFL_H)|(1<<ZFL_C)	; preserve C-, and H-flag
	ldpmx	temp2, sz53p_tab, z_a
	or	z_flags,temp2		;
	do_z80_flags_HP
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|DEC r     |***V1-|Decrement            |s=s-1                 |
;|INC [HL]  |***V0-|Increment            |[HL]=[HL]+1           |
;|INC [xx+d]|***V0-|Increment            |[xx+d]=[xx+d]+1       |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|DEC r     |**-P -|Increment            |r=r+1                 |
;|DEC [HL]  |**-P -|Increment            |[HL]=[HL]+1           |
;
;
do_op_dec:
	dec	opl
#if EM_Z80
	in    temp, sreg
#endif
	andi	z_flags,(1<<ZFL_H)|(1<<ZFL_C)	; preserve C-, and H-flag
	ldpmx	temp2, sz53p_tab, opl
	or	z_flags,temp2		;
	do_z80_flags_HP
	do_z80_flags_set_N
	ret

do_op_deca:
	dec	z_a
#if EM_Z80
	in    temp, sreg
#endif
	andi	z_flags,(1<<ZFL_H)|(1<<ZFL_C)	; preserve C-, and H-flag
	ldpmx	temp2, sz53p_tab, z_a
	or	z_flags,temp2		;
	do_z80_flags_HP
	do_z80_flags_set_N
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|INC xx    |------|Increment            |xx=xx+1               |
;|INC ss    |------|Increment            |ss=ss+1               |
;
; 
do_op_inc16:
	subi	opl,low(-1)
	sbci	oph,high(-1)
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|DEC xx    |------|Decrement            |xx=xx-1               |
;|DEC ss    |------|Decrement            |ss=ss-1               |
;
; 
do_op_dec16:
	subi   opl, 1
	sbci   oph, 0
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|RLCA      |--0-0*|Rotate Left Circular |A=A<-                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|RLCA      |---- *|Rotate Left Circular |A=A<-                 |
;
;
do_op_rlc:
	;Rotate Left Cyclical. All bits move 1 to the 
	;left, the msb becomes c and lsb.
	do_z80_flags_op_rotate
	lsl    opl
	brcc   do_op_rlc_noc
	ori    opl, 1
	ori    z_flags, (1<<ZFL_C)
do_op_rlc_noc:
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|RRCA      |--0-0*|Rotate Right Circular|A=->A                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|RRCA      |---- *|Rotate Right Circular|A=->A                 |
;
;
do_op_rrc: 
	;Rotate Right Cyclical. All bits move 1 to the 
	;right, the lsb becomes c and msb.
	do_z80_flags_op_rotate
	lsr    opl
	brcc   do_op_rrc_noc
	ori    opl, 0x80
	ori    z_flags, (1<<ZFL_C)
do_op_rrc_noc:
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|RRA       |--0-0*|Rotate Right Acc.    |A=->{CY,A}            |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|RRA       |---- *|Rotate Right Acc.    |A=->{CY,A}            |
;
; 
do_op_rr: 
	;Rotate Right. All bits move 1 to the right, the lsb 
	;becomes c, c becomes msb.
	clc				; get z80 carry to avr carry
	sbrc    z_flags,ZFL_C
	sec
	do_z80_flags_op_rotate		; (clear ZFL_C, doesn't change AVR_C)
	bmov	z_flags,ZFL_C, opl,0	; Bit 0 --> CY
	ror     opl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|RLA       |--0-0*|Rotate Left Acc.     |A={CY,A}<-            |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|RLA       |---- *|Rotate Left Acc.     |A={CY,A}<-            |
;
; 
do_op_rl:
	;Rotate Left. All bits move 1 to the left, the msb 
	;becomes c, c becomes lsb.
	clc
	sbrc z_flags,ZFL_C
	 sec
	do_z80_flags_op_rotate		; (clear ZFL_C, doesn't change AVR_C)
	bmov	z_flags,ZFL_C, opl,7	; Bit 7 --> CY
	rol opl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|ADD A,s   |***V0*|Add                  |A=A+s                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|ADD A,s   |***P *|Add                  |A=A+s                 |
;
;
do_op_adda:
	add z_a,opl
	in temp,sreg
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,P flag
	bmov	z_flags,ZFL_C, temp,AVR_C
	do_z80_flags_HP
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|ADC A,s   |***V0*|Add with Carry       |A=A+s+CY              |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|ADC A,s   |***P *|Add with Carry       |A=A+s+CY              |
;
;
do_op_adca:
	clc
	sbrc z_flags,ZFL_C
	 sec
	adc z_a,opl
	in temp,sreg
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,P
	bmov	z_flags,ZFL_C, temp,AVR_C
	do_z80_flags_HP
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|SUB s     |***V1*|Subtract             |A=A-s                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|SUB s     |***P *|Subtract             |A=A-s                 |

;
do_op_subfa:
	sub z_a,opl
	in temp,sreg
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,P
	bmov	z_flags,ZFL_C, temp,AVR_C
	do_z80_flags_HP
	do_z80_flags_set_N
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CP s      |***V1*|Compare              |A-s                   |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|CP s      |***P *|Compare              |A-s                   |

;
do_op_cpfa:
	mov temp,z_a
	sub temp,opl
	mov opl,temp
	in temp,sreg
	ldpmx	z_flags,sz53p_tab,opl		;S,Z,P
	bmov	z_flags,ZFL_C, temp,AVR_C
	do_z80_flags_HP
	do_z80_flags_set_N
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|SBC A,s   |***V1*|Subtract with Carry  |A=A-s-CY              |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|SBC A,s   |***P *|Subtract with Carry  |A=A-s-CY              |
;
;
do_op_sbcfa:
	clc
	sbrc z_flags,ZFL_C
	 sec
	sbc z_a,opl
	in temp,sreg
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,P
	bmov	z_flags,ZFL_C, temp,AVR_C
	do_z80_flags_HP
	do_z80_flags_set_N
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|AND s     |**1P00|Logical AND          |A=A&s                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|AND s     |**-P 0|Logical AND          |A=A&s                 |
;
; TODO H-Flag
do_op_anda:
	and z_a,opl				;
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,P,N,C
	do_z80_flags_op_and
	ret


;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|OR s      |**0P00|Logical inclusive OR |A=Avs                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|OR s      |**-P00|Logical inclusive OR |A=Avs                 |
;
; TODO: H-Flag
do_op_ora:
	or z_a,opl
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,H,P,N,C
	do_z80_flags_op_or
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|XOR s     |**0P00|Logical Exclusive OR |A=Axs                 |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|XOR s     |**-P 0|Logical Exclusive OR |A=Axs                 |
;
; TODO: H-Flag
do_op_xora:
	eor z_a,opl
	ldpmx	z_flags,sz53p_tab,z_a		;S,Z,H,P,N,C
	do_z80_flags_op_or
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|ADD HL,ss |--?-0*|Add                  |HL=HL+ss              |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|ADD HL,ss |---- *|Add                  |HL=HL+ss              |
;
;
do_op_addhl:
	add opl,z_l
	adc oph,z_h
	in temp,sreg
	bmov	z_flags,ZFL_H, temp,AVR_H
	bmov	z_flags,ZFL_C, temp,AVR_C
	do_z80_flags_clear_N
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|LD dst,src|------|Load                 |dst=src               |
;
;
do_op_sthl: ;store hl to mem loc in opl:h
	movw xl,opl
#if DRAM_WORD_ACCESS
	movw temp,z_l
	rcall memWriteWord
#else
	mov temp,z_l
	rcall memWriteByte
	adiw xl,1
	mov temp,z_h
	rcall memWriteByte
#endif
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|LD dst,src|------|Load                 |dst=src               |
;
; 
do_op_rmem16:
	movw xl,opl
#if DRAM_WORD_ACCESS
	rcall memReadWord
	movw opl,temp
#else
	rcall memReadByte
	mov opl,temp
	adiw xl,1
	rcall memReadByte
	mov oph,temp
#endif	
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|LD dst,src|------|Load                 |dst=src               |
;
;
do_op_rmem8:
	movw xl,opl
	rcall memReadByte
	mov opl,temp
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|DAA       |***P-*|Decimal Adjust Acc.  |                      |
;|----------|SZHP C|---------- 8080 ----------------------------|
;
; Not yet checked

; Description (http://www.z80.info/z80syntx.htm#DAA):
;  This instruction conditionally adjusts the accumulator for BCD addition
;  and subtraction operations. For addition (ADD, ADC, INC) or subtraction
;  (SUB, SBC, DEC, NEC), the following table indicates the operation performed:
;
; -------------------------------------------------------------------------------
; |          | C Flag  | HEX value in | H Flag | HEX value in | Number  | C flag|
; | Operation| Before  | upper digit  | Before | lower digit  | added   | After |
; |          | DAA     | (bit 7-4)    | DAA    | (bit 3-0)    | to byte | DAA   |
; |-----------------------------------------------------------------------------|
; |          |    0    |     0-9      |   0    |     0-9      |   00    |   0   |
; |   ADD    |    0    |     0-8      |   0    |     A-F      |   06    |   0   |
; |          |    0    |     0-9      |   1    |     0-3      |   06    |   0   |
; |   ADC    |    0    |     A-F      |   0    |     0-9      |   60    |   1   |
; |          |    0    |     9-F      |   0    |     A-F      |   66    |   1   |
; |   INC    |    0    |     A-F      |   1    |     0-3      |   66    |   1   |
; |          |    1    |     0-2      |   0    |     0-9      |   60    |   1   |
; |          |    1    |     0-2      |   0    |     A-F      |   66    |   1   |
; |          |    1    |     0-3      |   1    |     0-3      |   66    |   1   |
; |-----------------------------------------------------------------------------|
; |   SUB    |    0    |     0-9      |   0    |     0-9      |   00    |   0   |
; |   SBC    |    0    |     0-8      |   1    |     6-F      |   FA    |   0   |
; |   DEC    |    1    |     7-F      |   0    |     0-9      |   A0    |   1   |
; |   NEG    |    1    |     6-F      |   1    |     6-F      |   9A    |   1   |
; |-----------------------------------------------------------------------------|
;
; Flags:
;     C:   See instruction.
;     N:   Unaffected.
;     P/V: Set if Acc. is even parity after operation, reset otherwise.
;     H:   See instruction.
;     Z:   Set if Acc. is Zero after operation, reset otherwise.
;     S:   Set if most significant bit of Acc. is 1 after operation, reset otherwise.



#if 1
do_op_da:
	ldi 	oph,0				; what to add
	sbrc	z_flags,ZFL_H			; if H-Flag
	rjmp	op_da_06
	mov	temp,opl
	andi	temp,0x0f			; ... or lower digit > 9
	cpi	temp,0x0a
	brlo	op_da_06n
op_da_06:				
	ori	oph,0x06
op_da_06n:				
	sbrc	z_flags,(1<<ZFL_C)
	rjmp	op_da_60
	cpi	opl,0xa0
	brlo	op_da_60n
op_da_60:				
	ori	oph,0x60
op_da_60n:				
	cpi	opl,0x9a
	brlo	op_da_99n
	ori	z_flags,(1<<ZFL_C); set C
op_da_99n:
	sbrs	z_flags,ZFL_N			; if sub-op
	rjmp	op_da_add			; then
	sub	opl,oph
	rjmp	op_da_ex
op_da_add:					; else add-op
	cpi	opl,0x91
	brlo	op_da_60n2
	mov	temp,opl
	andi	temp,0x0f
	cpi	temp,0x0a
	brlo	op_da_60n2
	ori	oph,0x60
op_da_60n2:
	add	opl,oph
op_da_ex:
	in	temp,SREG	
	sbrc	temp,AVR_H
	ori	z_flags,(1<<ZFL_C)
	andi	z_flags,(1<<ZFL_N)|(1<<ZFL_C)	; preserve C,N
	ldpmx	temp2, sz53p_tab, opl		; get S,Z,P
	or	z_flags,temp2
	bmov	z_flags,ZFL_H, temp,AVR_H	; H  (?)
	ret
#else

do_op_da:
	sbrc	z_flags,ZFL_N			; if add-op	
	rjmp	do_op_da_sub			; then
	ldi		temp2,0			;
	mov		temp,opl		;
	andi	temp,0x0f			;
	cpi		temp,0x0a		;	if lower digit > 9
	brlo	do_op_da_h			;
	ori		temp2,0x06		;		add 6 to lower digit
do_op_da_h:					;
	sbrc	z_flags,ZFL_H			;   ... or H-Flag
	ori		temp2,0x06		;
	add		opl,temp2		;

	ldi		temp2,0			;
	mov		temp,opl		;
	andi	temp,0xf0			;
	cpi		temp,0xa0		;
	brlo	do_op_da_c			;
	ori		temp2,0x60		;
do_op_da_c:					; else sub-op
	sbrc	z_flags,ZFL_C			;
	ori		temp2,0x60		;
	andi	z_flags, ~( (1<<ZFL_S) | (1<<ZFL_Z) | (1<<ZFL_H) )
	add		opl,temp2		;
	in		temp,SREG		;
	bst		temp,AVR_Z		;Z-Flag
	bld		z_flags,ZFL_Z		;
	bst		temp,AVR_N		;S-Flag
	bst		z_flags,ZFL_S		;
	sbrc	temp2,5				;C-Flag, set if 0x06 added
	ori		z_flags,(1<<ZFL_C)	;
						;H-Flag?
	ret
	
do_op_da_sub:					;TODO:
	rcall do_op_inv
	ret
#endif

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|SCF       |--0-01|Set Carry Flag       |CY=1                  |
;|----------|SZHP C|---------- 8080 ----------------------------|
;
;
do_op_scf:
	andi	z_flags,~((1<<ZFL_H)|(1<<ZFL_N))
	ori 	z_flags,(1<<ZFL_C)
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CCF       |--?-0*|Complement Carry Flag|CY=~CY                |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|SCF       |---- 1|Set Carry Flag       |CY=1                  |
;
;TODO: H-Flag
do_op_ccf:
	do_z80_flags_clear_N
	ldi temp,(1<<ZFL_C)
	eor z_flags,temp
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CPL       |--1-1-|Complement           |A=~A                  |
;|----------|SZHP C|---------- 8080 ----------------------------|
;|CPL       |---- -|Complement           |A=~A                  |
;
;
do_op_cpl:
	com z_a
	do_z80_flags_set_HN
	ret


;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|PUSH xx   |------|Push                 |-[SP]=xx              |
;|PUSH qq   |------|Push                 |-[SP]=qq              |
;
;
do_op_push16:
	movw xl,z_spl
	subi xl,2
	sbci xh,0
	movw z_spl,xl
#if DRAM_WORD_ACCESS	
	movw temp,opl
	rcall memWriteWord
#else
	mov temp,opl
	rcall memWriteByte
	adiw xl,1
	mov temp,oph
	rcall memWriteByte
#endif

.if STACK_DBG
	rcall printstr
	.db "Stack push ",0
	mov temp,oph
	rcall printhex
	mov temp,opl
	rcall printhex
	rcall printstr
	.db ", SP is now ",0
	mov temp,z_sph
	rcall printhex
	mov temp,z_spl
	rcall printhex
	rcall printstr
	.db ".",13,0
.endif

	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|POP xx    |------|Pop                  |xx=[SP]+              |
;|POP qq    |------|Pop                  |qq=[SP]+              |
;
;
do_op_pop16:
	movw xl,z_spl
#if DRAM_WORD_ACCESS
	rcall memReadWord
	movw opl,temp
#else
	rcall memReadByte
	mov opl,temp
	adiw xl,1
	rcall memReadByte
	mov oph,temp
#endif	

	ldi temp,2
	add z_spl,temp
	adc z_sph,_0


.if STACK_DBG
	rcall printstr
	.db "Stack pop: val ",0
	mov temp,oph
	rcall printhex
	mov temp,opl
	rcall printhex
	rcall printstr
	.db ", SP is now",0
	mov temp,z_sph
	rcall printhex
	mov temp,z_spl
	rcall printhex
	rcall printstr
	.db ".",13,0
.endif
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|EX [SP],HL|------|Exchange             |[SP]<->HL             |
;|EX DE,HL  |------|Exchange             |DE<->HL               |
;
; 
do_op_exhl:
	mov temp,z_h
	mov z_h,oph
	mov oph,temp
	mov temp,z_l
	mov z_l,opl
	mov opl,temp
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;
; TODO: Implement IFF1, IFF2
do_op_di:
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;
; TODO: Implement IFF1, IFF2
do_op_ei:
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifnz:
	sbrs z_flags, ZFL_Z
	ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifz:
	sbrc z_flags, ZFL_Z
	ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifnc:
	sbrs z_flags, ZFL_C
	ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifc:
	sbrc z_flags, ZFL_C
	ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifpo:
	sbrs z_flags, ZFL_P
	ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifpe:
	sbrc z_flags, ZFL_P
	ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifp: ;sign positive, aka s=0
	sbrs z_flags, ZFL_S
	 ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|CALL cc,nn|------|Conditional Call     |If cc CALL            |
;|JP cc,nn  |------|Conditional Jump     |If cc JP              |
;|RET cc    |------|Conditional Return   |If cc RET             |
;
;
do_op_ifm: ;sign negative, aka s=1
	sbrc z_flags, ZFL_S
	 ret
	clr insdech
	clr insdecl
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|OUT [n],A |------|Output               |[n]=A                 |
;
;
;Interface with peripherials goes here :)
do_op_outa: ; out (opl),a
.if PORT_DEBUG
	rcall printstr
	.db 13,"Port write: ",0
	mov temp,z_a
	rcall printhex
	rcall printstr
	.db " -> (",0
	mov temp,opl
	rcall printhex
	rcall printstr
	.db ")",13,0
.endif
	mov temp,z_a
	mov temp2,opl
	rcall portWrite
	ret

;----------------------------------------------------------------
;|Mnemonic  |SZHPNC|Description          |Notes                 |
;----------------------------------------------------------------
;|IN A,[n]  |------|Input                |A=[n]                 |
;
;
do_op_in:	; in a,(opl)
.if PORT_DEBUG
	rcall printstr
	.db 13,"Port read: (",0
	mov temp,opl
	rcall printhex
	rcall printstr
	.db ") -> ",0
.endif

	mov temp2,opl
	rcall portRead
	mov opl,temp

.if PORT_DEBUG
	rcall printhex
	rcall printstr
	.db 13,0
.endif
	ret


;----------------------------------------------------------------
do_op_inv:
	rcall printstr
	.db "Invalid opcode @ PC=",0,0
	mov   temp,z_pch
	rcall printhex
	mov   temp,z_pcl
	rcall printhex

;----------------------------------------------------------------
haltinv:
	rjmp haltinv
	 
;----------------------------------------------------------------
; Lookup table, stolen from z80ex, Z80 emulation library.
; http://z80ex.sourceforge.net/

; The S, Z, 5 and 3 bits and the parity of the lookup value 
.org (PC+255) & 0xff00
sz53p_tab:
	.db 0x44,0x00,0x00,0x04,0x00,0x04,0x04,0x00
	.db 0x08,0x0c,0x0c,0x08,0x0c,0x08,0x08,0x0c
	.db 0x00,0x04,0x04,0x00,0x04,0x00,0x00,0x04
	.db 0x0c,0x08,0x08,0x0c,0x08,0x0c,0x0c,0x08
	.db 0x20,0x24,0x24,0x20,0x24,0x20,0x20,0x24
	.db 0x2c,0x28,0x28,0x2c,0x28,0x2c,0x2c,0x28
	.db 0x24,0x20,0x20,0x24,0x20,0x24,0x24,0x20
	.db 0x28,0x2c,0x2c,0x28,0x2c,0x28,0x28,0x2c
	.db 0x00,0x04,0x04,0x00,0x04,0x00,0x00,0x04
	.db 0x0c,0x08,0x08,0x0c,0x08,0x0c,0x0c,0x08
	.db 0x04,0x00,0x00,0x04,0x00,0x04,0x04,0x00
	.db 0x08,0x0c,0x0c,0x08,0x0c,0x08,0x08,0x0c
	.db 0x24,0x20,0x20,0x24,0x20,0x24,0x24,0x20
	.db 0x28,0x2c,0x2c,0x28,0x2c,0x28,0x28,0x2c
	.db 0x20,0x24,0x24,0x20,0x24,0x20,0x20,0x24
	.db 0x2c,0x28,0x28,0x2c,0x28,0x2c,0x2c,0x28
	.db 0x80,0x84,0x84,0x80,0x84,0x80,0x80,0x84
	.db 0x8c,0x88,0x88,0x8c,0x88,0x8c,0x8c,0x88
	.db 0x84,0x80,0x80,0x84,0x80,0x84,0x84,0x80
	.db 0x88,0x8c,0x8c,0x88,0x8c,0x88,0x88,0x8c
	.db 0xa4,0xa0,0xa0,0xa4,0xa0,0xa4,0xa4,0xa0
	.db 0xa8,0xac,0xac,0xa8,0xac,0xa8,0xa8,0xac
	.db 0xa0,0xa4,0xa4,0xa0,0xa4,0xa0,0xa0,0xa4
	.db 0xac,0xa8,0xa8,0xac,0xa8,0xac,0xac,0xa8
	.db 0x84,0x80,0x80,0x84,0x80,0x84,0x84,0x80
	.db 0x88,0x8c,0x8c,0x88,0x8c,0x88,0x88,0x8c
	.db 0x80,0x84,0x84,0x80,0x84,0x80,0x80,0x84
	.db 0x8c,0x88,0x88,0x8c,0x88,0x8c,0x8c,0x88
	.db 0xa0,0xa4,0xa4,0xa0,0xa4,0xa0,0xa0,0xa4
	.db 0xac,0xa8,0xa8,0xac,0xa8,0xac,0xac,0xa8
	.db 0xa4,0xa0,0xa0,0xa4,0xa0,0xa4,0xa4,0xa0
	.db 0xa8,0xac,0xac,0xa8,0xac,0xa8,0xa8,0xac
	

; ----------------------- Opcode decoding -------------------------

; Lookup table for Z80 opcodes. Translates the first byte of the instruction word into three
; operations: fetch, do something, store.
; The table is made of 256 words. These 16-bit words consist of 
; the fetch operation (bit 0-4), the processing operation (bit 10-16) and the store 
; operation (bit 5-9).
.org (PC+255) & 0xff00
inst_table:
.dw (FETCH_NOP  | OP_NOP	| STORE_NOP)	 ; 00		NOP
.dw (FETCH_DIR16| OP_NOP	| STORE_BC )	 ; 01 nn nn	LD BC,nn
.dw (FETCH_A    | OP_NOP	| STORE_MBC)	 ; 02		LD (BC),A
.dw (FETCH_BC   | OP_INC16	| STORE_BC )	 ; 03		INC BC
.dw (FETCH_B    | OP_INC	| STORE_B  )	 ; 04       	INC B
.dw (FETCH_B    | OP_DEC	| STORE_B  )	 ; 05       	DEC B
.dw (FETCH_DIR8	| OP_NOP	| STORE_B  )	 ; 06 nn    	LD B,n
.dw (FETCH_A    | OP_RLC	| STORE_A  )	 ; 07       	RLCA
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; 08       	EX AF,AF'	(Z80)
.dw (FETCH_BC   | OP_ADDHL	| STORE_HL )	 ; 09       	ADD HL,BC
.dw (FETCH_MBC	| OP_NOP	| STORE_A  )	 ; 0A       	LD A,(BC)
.dw (FETCH_BC   | OP_DEC16	| STORE_BC )	 ; 0B       	DEC BC
.dw (FETCH_C    | OP_INC	| STORE_C  )	 ; 0C       	INC C
.dw (FETCH_C    | OP_DEC	| STORE_C  )	 ; 0D       	DEC C
.dw (FETCH_DIR8 | OP_NOP	| STORE_C  )	 ; 0E nn    	LD C,n
.dw (FETCH_A    | OP_RRC	| STORE_A  )	 ; 0F       	RRCA
.dw (FETCH_NOP  | OP_INV	| STORE_NOP)	 ; 10 oo    	DJNZ o		(Z80)
.dw (FETCH_DIR16| OP_NOP	| STORE_DE )	 ; 11 nn nn	LD DE,nn
.dw (FETCH_A    | OP_NOP	| STORE_MDE)	 ; 12		LD (DE),A
.dw (FETCH_DE	| OP_INC16	| STORE_DE )	 ; 13		INC DE
.dw (FETCH_D	| OP_INC	| STORE_D  )	 ; 14		INC D
.dw (FETCH_D	| OP_DEC	| STORE_D  )	 ; 15		DEC D
.dw (FETCH_DIR8	| OP_NOP	| STORE_D  )	 ; 16 nn	LD D,n
.dw (FETCH_A    | OP_RL		| STORE_A  )	 ; 17		RLA
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; 18 oo	JR o		(Z80)
.dw (FETCH_DE	| OP_ADDHL	| STORE_HL )	 ; 19		ADD HL,DE
.dw (FETCH_MDE	| OP_NOP	| STORE_A  )	 ; 1A		LD A,(DE)
.dw (FETCH_DE	| OP_DEC16	| STORE_DE )	 ; 1B		DEC DE
.dw (FETCH_E	| OP_INC	| STORE_E  )	 ; 1C		INC E
.dw (FETCH_E	| OP_DEC	| STORE_E  )	 ; 1D		DEC E
.dw (FETCH_DIR8	| OP_NOP	| STORE_E  )	 ; 1E nn	LD E,n
.dw (FETCH_A    | OP_RR		| STORE_A  )	 ; 1F		RRA
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; 20 oo	JR NZ,o		(Z80)
.dw (FETCH_DIR16| OP_NOP	| STORE_HL )	 ; 21 nn nn	LD HL,nn
.dw (FETCH_DIR16| OP_STHL	| STORE_NOP)	 ; 22 nn nn	LD (nn),HL
.dw (FETCH_HL	| OP_INC16	| STORE_HL )	 ; 23		INC HL
.dw (FETCH_H	| OP_INC	| STORE_H  )	 ; 24		INC H
.dw (FETCH_H	| OP_DEC	| STORE_H  )	 ; 25		DEC H
.dw (FETCH_DIR8	| OP_NOP	| STORE_H  )	 ; 26 nn	LD H,n
.dw (FETCH_A    | OP_DA		| STORE_A  )	 ; 27		DAA
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; 28 oo	JR Z,o		(Z80)
.dw (FETCH_HL	| OP_ADDHL	| STORE_HL )	 ; 29		ADD HL,HL
.dw (FETCH_DIR16| OP_RMEM16	| STORE_HL )	 ; 2A nn nn	LD HL,(nn)
.dw (FETCH_HL	| OP_DEC16	| STORE_HL )	 ; 2B		DEC HL
.dw (FETCH_L	| OP_INC	| STORE_L  )	 ; 2C		INC L
.dw (FETCH_L	| OP_DEC	| STORE_L  )	 ; 2D		DEC L
.dw (FETCH_DIR8	| OP_NOP	| STORE_L  )	 ; 2E nn	LD L,n
.dw (FETCH_NOP  | OP_CPL	| STORE_NOP)	 ; 2F		CPL
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; 30 oo	JR NC,o		(Z80)
.dw (FETCH_DIR16| OP_NOP	| STORE_SP )	 ; 31 nn nn	LD SP,nn
.dw (FETCH_DIR16| OP_NOP	| STORE_AM )	 ; 32 nn nn	LD (nn),A
.dw (FETCH_SP	| OP_INC16	| STORE_SP )	 ; 33		INC SP
.dw (FETCH_MHL	| OP_INC	| STORE_MHL)	 ; 34		INC (HL)
.dw (FETCH_MHL	| OP_DEC	| STORE_MHL)	 ; 35		DEC (HL)
.dw (FETCH_DIR8	| OP_NOP	| STORE_MHL)	 ; 36 nn	LD (HL),n
.dw (FETCH_NOP	| OP_SCF	| STORE_NOP)	 ; 37		SCF
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; 38 oo	JR C,o		(Z80)
.dw (FETCH_SP	| OP_ADDHL	| STORE_HL )	 ; 39		ADD HL,SP
.dw (FETCH_DIR16| OP_RMEM8	| STORE_A  )	 ; 3A nn nn	LD A,(nn)
.dw (FETCH_SP	| OP_DEC16	| STORE_SP )	 ; 3B		DEC SP
.dw (FETCH_NOP  | OP_INCA	| STORE_NOP)	 ; 3C		INC A
.dw (FETCH_NOP  | OP_DECA	| STORE_NOP)	 ; 3D		DEC A
.dw (FETCH_DIR8	| OP_NOP	| STORE_A  )	 ; 3E nn	LD A,n
.dw (FETCH_NOP	| OP_CCF	| STORE_NOP)	 ; 3F		CCF (Complement Carry Flag, gvd)
.dw (FETCH_B	| OP_NOP	| STORE_B  )	 ; 40		LD B,r
.dw (FETCH_C	| OP_NOP	| STORE_B  )	 ; 41		LD B,r
.dw (FETCH_D	| OP_NOP	| STORE_B  )	 ; 42		LD B,r
.dw (FETCH_E	| OP_NOP	| STORE_B  )	 ; 43		LD B,r
.dw (FETCH_H	| OP_NOP	| STORE_B  )	 ; 44		LD B,r
.dw (FETCH_L	| OP_NOP	| STORE_B  )	 ; 45		LD B,r
.dw (FETCH_MHL	| OP_NOP	| STORE_B  )	 ; 46		LD B,r
.dw (FETCH_A    | OP_NOP	| STORE_B  )	 ; 47		LD B,r
.dw (FETCH_B	| OP_NOP	| STORE_C  )	 ; 48		LD C,r
.dw (FETCH_C	| OP_NOP	| STORE_C  )	 ; 49		LD C,r
.dw (FETCH_D	| OP_NOP	| STORE_C  )	 ; 4A		LD C,r
.dw (FETCH_E	| OP_NOP	| STORE_C  )	 ; 4B		LD C,r
.dw (FETCH_H	| OP_NOP	| STORE_C  )	 ; 4C		LD C,r
.dw (FETCH_L	| OP_NOP	| STORE_C  )	 ; 4D		LD C,r
.dw (FETCH_MHL	| OP_NOP	| STORE_C  )	 ; 4E		LD C,r
.dw (FETCH_A    | OP_NOP	| STORE_C  )	 ; 4F		LD C,r
.dw (FETCH_B	| OP_NOP	| STORE_D  )	 ; 50		LD D,r
.dw (FETCH_C	| OP_NOP	| STORE_D  )	 ; 51		LD D,r
.dw (FETCH_D	| OP_NOP	| STORE_D  )	 ; 52		LD D,r
.dw (FETCH_E	| OP_NOP	| STORE_D  )	 ; 53		LD D,r
.dw (FETCH_H	| OP_NOP	| STORE_D  )	 ; 54		LD D,r
.dw (FETCH_L	| OP_NOP	| STORE_D  )	 ; 55		LD D,r
.dw (FETCH_MHL	| OP_NOP	| STORE_D  )	 ; 56		LD D,r
.dw (FETCH_A    | OP_NOP	| STORE_D  )	 ; 57		LD D,r
.dw (FETCH_B	| OP_NOP	| STORE_E  )	 ; 58		LD E,r
.dw (FETCH_C	| OP_NOP	| STORE_E  )	 ; 59		LD E,r
.dw (FETCH_D	| OP_NOP	| STORE_E  )	 ; 5A		LD E,r
.dw (FETCH_E	| OP_NOP	| STORE_E  )	 ; 5B		LD E,r
.dw (FETCH_H	| OP_NOP	| STORE_E  )	 ; 5C		LD E,r
.dw (FETCH_L	| OP_NOP	| STORE_E  )	 ; 5D		LD E,r
.dw (FETCH_MHL	| OP_NOP	| STORE_E  )	 ; 5E		LD E,r
.dw (FETCH_A    | OP_NOP	| STORE_E  )	 ; 5F		LD E,r
.dw (FETCH_B	| OP_NOP	| STORE_H  )	 ; 60		LD H,r
.dw (FETCH_C	| OP_NOP	| STORE_H  )	 ; 61		LD H,r
.dw (FETCH_D	| OP_NOP	| STORE_H  )	 ; 62		LD H,r
.dw (FETCH_E	| OP_NOP	| STORE_H  )	 ; 63		LD H,r
.dw (FETCH_H	| OP_NOP	| STORE_H  )	 ; 64		LD H,r
.dw (FETCH_L	| OP_NOP	| STORE_H  )	 ; 65		LD H,r
.dw (FETCH_MHL	| OP_NOP	| STORE_H  )	 ; 66		LD H,r
.dw (FETCH_A    | OP_NOP	| STORE_H  )	 ; 67		LD H,r
.dw (FETCH_B	| OP_NOP	| STORE_L  )	 ; 68		LD L,r
.dw (FETCH_C	| OP_NOP	| STORE_L  )	 ; 69		LD L,r
.dw (FETCH_D	| OP_NOP	| STORE_L  )	 ; 6A		LD L,r
.dw (FETCH_E	| OP_NOP	| STORE_L  )	 ; 6B		LD L,r
.dw (FETCH_H	| OP_NOP	| STORE_L  )	 ; 6C		LD L,r
.dw (FETCH_L	| OP_NOP	| STORE_L  )	 ; 6D		LD L,r
.dw (FETCH_MHL	| OP_NOP	| STORE_L  )	 ; 6E		LD L,r
.dw (FETCH_A    | OP_NOP	| STORE_L  )	 ; 6F		LD L,r
.dw (FETCH_B	| OP_NOP	| STORE_MHL)	 ; 70		LD (HL),r
.dw (FETCH_C	| OP_NOP	| STORE_MHL)	 ; 71		LD (HL),r
.dw (FETCH_D	| OP_NOP	| STORE_MHL)	 ; 72		LD (HL),r
.dw (FETCH_E	| OP_NOP	| STORE_MHL)	 ; 73		LD (HL),r
.dw (FETCH_H	| OP_NOP	| STORE_MHL)	 ; 74		LD (HL),r
.dw (FETCH_L	| OP_NOP	| STORE_MHL)	 ; 75		LD (HL),r
.dw (FETCH_NOP	| OP_NOP	| STORE_NOP)	 ; 76		HALT
.dw (FETCH_A    | OP_NOP	| STORE_MHL)	 ; 77		LD (HL),r
.dw (FETCH_B	| OP_NOP	| STORE_A  )	 ; 78		LD A,r
.dw (FETCH_C	| OP_NOP	| STORE_A  )	 ; 79		LD A,r
.dw (FETCH_D	| OP_NOP	| STORE_A  )	 ; 7A		LD A,r
.dw (FETCH_E	| OP_NOP	| STORE_A  )	 ; 7B		LD A,r
.dw (FETCH_H	| OP_NOP	| STORE_A  )	 ; 7C		LD A,r
.dw (FETCH_L	| OP_NOP	| STORE_A  )	 ; 7D		LD A,r
.dw (FETCH_MHL	| OP_NOP	| STORE_A  )	 ; 7E		LD A,r
.dw (FETCH_A    | OP_NOP	| STORE_A  )	 ; 7F		LD A,r
.dw (FETCH_B	| OP_ADDA	| STORE_NOP)	 ; 80		ADD A,r
.dw (FETCH_C	| OP_ADDA	| STORE_NOP)	 ; 81		ADD A,r
.dw (FETCH_D	| OP_ADDA	| STORE_NOP)	 ; 82		ADD A,r
.dw (FETCH_E	| OP_ADDA	| STORE_NOP)	 ; 83		ADD A,r
.dw (FETCH_H	| OP_ADDA	| STORE_NOP)	 ; 84		ADD A,r
.dw (FETCH_L	| OP_ADDA	| STORE_NOP)	 ; 85		ADD A,r
.dw (FETCH_MHL	| OP_ADDA	| STORE_NOP)	 ; 86		ADD A,r
.dw (FETCH_A    | OP_ADDA	| STORE_NOP)	 ; 87		ADD A,r
.dw (FETCH_B	| OP_ADCA	| STORE_NOP)	 ; 88		ADC A,r
.dw (FETCH_C	| OP_ADCA	| STORE_NOP)	 ; 89		ADC A,r
.dw (FETCH_D	| OP_ADCA	| STORE_NOP)	 ; 8A		ADC A,r
.dw (FETCH_E	| OP_ADCA	| STORE_NOP)	 ; 8B		ADC A,r
.dw (FETCH_H	| OP_ADCA	| STORE_NOP)	 ; 8C		ADC A,r
.dw (FETCH_L	| OP_ADCA	| STORE_NOP)	 ; 8D		ADC A,r
.dw (FETCH_MHL	| OP_ADCA	| STORE_NOP)	 ; 8E		ADC A,r
.dw (FETCH_A    | OP_ADCA	| STORE_NOP)	 ; 8F		ADC A,r
.dw (FETCH_B	| OP_SUBFA	| STORE_NOP)	 ; 90		SUB A,r
.dw (FETCH_C	| OP_SUBFA	| STORE_NOP)	 ; 91		SUB A,r
.dw (FETCH_D	| OP_SUBFA	| STORE_NOP)	 ; 92		SUB A,r
.dw (FETCH_E	| OP_SUBFA	| STORE_NOP)	 ; 93		SUB A,r
.dw (FETCH_H	| OP_SUBFA	| STORE_NOP)	 ; 94		SUB A,r
.dw (FETCH_L	| OP_SUBFA	| STORE_NOP)	 ; 95		SUB A,r
.dw (FETCH_MHL	| OP_SUBFA	| STORE_NOP)	 ; 96		SUB A,r
.dw (FETCH_A    | OP_SUBFA	| STORE_NOP)	 ; 97		SUB A,r
.dw (FETCH_B	| OP_SBCFA	| STORE_NOP)	 ; 98		SBC A,r
.dw (FETCH_C	| OP_SBCFA	| STORE_NOP)	 ; 99		SBC A,r
.dw (FETCH_D	| OP_SBCFA	| STORE_NOP)	 ; 9A		SBC A,r
.dw (FETCH_E	| OP_SBCFA	| STORE_NOP)	 ; 9B		SBC A,r
.dw (FETCH_H	| OP_SBCFA	| STORE_NOP)	 ; 9C		SBC A,r
.dw (FETCH_L	| OP_SBCFA	| STORE_NOP)	 ; 9D		SBC A,r
.dw (FETCH_MHL	| OP_SBCFA	| STORE_NOP)	 ; 9E		SBC A,r
.dw (FETCH_A    | OP_SBCFA	| STORE_NOP)	 ; 9F		SBC A,r
.dw (FETCH_B	| OP_ANDA	| STORE_NOP)	 ; A0		AND A,r
.dw (FETCH_C	| OP_ANDA	| STORE_NOP)	 ; A1		AND A,r
.dw (FETCH_D	| OP_ANDA	| STORE_NOP)	 ; A2		AND A,r
.dw (FETCH_E	| OP_ANDA	| STORE_NOP)	 ; A3		AND A,r
.dw (FETCH_H	| OP_ANDA	| STORE_NOP)	 ; A4		AND A,r
.dw (FETCH_L	| OP_ANDA	| STORE_NOP)	 ; A5		AND A,r
.dw (FETCH_MHL	| OP_ANDA	| STORE_NOP)	 ; A6		AND A,r
.dw (FETCH_A    | OP_ANDA	| STORE_NOP)	 ; A7		AND A,r
.dw (FETCH_B	| OP_XORA	| STORE_NOP)	 ; A8		XOR A,r
.dw (FETCH_C	| OP_XORA	| STORE_NOP)	 ; A9		XOR A,r
.dw (FETCH_D	| OP_XORA	| STORE_NOP)	 ; AA		XOR A,r
.dw (FETCH_E	| OP_XORA	| STORE_NOP)	 ; AB		XOR A,r
.dw (FETCH_H	| OP_XORA	| STORE_NOP)	 ; AC		XOR A,r
.dw (FETCH_L	| OP_XORA	| STORE_NOP)	 ; AD		XOR A,r
.dw (FETCH_MHL	| OP_XORA	| STORE_NOP)	 ; AE		XOR A,r
.dw (FETCH_A    | OP_XORA	| STORE_NOP)	 ; AF		XOR A,r
.dw (FETCH_B	| OP_ORA	| STORE_NOP)	 ; B0		OR A,r
.dw (FETCH_C	| OP_ORA	| STORE_NOP)	 ; B1		OR A,r
.dw (FETCH_D	| OP_ORA	| STORE_NOP)	 ; B2		OR A,r
.dw (FETCH_E	| OP_ORA	| STORE_NOP)	 ; B3		OR A,r
.dw (FETCH_H	| OP_ORA	| STORE_NOP)	 ; B4		OR A,r
.dw (FETCH_L	| OP_ORA	| STORE_NOP)	 ; B5		OR A,r
.dw (FETCH_MHL	| OP_ORA	| STORE_NOP)	 ; B6		OR A,r
.dw (FETCH_A    | OP_ORA	| STORE_NOP)	 ; B7		OR A,r
.dw (FETCH_B	| OP_CPFA	| STORE_NOP)	 ; B8		CP A,r
.dw (FETCH_C	| OP_CPFA	| STORE_NOP)	 ; B9		CP A,r
.dw (FETCH_D	| OP_CPFA	| STORE_NOP)	 ; BA		CP A,r
.dw (FETCH_E	| OP_CPFA	| STORE_NOP)	 ; BB		CP A,r
.dw (FETCH_H	| OP_CPFA	| STORE_NOP)	 ; BC		CP A,r
.dw (FETCH_L	| OP_CPFA	| STORE_NOP)	 ; BD		CP A,r
.dw (FETCH_MHL	| OP_CPFA	| STORE_NOP)	 ; BE		CP A,r
.dw (FETCH_A    | OP_CPFA	| STORE_NOP)	 ; BF	 	CP A,r
.dw (FETCH_NOP  | OP_IFNZ	| STORE_RET)	 ; C0		RET NZ
.dw (FETCH_NOP  | OP_POP16	| STORE_BC )	 ; C1		POP BC
.dw (FETCH_DIR16| OP_IFNZ	| STORE_PC )	 ; C2 nn nn	JP NZ,nn
.dw (FETCH_DIR16| OP_NOP	| STORE_PC )	 ; C3 nn nn	JP nn
.dw (FETCH_DIR16| OP_IFNZ	| STORE_CALL)	 ; C4 nn nn	CALL NZ,nn
.dw (FETCH_BC	| OP_PUSH16	| STORE_NOP)	 ; C5		PUSH BC
.dw (FETCH_DIR8	| OP_ADDA	| STORE_NOP)	 ; C6 nn	ADD A,n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; C7		RST 0
.dw (FETCH_NOP	| OP_IFZ	| STORE_RET)	 ; C8		RET Z
.dw (FETCH_NOP	| OP_NOP	| STORE_RET)	 ; C9		RET
.dw (FETCH_DIR16| OP_IFZ	| STORE_PC )	 ; CA nn nn	JP Z,nn
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; CB 		(Z80 specific)
.dw (FETCH_DIR16| OP_IFZ	| STORE_CALL)	 ; CC nn nn	CALL Z,nn
.dw (FETCH_DIR16| OP_NOP	| STORE_CALL)	 ; CD nn nn	CALL nn
.dw (FETCH_DIR8	| OP_ADCA	| STORE_NOP)	 ; CE nn	ADC A,n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; CF		RST 8H
.dw (FETCH_NOP	| OP_IFNC	| STORE_RET)	 ; D0		RET NC
.dw (FETCH_NOP  | OP_POP16	| STORE_DE )	 ; D1		POP DE
.dw (FETCH_DIR16| OP_IFNC	| STORE_PC )	 ; D2 nn nn	JP NC,nn
.dw (FETCH_DIR8	| OP_OUTA	| STORE_NOP)	 ; D3 nn	OUT (n),A
.dw (FETCH_DIR16| OP_IFNC	| STORE_CALL)	 ; D4 nn nn	CALL NC,nn
.dw (FETCH_DE	| OP_PUSH16	| STORE_NOP)	 ; D5		PUSH DE
.dw (FETCH_DIR8	| OP_SUBFA	| STORE_NOP)	 ; D6 nn	SUB n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; D7		RST 10H
.dw (FETCH_NOP	| OP_IFC	| STORE_RET)	 ; D8		RET C
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; D9		EXX		(Z80)
.dw (FETCH_DIR16| OP_IFC	| STORE_PC )	 ; DA nn nn	JP C,nn
.dw (FETCH_DIR8	| OP_IN 	| STORE_A  )	 ; DB nn	IN A,(n)
.dw (FETCH_DIR16| OP_IFC	| STORE_CALL)	 ; DC nn nn	CALL C,nn
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; DD 		(Z80)
.dw (FETCH_DIR8	| OP_SBCFA	| STORE_NOP)	 ; DE nn	SBC A,n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; DF		RST 18H
.dw (FETCH_NOP	| OP_IFPO	| STORE_RET)	 ; E0		RET PO
.dw (FETCH_NOP	| OP_POP16	| STORE_HL )	 ; E1		POP HL
.dw (FETCH_DIR16| OP_IFPO	| STORE_PC )	 ; E2 nn nn	JP PO,nn
.dw (FETCH_MSP	| OP_EXHL	| STORE_MSP)	 ; E3		EX (SP),HL
.dw (FETCH_DIR16| OP_IFPO	| STORE_CALL)	 ; E4 nn nn	CALL PO,nn
.dw (FETCH_HL	| OP_PUSH16	| STORE_NOP)	 ; E5		PUSH HL
.dw (FETCH_DIR8	| OP_ANDA	| STORE_NOP)	 ; E6 nn	AND n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; E7		RST 20H
.dw (FETCH_NOP	| OP_IFPE	| STORE_RET)	 ; E8		RET PE
.dw (FETCH_HL	| OP_NOP	| STORE_PC )	 ; E9		JP (HL)
.dw (FETCH_DIR16| OP_IFPE	| STORE_PC )	 ; EA nn nn	JP PE,nn
.dw (FETCH_DE	| OP_EXHL	| STORE_DE )	 ; EB		EX DE,HL
.dw (FETCH_DIR16| OP_IFPE	| STORE_CALL)	 ; EC nn nn	CALL PE,nn
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; ED		(Z80 specific)
.dw (FETCH_DIR8	| OP_XORA	| STORE_NOP)	 ; EE nn	XOR n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; EF		RST 28H
.dw (FETCH_NOP	| OP_IFP	| STORE_RET)	 ; F0		RET P
.dw (FETCH_NOP	| OP_POP16	| STORE_AF )	 ; F1		POP AF
.dw (FETCH_DIR16| OP_IFP	| STORE_PC )	 ; F2 nn nn	JP P,nn
.dw (FETCH_NOP	| OP_DI		| STORE_NOP)	 ; F3		DI
.dw (FETCH_DIR16| OP_IFP	| STORE_CALL)	 ; F4 nn nn	CALL P,nn
.dw (FETCH_AF	| OP_PUSH16	| STORE_NOP)	 ; F5		PUSH AF
.dw (FETCH_DIR8	| OP_ORA	| STORE_NOP)	 ; F6 nn	OR n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; F7		RST 30H
.dw (FETCH_NOP	| OP_IFM	| STORE_RET)	 ; F8		RET M
.dw (FETCH_HL	| OP_NOP	| STORE_SP )	 ; F9		LD SP,HL
.dw (FETCH_DIR16| OP_IFM	| STORE_PC )	 ; FA nn nn	JP M,nn
.dw (FETCH_NOP	| OP_EI 	| STORE_NOP)	 ; FB		EI
.dw (FETCH_DIR16| OP_IFM	| STORE_CALL)	 ; FC nn nn	CALL M,nn
.dw (FETCH_NOP	| OP_INV	| STORE_NOP)	 ; FD 		(Z80 specific)
.dw (FETCH_DIR8	| OP_CPFA	| STORE_NOP)	 ; FE nn	CP n
.dw (FETCH_RST	| OP_NOP	| STORE_CALL)	 ; FF		RST 38H

; vim:set ts=8 noet nowrap

;    Configuration, hardware definition, ...
;    
;    Copyright (C) 2010 Sprite_tm
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
;    $Id: config.inc 178 2012-03-16 14:34:52Z leo $
;


#define VMAJOR 2		/* Version number */
#define VMINOR 3

#ifndef DRAM_8BIT
  #define DRAM_8BIT 1		/* 1 = 8bit wide DRAM */
#endif
#ifndef F_CPU
  #define F_CPU  20000000	/* system clock in Hz; defaults to 20MHz */
#endif
#ifndef BAUD
  #define BAUD   38400		/* console baud rate */
#endif
 
#define K 1024
#define M 1024*K

;#define RAMSIZE 256*K*4	/* 1 chip 256Kx4 */
#define  RAMSIZE   4*M*4 * 2	/* 2 chips 4Mx4  */

#define EM_Z80	1		/* Emulate Z80 if true */

#ifndef FAT16_SUPPORT
  #define FAT16_SUPPORT 1	/* Include Support for FAT16 Partitions */
#endif				/*  which may contain CP/M image files. */ 
#define RAMDISKCNT    0		/* Number of RAM disks */
#define RAMDISKNR     'I'-'A'	/* Drive "letter" for first RAM disk */

#define PARTID 0x52		/* Partition table id */
				/* http://www.win.tue.nl/~aeb/partitions/partition_types-1.html */
#define IPLADDR	0x2000		/* Bootloader load address */

#define DRAM_WAITSTATES 1	/* Number of additional clock cycles for dram read access */
#define REFR_RATE   64000       /* dram refresh rate in cycles/s. */
				/* Most drams need 1/15.6µs. */
#define	RXBUFSIZE 128		/* USART recieve buffer size. Must be power of 2 */
#define	TXBUFSIZE 128		/* USART transmit buffer size. Must be power of 2 */


#if EM_Z80
  #define CPUSTR "Z80"
#else
  #define CPUSTR "8080"
#endif

.equ BOOTWAIT      = 1
.equ MEMTEST       = 1
.equ MEMFILL       = 1
.equ MMC_DEBUG     = 0		/* Increase for more debugging */
.equ FAT16_DEBUG   = 0
.equ FAT16_RWDEBUG = 0
.equ FAT16_DBG_FAT = 0
.equ DISK_DEBUG    = 0		/* Increase for more debugging */
.equ HOSTRW_DEBUG  = 0
.equ HEAP_DEBUG	   = 0
.equ PORT_DEBUG    = 0
.equ INS_DEBUG     = 0
.equ STACK_DBG     = 0
.equ PRINT_PC      = 0

#define MMC_SPI2X  1		/* 0 = SPI CLK/4, 1 = SPI CLK/2 */

#define MEMFILL_VAL      0x76	/* Fill ram with HALT opcode. */
#define DBG_TRACE_BOTTOM 0x01	/* Page boundaries for INS_DEBUG and PRINT_PC  */
#define DBG_TRACE_TOP	 0xdc	/* Trace is off, below bottom page and above top page. */

;-----------------------------------------------------------------------
; Port declarations

#if DRAM_8BIT	/* Implies software uart */

;Port D
.equ RAM_D0	= 0
.equ RAM_D1	= 1
.equ RAM_D2	= 2
.equ RAM_D3	= 3
.equ RAM_D4	= 4
.equ RAM_D5	= 5
.equ RAM_D6	= 6
.equ RAM_D7	= 7
.equ RAM_A0	= 0
.equ RAM_A1	= 1
.equ RAM_A2	= 2
.equ RAM_A3	= 3
.equ RAM_A4	= 4
.equ RAM_A5	= 5
.equ RAM_A6	= 6
.equ RAM_A7	= 7

;Port B
.equ MMC_MOSI	= 3
.equ MMC_MISO	= 4
.equ MMC_SCK	= 5
.equ RAM_A8	= 3
.equ RAM_A9	= 4
.equ RAM_A10	= 5

.equ RXD	= 0
.equ TXD	= 1
.equ MMC_CS	= 2

.equ P_MMC_CS	= PORTB
.equ P_A8	= PORTB
.equ P_RXD	= PORTB

;Port C
.equ RAM_RAS	= 0
.equ RAM_CAS	= 1
.equ RAM_OE	= 2
.equ RAM_W	= 3

.equ P_RAS	= PORTC
.equ P_CAS	= PORTC
.equ P_OE	= PORTC
.equ P_W	= PORTC


#else	/* 4 bit RAM, hardware uart */

; Port D
.equ RXD	= 0
.equ TXD	= 1
.equ RAM_OE	= 2
.equ RAM_A8	= 3
.equ MMC_CS	= 4
.equ RAM_A5	= 5
.equ RAM_A6	= 6
.equ RAM_A7	= 7

.equ P_RXD	= PORTD
.equ P_OE	= PORTD
.equ P_A8	= PORTD
.equ P_MMC_CS	= PORTD
.equ P_AH	= PORTD
	             ; RAM_A[7..5]
.equ RAM_AH_MASK = (1<<RAM_A8)|(1<<RAM_A7)|(1<<RAM_A6)|(1<<RAM_A5)
.equ PD_OUTPUT_MASK = (1<<MMC_CS) | (1<<RAM_OE) | RAM_AH_MASK


;Port B
.equ RAM_A4	= 0
.equ RAM_A3	= 1
.equ RAM_A2	= 2
.equ RAM_A1	= 3
.equ MMC_MOSI	= 3
.equ RAM_A0	= 4
.equ MMC_MISO	= 4
.equ RAM_RAS	= 5
.equ MMC_SCK	= 5

.equ P_RAS	= PORTB
.equ P_AL	= PORTB
			; RAM_A[4..0]
.equ RAM_AL_MASK = (1<<RAM_A4)|(1<<RAM_A3)|(1<<RAM_A2)|(1<<RAM_A1)|(1<<RAM_A0)
.equ PB_OUTPUT_MASK = (1<<RAM_ras) | RAM_AL_MASK

;Port C
.equ RAM_D0	= 0
.equ RAM_D1	= 1
.equ RAM_D2	= 2
.equ RAM_D3	= 3
.equ RAM_W	= 4
.equ RAM_CAS	= 5

.equ P_DQ	= PORTC
.equ P_W	= PORTC
.equ P_CAS	= PORTC

.equ RAM_DQ_MASK = (1<<RAM_D3)|(1<<RAM_D2)|(1<<RAM_D1)|(1<<RAM_D0)
.equ PC_OUTPUT_MASK = (1<<RAM_CAS)|(1<<RAM_W)

#endif	/* DRAM_8BIT */


;-----------------------------------------------------------------------
;Register definitions

.def	_tmp0	= r0
.def	_tmp1	= r1

.def	_RAS0	= r2
.def	_CAS0	= r3
.def	_OE	= r4
.def	_WE	= r5
.def	_255	= r6
.def	_0	= r7
.def	z_a	= r9
;.def	stx_bitcount  = r9
;.def	stx_dr	      = r10

.def	srx_lastedgel = r10
.def	srx_lastedgeh = r11
;.def	insstore= r8	;
;.def	insop	= r13	;

.def	insdecl	= r12	;
.def	insdech	= r13	;
.def	z_spl	= r14
.def	z_sph	= r15	;
.def	temp	= r16 	;
.def	temp2	= r17 	;
.def	temp3	= r18
.def	temp4	= r19
.def	z_flags	= r20	;
.def	intstat	= r21	; interpreter status / interrupt status
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


.equ	i_break = 0			;break detected flag
.equ	i_trace = 1			;cpu interpreter trace flag
.equ	i_halt  = 2			;executing halt instruction

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

	.equ	prefixfd = 1		;Opcode prefix DD=0, FD=1
	.equ	trace	 = 0

; This is the base z80 port address for clock access
#define	TIMERPORT   0x40
#define TIMER_CTL   TIMERPORT
#define TIMER_MSECS TIMERPORT+1
#define TIMER_SECS  TIMER_MSECS+2

#define starttimercmd	1
#define quitTimerCmd	2
#define printTimerCmd	15
#define uptimeCmd	16

#define DEBUGPORT   0x4F

#define startTraceCmd 1
#define stopTraceCmd  0


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

; vim:set ts=8 noet nowrap



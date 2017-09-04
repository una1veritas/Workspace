;***** THIS IS A MACHINE GENERATED FILE - DO NOT EDIT ********************
;***** Created: 2009-11-03 14:40 ******* Source: ATmega328P.xml **********
;*************************************************************************
;* A P P L I C A T I O N   N O T E   F O R   T H E   A V R   F A M I L Y
;* 
;* Number            : AVR000
;* File Name         : "m328Pdef.inc"
;* Title             : Register/Bit Definitions for the ATmega328P
;* Date              : 2009-11-03
;* Version           : 2.35
;* Support E-mail    : avr@atmel.com
;* Target MCU        : ATmega328P
;* 
;* DESCRIPTION
;* When including this file in the assembly program file, all I/O register 
;* names and I/O register bit names appearing in the data book can be used.
;* In addition, the six registers forming the three data pointers X, Y and 
;* Z have been assigned names XL - ZH. Highest RAM address for Internal 
;* SRAM is also defined 
;* 
;* The Register names are represented by their hexadecimal address.
;* 
;* The Register Bit names are represented by their bit number (0-7).
;* 
;* Please observe the difference in using the bit names with instructions
;* such as "sbr"/"cbr" (set/clear bit in register) and "sbrs"/"sbrc"
;* (skip if bit in register set/cleared). The following example illustrates
;* this:
;* 
;* in    r16,PORTB             ;read PORTB latch
;* sbr   r16,(1<<PB6)+(1<<PB5) ;set PB6 and PB5 (use masks, not bit#)
;* out   PORTB,r16             ;output to PORTB
;* 
;* in    r16,TIFR              ;read the Timer Interrupt Flag Register
;* sbrc  r16,TOV0              ;test the overflow flag (use bit#)
;* rjmp  TOV0_is_set           ;jump if set
;* ...                         ;otherwise do something else
;*************************************************************************

#ifndef _M328PDEF_INC_
#define _M328PDEF_INC_


#pragma partinc 0

; ***** SPECIFY DEVICE ***************************************************
.device ATmega328P
#pragma AVRPART ADMIN PART_NAME ATmega328P
#define	SIGNATURE_000	  0x1e
#define	SIGNATURE_001	  0x95
#define	SIGNATURE_002	  0x0f

#pragma AVRPART CORE CORE_VERSION V2E


; ***** I/O REGISTER DEFINITIONS *****************************************
; NOTE:
; Definitions marked "MEMORY MAPPED"are extended I/O ports
; and cannot be used with IN/OUT instructions
#define	UDR0	  0xc6	; MEMORY MAPPED
#define	UBRR0L	  0xc4	; MEMORY MAPPED
#define	UBRR0H	  0xc5	; MEMORY MAPPED
#define	UCSR0C	  0xc2	; MEMORY MAPPED
#define	UCSR0B	  0xc1	; MEMORY MAPPED
#define	UCSR0A	  0xc0	; MEMORY MAPPED
#define	TWAMR	  0xbd	; MEMORY MAPPED
#define	TWCR	  0xbc	; MEMORY MAPPED
#define	TWDR	  0xbb	; MEMORY MAPPED
#define	TWAR	  0xba	; MEMORY MAPPED
#define	TWSR	  0xb9	; MEMORY MAPPED
#define	TWBR	  0xb8	; MEMORY MAPPED
#define	ASSR	  0xb6	; MEMORY MAPPED
#define	OCR2B	  0xb4	; MEMORY MAPPED
#define	OCR2A	  0xb3	; MEMORY MAPPED
#define	TCNT2	  0xb2	; MEMORY MAPPED
#define	TCCR2B	  0xb1	; MEMORY MAPPED
#define	TCCR2A	  0xb0	; MEMORY MAPPED
#define	OCR1BL	  0x8a	; MEMORY MAPPED
#define	OCR1BH	  0x8b	; MEMORY MAPPED
#define	OCR1AL	  0x88	; MEMORY MAPPED
#define	OCR1AH	  0x89	; MEMORY MAPPED
#define	ICR1L	  0x86	; MEMORY MAPPED
#define	ICR1H	  0x87	; MEMORY MAPPED
#define	TCNT1L	  0x84	; MEMORY MAPPED
#define	TCNT1H	  0x85	; MEMORY MAPPED
#define	TCCR1C	  0x82	; MEMORY MAPPED
#define	TCCR1B	  0x81	; MEMORY MAPPED
#define	TCCR1A	  0x80	; MEMORY MAPPED
#define	DIDR1	  0x7f	; MEMORY MAPPED
#define	DIDR0	  0x7e	; MEMORY MAPPED
#define	ADMUX	  0x7c	; MEMORY MAPPED
#define	ADCSRB	  0x7b	; MEMORY MAPPED
#define	ADCSRA	  0x7a	; MEMORY MAPPED
#define	ADCH	  0x79	; MEMORY MAPPED
#define	ADCL	  0x78	; MEMORY MAPPED
#define	TIMSK2	  0x70	; MEMORY MAPPED
#define	TIMSK1	  0x6f	; MEMORY MAPPED
#define	TIMSK0	  0x6e	; MEMORY MAPPED
#define	PCMSK1	  0x6c	; MEMORY MAPPED
#define	PCMSK2	  0x6d	; MEMORY MAPPED
#define	PCMSK0	  0x6b	; MEMORY MAPPED
#define	EICRA	  0x69	; MEMORY MAPPED
#define	PCICR	  0x68	; MEMORY MAPPED
#define	OSCCAL	  0x66	; MEMORY MAPPED
#define	PRR	  0x64	; MEMORY MAPPED
#define	CLKPR	  0x61	; MEMORY MAPPED
#define	WDTCSR	  0x60	; MEMORY MAPPED
#define	SREG	  0x3f
#define	SPL	  0x3d
#define	SPH	  0x3e
#define	SPMCSR	  0x37
#define	MCUCR	  0x35
#define	MCUSR	  0x34
#define	SMCR	  0x33
#define	ACSR	  0x30
#define	SPDR	  0x2e
#define	SPSR	  0x2d
#define	SPCR	  0x2c
#define	GPIOR2	  0x2b
#define	GPIOR1	  0x2a
#define	OCR0B	  0x28
#define	OCR0A	  0x27
#define	TCNT0	  0x26
#define	TCCR0B	  0x25
#define	TCCR0A	  0x24
#define	GTCCR	  0x23
#define	EEARH	  0x22
#define	EEARL	  0x21
#define	EEDR	  0x20
#define	EECR	  0x1f
#define	GPIOR0	  0x1e
#define	EIMSK	  0x1d
#define	EIFR	  0x1c
#define	PCIFR	  0x1b
#define	TIFR2	  0x17
#define	TIFR1	  0x16
#define	TIFR0	  0x15
#define	PORTD	  0x0b
#define	DDRD	  0x0a
#define	PIND	  0x09
#define	PORTC	  0x08
#define	DDRC	  0x07
#define	PINC	  0x06
#define	PORTB	  0x05
#define	DDRB	  0x04
#define	PINB	  0x03


; ***** BIT DEFINITIONS **************************************************

; ***** USART0 ***********************
; UDR0 - USART I/O Data Register
#define	UDR0_0	  0	; USART I/O Data Register bit 0
#define	UDR0_1	  1	; USART I/O Data Register bit 1
#define	UDR0_2	  2	; USART I/O Data Register bit 2
#define	UDR0_3	  3	; USART I/O Data Register bit 3
#define	UDR0_4	  4	; USART I/O Data Register bit 4
#define	UDR0_5	  5	; USART I/O Data Register bit 5
#define	UDR0_6	  6	; USART I/O Data Register bit 6
#define	UDR0_7	  7	; USART I/O Data Register bit 7

; UCSR0A - USART Control and Status Register A
#define	MPCM0	  0	; Multi-processor Communication Mode
#define	U2X0	  1	; Double the USART transmission speed
#define	UPE0	  2	; Parity Error
#define	DOR0	  3	; Data overRun
#define	FE0	  4	; Framing Error
#define	UDRE0	  5	; USART Data Register Empty
#define	TXC0	  6	; USART Transmitt Complete
#define	RXC0	  7	; USART Receive Complete

; UCSR0B - USART Control and Status Register B
#define	TXB80	  0	; Transmit Data Bit 8
#define	RXB80	  1	; Receive Data Bit 8
#define	UCSZ02	  2	; Character Size
#define	TXEN0	  3	; Transmitter Enable
#define	RXEN0	  4	; Receiver Enable
#define	UDRIE0	  5	; USART Data register Empty Interrupt Enable
#define	TXCIE0	  6	; TX Complete Interrupt Enable
#define	RXCIE0	  7	; RX Complete Interrupt Enable

; UCSR0C - USART Control and Status Register C
#define	UCPOL0	  0	; Clock Polarity
#define	UCSZ00	  1	; Character Size
#define	UCPHA0	  UCSZ00	; For compatibility
#define	UCSZ01	  2	; Character Size
#define	UDORD0	  UCSZ01	; For compatibility
#define	USBS0	  3	; Stop Bit Select
#define	UPM00	  4	; Parity Mode Bit 0
#define	UPM01	  5	; Parity Mode Bit 1
#define	UMSEL00	  6	; USART Mode Select
#define	UMSEL0	  UMSEL00	; For compatibility
#define	UMSEL01	  7	; USART Mode Select
#define	UMSEL1	  UMSEL01	; For compatibility

; UBRR0H - USART Baud Rate Register High Byte
#define	UBRR8	  0	; USART Baud Rate Register bit 8
#define	UBRR9	  1	; USART Baud Rate Register bit 9
#define	UBRR10	  2	; USART Baud Rate Register bit 10
#define	UBRR11	  3	; USART Baud Rate Register bit 11

; UBRR0L - USART Baud Rate Register Low Byte
#define	_UBRR0	  0	; USART Baud Rate Register bit 0
#define	_UBRR1	  1	; USART Baud Rate Register bit 1
#define	UBRR2	  2	; USART Baud Rate Register bit 2
#define	UBRR3	  3	; USART Baud Rate Register bit 3
#define	UBRR4	  4	; USART Baud Rate Register bit 4
#define	UBRR5	  5	; USART Baud Rate Register bit 5
#define	UBRR6	  6	; USART Baud Rate Register bit 6
#define	UBRR7	  7	; USART Baud Rate Register bit 7


; ***** TWI **************************
; TWAMR - TWI (Slave) Address Mask Register
#define	TWAM0	  1	; 
#define	TWAMR0	  TWAM0	; For compatibility
#define	TWAM1	  2	; 
#define	TWAMR1	  TWAM1	; For compatibility
#define	TWAM2	  3	; 
#define	TWAMR2	  TWAM2	; For compatibility
#define	TWAM3	  4	; 
#define	TWAMR3	  TWAM3	; For compatibility
#define	TWAM4	  5	; 
#define	TWAMR4	  TWAM4	; For compatibility
#define	TWAM5	  6	; 
#define	TWAMR5	  TWAM5	; For compatibility
#define	TWAM6	  7	; 
#define	TWAMR6	  TWAM6	; For compatibility

; TWBR - TWI Bit Rate register
#define	TWBR0	  0	; 
#define	TWBR1	  1	; 
#define	TWBR2	  2	; 
#define	TWBR3	  3	; 
#define	TWBR4	  4	; 
#define	TWBR5	  5	; 
#define	TWBR6	  6	; 
#define	TWBR7	  7	; 

; TWCR - TWI Control Register
#define	TWIE	  0	; TWI Interrupt Enable
#define	TWEN	  2	; TWI Enable Bit
#define	TWWC	  3	; TWI Write Collition Flag
#define	TWSTO	  4	; TWI Stop Condition Bit
#define	TWSTA	  5	; TWI Start Condition Bit
#define	TWEA	  6	; TWI Enable Acknowledge Bit
#define	TWINT	  7	; TWI Interrupt Flag

; TWSR - TWI Status Register
#define	TWPS0	  0	; TWI Prescaler
#define	TWPS1	  1	; TWI Prescaler
#define	TWS3	  3	; TWI Status
#define	TWS4	  4	; TWI Status
#define	TWS5	  5	; TWI Status
#define	TWS6	  6	; TWI Status
#define	TWS7	  7	; TWI Status

; TWDR - TWI Data register
#define	TWD0	  0	; TWI Data Register Bit 0
#define	TWD1	  1	; TWI Data Register Bit 1
#define	TWD2	  2	; TWI Data Register Bit 2
#define	TWD3	  3	; TWI Data Register Bit 3
#define	TWD4	  4	; TWI Data Register Bit 4
#define	TWD5	  5	; TWI Data Register Bit 5
#define	TWD6	  6	; TWI Data Register Bit 6
#define	TWD7	  7	; TWI Data Register Bit 7

; TWAR - TWI (Slave) Address register
#define	TWGCE	  0	; TWI General Call Recognition Enable Bit
#define	TWA0	  1	; TWI (Slave) Address register Bit 0
#define	TWA1	  2	; TWI (Slave) Address register Bit 1
#define	TWA2	  3	; TWI (Slave) Address register Bit 2
#define	TWA3	  4	; TWI (Slave) Address register Bit 3
#define	TWA4	  5	; TWI (Slave) Address register Bit 4
#define	TWA5	  6	; TWI (Slave) Address register Bit 5
#define	TWA6	  7	; TWI (Slave) Address register Bit 6


; ***** TIMER_COUNTER_1 **************
; TIMSK1 - Timer/Counter Interrupt Mask Register
#define	TOIE1	  0	; Timer/Counter1 Overflow Interrupt Enable
#define	OCIE1A	  1	; Timer/Counter1 Output CompareA Match Interrupt Enable
#define	OCIE1B	  2	; Timer/Counter1 Output CompareB Match Interrupt Enable
#define	ICIE1	  5	; Timer/Counter1 Input Capture Interrupt Enable

; TIFR1 - Timer/Counter Interrupt Flag register
#define	TOV1	  0	; Timer/Counter1 Overflow Flag
#define	OCF1A	  1	; Output Compare Flag 1A
#define	OCF1B	  2	; Output Compare Flag 1B
#define	ICF1	  5	; Input Capture Flag 1

; TCCR1A - Timer/Counter1 Control Register A
#define	WGM10	  0	; Waveform Generation Mode
#define	WGM11	  1	; Waveform Generation Mode
#define	COM1B0	  4	; Compare Output Mode 1B, bit 0
#define	COM1B1	  5	; Compare Output Mode 1B, bit 1
#define	COM1A0	  6	; Comparet Ouput Mode 1A, bit 0
#define	COM1A1	  7	; Compare Output Mode 1A, bit 1

; TCCR1B - Timer/Counter1 Control Register B
#define	CS10	  0	; Prescaler source of Timer/Counter 1
#define	CS11	  1	; Prescaler source of Timer/Counter 1
#define	CS12	  2	; Prescaler source of Timer/Counter 1
#define	WGM12	  3	; Waveform Generation Mode
#define	WGM13	  4	; Waveform Generation Mode
#define	ICES1	  6	; Input Capture 1 Edge Select
#define	ICNC1	  7	; Input Capture 1 Noise Canceler

; TCCR1C - Timer/Counter1 Control Register C
#define	FOC1B	  6	; 
#define	FOC1A	  7	; 

; GTCCR - General Timer/Counter Control Register
#define	PSRSYNC	  0	; Prescaler Reset Timer/Counter1 and Timer/Counter0
#define	TSM	  7	; Timer/Counter Synchronization Mode


; ***** TIMER_COUNTER_2 **************
; TIMSK2 - Timer/Counter Interrupt Mask register
#define	TOIE2	  0	; Timer/Counter2 Overflow Interrupt Enable
#define	TOIE2A	  TOIE2	; For compatibility
#define	OCIE2A	  1	; Timer/Counter2 Output Compare Match A Interrupt Enable
#define	OCIE2B	  2	; Timer/Counter2 Output Compare Match B Interrupt Enable

; TIFR2 - Timer/Counter Interrupt Flag Register
#define	TOV2	  0	; Timer/Counter2 Overflow Flag
#define	OCF2A	  1	; Output Compare Flag 2A
#define	OCF2B	  2	; Output Compare Flag 2B

; TCCR2A - Timer/Counter2 Control Register A
#define	WGM20	  0	; Waveform Genration Mode
#define	WGM21	  1	; Waveform Genration Mode
#define	COM2B0	  4	; Compare Output Mode bit 0
#define	COM2B1	  5	; Compare Output Mode bit 1
#define	COM2A0	  6	; Compare Output Mode bit 1
#define	COM2A1	  7	; Compare Output Mode bit 1

; TCCR2B - Timer/Counter2 Control Register B
#define	CS20	  0	; Clock Select bit 0
#define	CS21	  1	; Clock Select bit 1
#define	CS22	  2	; Clock Select bit 2
#define	WGM22	  3	; Waveform Generation Mode
#define	FOC2B	  6	; Force Output Compare B
#define	FOC2A	  7	; Force Output Compare A

; TCNT2 - Timer/Counter2
#define	TCNT2_0	  0	; Timer/Counter 2 bit 0
#define	TCNT2_1	  1	; Timer/Counter 2 bit 1
#define	TCNT2_2	  2	; Timer/Counter 2 bit 2
#define	TCNT2_3	  3	; Timer/Counter 2 bit 3
#define	TCNT2_4	  4	; Timer/Counter 2 bit 4
#define	TCNT2_5	  5	; Timer/Counter 2 bit 5
#define	TCNT2_6	  6	; Timer/Counter 2 bit 6
#define	TCNT2_7	  7	; Timer/Counter 2 bit 7

; OCR2A - Timer/Counter2 Output Compare Register A
#define	OCR2A_0	  0	; Timer/Counter2 Output Compare Register Bit 0
#define	OCR2A_1	  1	; Timer/Counter2 Output Compare Register Bit 1
#define	OCR2A_2	  2	; Timer/Counter2 Output Compare Register Bit 2
#define	OCR2A_3	  3	; Timer/Counter2 Output Compare Register Bit 3
#define	OCR2A_4	  4	; Timer/Counter2 Output Compare Register Bit 4
#define	OCR2A_5	  5	; Timer/Counter2 Output Compare Register Bit 5
#define	OCR2A_6	  6	; Timer/Counter2 Output Compare Register Bit 6
#define	OCR2A_7	  7	; Timer/Counter2 Output Compare Register Bit 7

; OCR2B - Timer/Counter2 Output Compare Register B
#define	OCR2B_0	  0	; Timer/Counter2 Output Compare Register Bit 0
#define	OCR2B_1	  1	; Timer/Counter2 Output Compare Register Bit 1
#define	OCR2B_2	  2	; Timer/Counter2 Output Compare Register Bit 2
#define	OCR2B_3	  3	; Timer/Counter2 Output Compare Register Bit 3
#define	OCR2B_4	  4	; Timer/Counter2 Output Compare Register Bit 4
#define	OCR2B_5	  5	; Timer/Counter2 Output Compare Register Bit 5
#define	OCR2B_6	  6	; Timer/Counter2 Output Compare Register Bit 6
#define	OCR2B_7	  7	; Timer/Counter2 Output Compare Register Bit 7

; ASSR - Asynchronous Status Register
#define	TCR2BUB	  0	; Timer/Counter Control Register2 Update Busy
#define	TCR2AUB	  1	; Timer/Counter Control Register2 Update Busy
#define	OCR2BUB	  2	; Output Compare Register 2 Update Busy
#define	OCR2AUB	  3	; Output Compare Register2 Update Busy
#define	TCN2UB	  4	; Timer/Counter2 Update Busy
#define	AS2	  5	; Asynchronous Timer/Counter2
#define	EXCLK	  6	; Enable External Clock Input

; GTCCR - General Timer Counter Control register
#define	PSRASY	  1	; Prescaler Reset Timer/Counter2
#define	PSR2	  PSRASY	; For compatibility
;.equ	TSM	  7	; Timer/Counter Synchronization Mode


; ***** AD_CONVERTER *****************
; ADMUX - The ADC multiplexer Selection Register
#define	MUX0	  0	; Analog Channel and Gain Selection Bits
#define	MUX1	  1	; Analog Channel and Gain Selection Bits
#define	MUX2	  2	; Analog Channel and Gain Selection Bits
#define	MUX3	  3	; Analog Channel and Gain Selection Bits
#define	ADLAR	  5	; Left Adjust Result
#define	REFS0	  6	; Reference Selection Bit 0
#define	REFS1	  7	; Reference Selection Bit 1

; ADCSRA - The ADC Control and Status register A
#define	ADPS0	  0	; ADC  Prescaler Select Bits
#define	ADPS1	  1	; ADC  Prescaler Select Bits
#define	ADPS2	  2	; ADC  Prescaler Select Bits
#define	ADIE	  3	; ADC Interrupt Enable
#define	ADIF	  4	; ADC Interrupt Flag
#define	ADATE	  5	; ADC  Auto Trigger Enable
#define	ADSC	  6	; ADC Start Conversion
#define	ADEN	  7	; ADC Enable

; ADCSRB - The ADC Control and Status register B
#define	ADTS0	  0	; ADC Auto Trigger Source bit 0
#define	ADTS1	  1	; ADC Auto Trigger Source bit 1
#define	ADTS2	  2	; ADC Auto Trigger Source bit 2
#define	ACME	  6	; 

; ADCH - ADC Data Register High Byte
#define	ADCH0	  0	; ADC Data Register High Byte Bit 0
#define	ADCH1	  1	; ADC Data Register High Byte Bit 1
#define	ADCH2	  2	; ADC Data Register High Byte Bit 2
#define	ADCH3	  3	; ADC Data Register High Byte Bit 3
#define	ADCH4	  4	; ADC Data Register High Byte Bit 4
#define	ADCH5	  5	; ADC Data Register High Byte Bit 5
#define	ADCH6	  6	; ADC Data Register High Byte Bit 6
#define	ADCH7	  7	; ADC Data Register High Byte Bit 7

; ADCL - ADC Data Register Low Byte
#define	ADCL0	  0	; ADC Data Register Low Byte Bit 0
#define	ADCL1	  1	; ADC Data Register Low Byte Bit 1
#define	ADCL2	  2	; ADC Data Register Low Byte Bit 2
#define	ADCL3	  3	; ADC Data Register Low Byte Bit 3
#define	ADCL4	  4	; ADC Data Register Low Byte Bit 4
#define	ADCL5	  5	; ADC Data Register Low Byte Bit 5
#define	ADCL6	  6	; ADC Data Register Low Byte Bit 6
#define	ADCL7	  7	; ADC Data Register Low Byte Bit 7

; DIDR0 - Digital Input Disable Register
#define	ADC0D	  0	; 
#define	ADC1D	  1	; 
#define	ADC2D	  2	; 
#define	ADC3D	  3	; 
#define	ADC4D	  4	; 
#define	ADC5D	  5	; 


; ***** ANALOG_COMPARATOR ************
; ACSR - Analog Comparator Control And Status Register
#define	ACIS0	  0	; Analog Comparator Interrupt Mode Select bit 0
#define	ACIS1	  1	; Analog Comparator Interrupt Mode Select bit 1
#define	ACIC	  2	; Analog Comparator Input Capture Enable
#define	ACIE	  3	; Analog Comparator Interrupt Enable
#define	ACI	  4	; Analog Comparator Interrupt Flag
#define	ACO	  5	; Analog Compare Output
#define	ACBG	  6	; Analog Comparator Bandgap Select
#define	ACD	  7	; Analog Comparator Disable

; DIDR1 - Digital Input Disable Register 1
#define	AIN0D	  0	; AIN0 Digital Input Disable
#define	AIN1D	  1	; AIN1 Digital Input Disable


; ***** PORTB ************************
; PORTB - Port B Data Register
#define	PORTB0	  0	; Port B Data Register bit 0
#define	PB0	  0	; For compatibility
#define	PORTB1	  1	; Port B Data Register bit 1
#define	PB1	  1	; For compatibility
#define	PORTB2	  2	; Port B Data Register bit 2
#define	PB2	  2	; For compatibility
#define	PORTB3	  3	; Port B Data Register bit 3
#define	PB3	  3	; For compatibility
#define	PORTB4	  4	; Port B Data Register bit 4
#define	PB4	  4	; For compatibility
#define	PORTB5	  5	; Port B Data Register bit 5
#define	PB5	  5	; For compatibility
#define	PORTB6	  6	; Port B Data Register bit 6
#define	PB6	  6	; For compatibility
#define	PORTB7	  7	; Port B Data Register bit 7
#define	PB7	  7	; For compatibility

; DDRB - Port B Data Direction Register
#define	DDB0	  0	; Port B Data Direction Register bit 0
#define	DDB1	  1	; Port B Data Direction Register bit 1
#define	DDB2	  2	; Port B Data Direction Register bit 2
#define	DDB3	  3	; Port B Data Direction Register bit 3
#define	DDB4	  4	; Port B Data Direction Register bit 4
#define	DDB5	  5	; Port B Data Direction Register bit 5
#define	DDB6	  6	; Port B Data Direction Register bit 6
#define	DDB7	  7	; Port B Data Direction Register bit 7

; PINB - Port B Input Pins
#define	PINB0	  0	; Port B Input Pins bit 0
#define	PINB1	  1	; Port B Input Pins bit 1
#define	PINB2	  2	; Port B Input Pins bit 2
#define	PINB3	  3	; Port B Input Pins bit 3
#define	PINB4	  4	; Port B Input Pins bit 4
#define	PINB5	  5	; Port B Input Pins bit 5
#define	PINB6	  6	; Port B Input Pins bit 6
#define	PINB7	  7	; Port B Input Pins bit 7


; ***** PORTC ************************
; PORTC - Port C Data Register
#define	PORTC0	  0	; Port C Data Register bit 0
#define	PC0	  0	; For compatibility
#define	PORTC1	  1	; Port C Data Register bit 1
#define	PC1	  1	; For compatibility
#define	PORTC2	  2	; Port C Data Register bit 2
#define	PC2	  2	; For compatibility
#define	PORTC3	  3	; Port C Data Register bit 3
#define	PC3	  3	; For compatibility
#define	PORTC4	  4	; Port C Data Register bit 4
#define	PC4	  4	; For compatibility
#define	PORTC5	  5	; Port C Data Register bit 5
#define	PC5	  5	; For compatibility
#define	PORTC6	  6	; Port C Data Register bit 6
#define	PC6	  6	; For compatibility

; DDRC - Port C Data Direction Register
#define	DDC0	  0	; Port C Data Direction Register bit 0
#define	DDC1	  1	; Port C Data Direction Register bit 1
#define	DDC2	  2	; Port C Data Direction Register bit 2
#define	DDC3	  3	; Port C Data Direction Register bit 3
#define	DDC4	  4	; Port C Data Direction Register bit 4
#define	DDC5	  5	; Port C Data Direction Register bit 5
#define	DDC6	  6	; Port C Data Direction Register bit 6

; PINC - Port C Input Pins
#define	PINC0	  0	; Port C Input Pins bit 0
#define	PINC1	  1	; Port C Input Pins bit 1
#define	PINC2	  2	; Port C Input Pins bit 2
#define	PINC3	  3	; Port C Input Pins bit 3
#define	PINC4	  4	; Port C Input Pins bit 4
#define	PINC5	  5	; Port C Input Pins bit 5
#define	PINC6	  6	; Port C Input Pins bit 6


; ***** PORTD ************************
; PORTD - Port D Data Register
#define	PORTD0	  0	; Port D Data Register bit 0
#define	PD0	  0	; For compatibility
#define	PORTD1	  1	; Port D Data Register bit 1
#define	PD1	  1	; For compatibility
#define	PORTD2	  2	; Port D Data Register bit 2
#define	PD2	  2	; For compatibility
#define	PORTD3	  3	; Port D Data Register bit 3
#define	PD3	  3	; For compatibility
#define	PORTD4	  4	; Port D Data Register bit 4
#define	PD4	  4	; For compatibility
#define	PORTD5	  5	; Port D Data Register bit 5
#define	PD5	  5	; For compatibility
#define	PORTD6	  6	; Port D Data Register bit 6
#define	PD6	  6	; For compatibility
#define	PORTD7	  7	; Port D Data Register bit 7
#define	PD7	  7	; For compatibility

; DDRD - Port D Data Direction Register
#define	DDD0	  0	; Port D Data Direction Register bit 0
#define	DDD1	  1	; Port D Data Direction Register bit 1
#define	DDD2	  2	; Port D Data Direction Register bit 2
#define	DDD3	  3	; Port D Data Direction Register bit 3
#define	DDD4	  4	; Port D Data Direction Register bit 4
#define	DDD5	  5	; Port D Data Direction Register bit 5
#define	DDD6	  6	; Port D Data Direction Register bit 6
#define	DDD7	  7	; Port D Data Direction Register bit 7

; PIND - Port D Input Pins
#define	PIND0	  0	; Port D Input Pins bit 0
#define	PIND1	  1	; Port D Input Pins bit 1
#define	PIND2	  2	; Port D Input Pins bit 2
#define	PIND3	  3	; Port D Input Pins bit 3
#define	PIND4	  4	; Port D Input Pins bit 4
#define	PIND5	  5	; Port D Input Pins bit 5
#define	PIND6	  6	; Port D Input Pins bit 6
#define	PIND7	  7	; Port D Input Pins bit 7


; ***** TIMER_COUNTER_0 **************
; TIMSK0 - Timer/Counter0 Interrupt Mask Register
#define	TOIE0	  0	; Timer/Counter0 Overflow Interrupt Enable
#define	OCIE0A	  1	; Timer/Counter0 Output Compare Match A Interrupt Enable
#define	OCIE0B	  2	; Timer/Counter0 Output Compare Match B Interrupt Enable

; TIFR0 - Timer/Counter0 Interrupt Flag register
#define	TOV0	  0	; Timer/Counter0 Overflow Flag
#define	OCF0A	  1	; Timer/Counter0 Output Compare Flag 0A
#define	OCF0B	  2	; Timer/Counter0 Output Compare Flag 0B

; TCCR0A - Timer/Counter  Control Register A
#define	WGM00	  0	; Waveform Generation Mode
#define	WGM01	  1	; Waveform Generation Mode
#define	COM0B0	  4	; Compare Output Mode, Fast PWm
#define	COM0B1	  5	; Compare Output Mode, Fast PWm
#define	COM0A0	  6	; Compare Output Mode, Phase Correct PWM Mode
#define	COM0A1	  7	; Compare Output Mode, Phase Correct PWM Mode

; TCCR0B - Timer/Counter Control Register B
#define	CS00	  0	; Clock Select
#define	CS01	  1	; Clock Select
#define	CS02	  2	; Clock Select
#define	WGM02	  3	; 
#define	FOC0B	  6	; Force Output Compare B
#define	FOC0A	  7	; Force Output Compare A

; TCNT0 - Timer/Counter0
#define	TCNT0_0	  0	; 
#define	TCNT0_1	  1	; 
#define	TCNT0_2	  2	; 
#define	TCNT0_3	  3	; 
#define	TCNT0_4	  4	; 
#define	TCNT0_5	  5	; 
#define	TCNT0_6	  6	; 
#define	TCNT0_7	  7	; 

; OCR0A - Timer/Counter0 Output Compare Register
#define	OCR0A_0	  0	; 
#define	OCR0A_1	  1	; 
#define	OCR0A_2	  2	; 
#define	OCR0A_3	  3	; 
#define	OCR0A_4	  4	; 
#define	OCR0A_5	  5	; 
#define	OCR0A_6	  6	; 
#define	OCR0A_7	  7	; 

; OCR0B - Timer/Counter0 Output Compare Register
#define	OCR0B_0	  0	; 
#define	OCR0B_1	  1	; 
#define	OCR0B_2	  2	; 
#define	OCR0B_3	  3	; 
#define	OCR0B_4	  4	; 
#define	OCR0B_5	  5	; 
#define	OCR0B_6	  6	; 
#define	OCR0B_7	  7	; 

; GTCCR - General Timer/Counter Control Register
;.equ	PSRSYNC	  0	; Prescaler Reset Timer/Counter1 and Timer/Counter0
#define	PSR10	  PSRSYNC	; For compatibility
;.equ	TSM	  7	; Timer/Counter Synchronization Mode


; ***** EXTERNAL_INTERRUPT ***********
; EICRA - External Interrupt Control Register
#define	ISC00	  0	; External Interrupt Sense Control 0 Bit 0
#define	ISC01	  1	; External Interrupt Sense Control 0 Bit 1
#define	ISC10	  2	; External Interrupt Sense Control 1 Bit 0
#define	ISC11	  3	; External Interrupt Sense Control 1 Bit 1

; EIMSK - External Interrupt Mask Register
#define	INT0	  0	; External Interrupt Request 0 Enable
#define	INT1	  1	; External Interrupt Request 1 Enable

; EIFR - External Interrupt Flag Register
#define	INTF0	  0	; External Interrupt Flag 0
#define	INTF1	  1	; External Interrupt Flag 1

; PCICR - Pin Change Interrupt Control Register
#define	PCIE0	  0	; Pin Change Interrupt Enable 0
#define	PCIE1	  1	; Pin Change Interrupt Enable 1
#define	PCIE2	  2	; Pin Change Interrupt Enable 2

; PCMSK2 - Pin Change Mask Register 2
#define	PCINT16	  0	; Pin Change Enable Mask 16
#define	PCINT17	  1	; Pin Change Enable Mask 17
#define	PCINT18	  2	; Pin Change Enable Mask 18
#define	PCINT19	  3	; Pin Change Enable Mask 19
#define	PCINT20	  4	; Pin Change Enable Mask 20
#define	PCINT21	  5	; Pin Change Enable Mask 21
#define	PCINT22	  6	; Pin Change Enable Mask 22
#define	PCINT23	  7	; Pin Change Enable Mask 23

; PCMSK1 - Pin Change Mask Register 1
#define	PCINT8	  0	; Pin Change Enable Mask 8
#define	PCINT9	  1	; Pin Change Enable Mask 9
#define	PCINT10	  2	; Pin Change Enable Mask 10
#define	PCINT11	  3	; Pin Change Enable Mask 11
#define	PCINT12	  4	; Pin Change Enable Mask 12
#define	PCINT13	  5	; Pin Change Enable Mask 13
#define	PCINT14	  6	; Pin Change Enable Mask 14

; PCMSK0 - Pin Change Mask Register 0
#define	PCINT0	  0	; Pin Change Enable Mask 0
#define	PCINT1	  1	; Pin Change Enable Mask 1
#define	PCINT2	  2	; Pin Change Enable Mask 2
#define	PCINT3	  3	; Pin Change Enable Mask 3
#define	PCINT4	  4	; Pin Change Enable Mask 4
#define	PCINT5	  5	; Pin Change Enable Mask 5
#define	PCINT6	  6	; Pin Change Enable Mask 6
#define	PCINT7	  7	; Pin Change Enable Mask 7

; PCIFR - Pin Change Interrupt Flag Register
#define	PCIF0	  0	; Pin Change Interrupt Flag 0
#define	PCIF1	  1	; Pin Change Interrupt Flag 1
#define	PCIF2	  2	; Pin Change Interrupt Flag 2


; ***** SPI **************************
; SPDR - SPI Data Register
#define	SPDR0	  0	; SPI Data Register bit 0
#define	SPDR1	  1	; SPI Data Register bit 1
#define	SPDR2	  2	; SPI Data Register bit 2
#define	SPDR3	  3	; SPI Data Register bit 3
#define	SPDR4	  4	; SPI Data Register bit 4
#define	SPDR5	  5	; SPI Data Register bit 5
#define	SPDR6	  6	; SPI Data Register bit 6
#define	SPDR7	  7	; SPI Data Register bit 7

; SPSR - SPI Status Register
#define	SPI2X	  0	; Double SPI Speed Bit
#define	WCOL	  6	; Write Collision Flag
#define	SPIF	  7	; SPI Interrupt Flag

; SPCR - SPI Control Register
#define	SPR0	  0	; SPI Clock Rate Select 0
#define	SPR1	  1	; SPI Clock Rate Select 1
#define	CPHA	  2	; Clock Phase
#define	CPOL	  3	; Clock polarity
#define	MSTR	  4	; Master/Slave Select
#define	DORD	  5	; Data Order
#define	SPE	  6	; SPI Enable
#define	SPIE	  7	; SPI Interrupt Enable


; ***** WATCHDOG *********************
; WDTCSR - Watchdog Timer Control Register
#define	WDP0	  0	; Watch Dog Timer Prescaler bit 0
#define	WDP1	  1	; Watch Dog Timer Prescaler bit 1
#define	WDP2	  2	; Watch Dog Timer Prescaler bit 2
#define	WDE	  3	; Watch Dog Enable
#define	WDCE	  4	; Watchdog Change Enable
#define	WDP3	  5	; Watchdog Timer Prescaler Bit 3
#define	WDIE	  6	; Watchdog Timeout Interrupt Enable
#define	WDIF	  7	; Watchdog Timeout Interrupt Flag


; ***** CPU **************************
; SREG - Status Register
#define	SREG_C	  0	; Carry Flag
#define	SREG_Z	  1	; Zero Flag
#define	SREG_N	  2	; Negative Flag
#define	SREG_V	  3	; Two's Complement Overflow Flag
#define	SREG_S	  4	; Sign Bit
#define	SREG_H	  5	; Half Carry Flag
#define	SREG_T	  6	; Bit Copy Storage
#define	SREG_I	  7	; Global Interrupt Enable

; OSCCAL - Oscillator Calibration Value
#define	CAL0	  0	; Oscillator Calibration Value Bit0
#define	CAL1	  1	; Oscillator Calibration Value Bit1
#define	CAL2	  2	; Oscillator Calibration Value Bit2
#define	CAL3	  3	; Oscillator Calibration Value Bit3
#define	CAL4	  4	; Oscillator Calibration Value Bit4
#define	CAL5	  5	; Oscillator Calibration Value Bit5
#define	CAL6	  6	; Oscillator Calibration Value Bit6
#define	CAL7	  7	; Oscillator Calibration Value Bit7

; CLKPR - Clock Prescale Register
#define	CLKPS0	  0	; Clock Prescaler Select Bit 0
#define	CLKPS1	  1	; Clock Prescaler Select Bit 1
#define	CLKPS2	  2	; Clock Prescaler Select Bit 2
#define	CLKPS3	  3	; Clock Prescaler Select Bit 3
#define	CLKPCE	  7	; Clock Prescaler Change Enable

; SPMCSR - Store Program Memory Control and Status Register
#define	SELFPRGEN	  0	; Self Programming Enable
#define	PGERS	  1	; Page Erase
#define	PGWRT	  2	; Page Write
#define	BLBSET	  3	; Boot Lock Bit Set
#define	RWWSRE	  4	; Read-While-Write section read enable
#define	RWWSB	  6	; Read-While-Write Section Busy
#define	SPMIE	  7	; SPM Interrupt Enable

; MCUCR - MCU Control Register
#define	IVCE	  0	; 
#define	IVSEL	  1	; 
#define	PUD	  4	; 
#define	BODSE	  5	; BOD Sleep Enable
#define	BODS	  6	; BOD Sleep

; MCUSR - MCU Status Register
#define	PORF	  0	; Power-on reset flag
#define	EXTRF	  1	; External Reset Flag
#define	EXTREF	  EXTRF	; For compatibility
#define	BORF	  2	; Brown-out Reset Flag
#define	WDRF	  3	; Watchdog Reset Flag

; SMCR - Sleep Mode Control Register
#define	SE	  0	; Sleep Enable
#define	SM0	  1	; Sleep Mode Select Bit 0
#define	SM1	  2	; Sleep Mode Select Bit 1
#define	SM2	  3	; Sleep Mode Select Bit 2

; GPIOR2 - General Purpose I/O Register 2
#define	GPIOR20	  0	; 
#define	GPIOR21	  1	; 
#define	GPIOR22	  2	; 
#define	GPIOR23	  3	; 
#define	GPIOR24	  4	; 
#define	GPIOR25	  5	; 
#define	GPIOR26	  6	; 
#define	GPIOR27	  7	; 

; GPIOR1 - General Purpose I/O Register 1
#define	GPIOR10	  0	; 
#define	GPIOR11	  1	; 
#define	GPIOR12	  2	; 
#define	GPIOR13	  3	; 
#define	GPIOR14	  4	; 
#define	GPIOR15	  5	; 
#define	GPIOR16	  6	; 
#define	GPIOR17	  7	; 

; GPIOR0 - General Purpose I/O Register 0
#define	GPIOR00	  0	; 
#define	GPIOR01	  1	; 
#define	GPIOR02	  2	; 
#define	GPIOR03	  3	; 
#define	GPIOR04	  4	; 
#define	GPIOR05	  5	; 
#define	GPIOR06	  6	; 
#define	GPIOR07	  7	; 

; PRR - Power Reduction Register
#define	PRADC	  0	; Power Reduction ADC
#define	PRUSART0	  1	; Power Reduction USART
#define	PRSPI	  2	; Power Reduction Serial Peripheral Interface
#define	PRTIM1	  3	; Power Reduction Timer/Counter1
#define	PRTIM0	  5	; Power Reduction Timer/Counter0
#define	PRTIM2	  6	; Power Reduction Timer/Counter2
#define	PRTWI	  7	; Power Reduction TWI


; ***** EEPROM ***********************
; EEARL - EEPROM Address Register Low Byte
#define	EEAR0	  0	; EEPROM Read/Write Access Bit 0
#define	EEAR1	  1	; EEPROM Read/Write Access Bit 1
#define	EEAR2	  2	; EEPROM Read/Write Access Bit 2
#define	EEAR3	  3	; EEPROM Read/Write Access Bit 3
#define	EEAR4	  4	; EEPROM Read/Write Access Bit 4
#define	EEAR5	  5	; EEPROM Read/Write Access Bit 5
#define	EEAR6	  6	; EEPROM Read/Write Access Bit 6
#define	EEAR7	  7	; EEPROM Read/Write Access Bit 7

; EEARH - EEPROM Address Register High Byte
#define	EEAR8	  0	; EEPROM Read/Write Access Bit 8
#define	EEAR9	  1	; EEPROM Read/Write Access Bit 9

; EEDR - EEPROM Data Register
#define	EEDR0	  0	; EEPROM Data Register bit 0
#define	EEDR1	  1	; EEPROM Data Register bit 1
#define	EEDR2	  2	; EEPROM Data Register bit 2
#define	EEDR3	  3	; EEPROM Data Register bit 3
#define	EEDR4	  4	; EEPROM Data Register bit 4
#define	EEDR5	  5	; EEPROM Data Register bit 5
#define	EEDR6	  6	; EEPROM Data Register bit 6
#define	EEDR7	  7	; EEPROM Data Register bit 7

; EECR - EEPROM Control Register
#define	EERE	  0	; EEPROM Read Enable
#define	EEPE	  1	; EEPROM Write Enable
#define	EEMPE	  2	; EEPROM Master Write Enable
#define	EERIE	  3	; EEPROM Ready Interrupt Enable
#define	EEPM0	  4	; EEPROM Programming Mode Bit 0
#define	EEPM1	  5	; EEPROM Programming Mode Bit 1



; ***** LOCKSBITS ********************************************************
#define	LB1	  0	; Lock bit
#define	LB2	  1	; Lock bit
#define	BLB01	  2	; Boot Lock bit
#define	BLB02	  3	; Boot Lock bit
#define	BLB11	  4	; Boot lock bit
#define	BLB12	  5	; Boot lock bit


; ***** FUSES ************************************************************
; LOW fuse bits
#define	CKSEL0	  0	; Select Clock Source
#define	CKSEL1	  1	; Select Clock Source
#define	CKSEL2	  2	; Select Clock Source
#define	CKSEL3	  3	; Select Clock Source
#define	SUT0	  4	; Select start-up time
#define	SUT1	  5	; Select start-up time
#define	CKOUT	  6	; Clock output
#define	CKDIV8	  7	; Divide clock by 8

; HIGH fuse bits
#define	BOOTRST	  0	; Select reset vector
#define	BOOTSZ0	  1	; Select boot size
#define	BOOTSZ1	  2	; Select boot size
#define	EESAVE	  3	; EEPROM memory is preserved through chip erase
#define	WDTON	  4	; Watchdog Timer Always On
#define	SPIEN	  5	; Enable Serial programming and Data Downloading
#define	DWEN	  6	; debugWIRE Enable
#define	RSTDISBL	  7	; External reset disable

; EXTENDED fuse bits
#define	BODLEVEL0	  0	; Brown-out Detector trigger level
#define	BODLEVEL1	  1	; Brown-out Detector trigger level
#define	BODLEVEL2	  2	; Brown-out Detector trigger level



; ***** CPU REGISTER DEFINITIONS *****************************************
.def	XH	  r27
.def	XL	  r26
.def	YH	  r29
.def	YL	  r28
.def	ZH	  r31
.def	ZL	  r30



; ***** DATA MEMORY DECLARATIONS *****************************************
#define	FLASHEND	  0x3fff	; Note: Word address
#define	IOEND	  0x00ff
#define	SRAM_START	  0x0100
#define	SRAM_SIZE	  2048
#define	RAMEND	  0x08ff
#define	XRAMEND	  0x0000
#define	E2END	  0x03ff
#define	EEPROMEND	  0x03ff
#define	EEADRBITS	  10
#pragma AVRPART MEMORY PROG_FLASH 32768
#pragma AVRPART MEMORY EEPROM 1024
#pragma AVRPART MEMORY INT_SRAM SIZE 2048
#pragma AVRPART MEMORY INT_SRAM START_ADDR 0x100



; ***** BOOTLOADER DECLARATIONS ******************************************
#define	NRWW_START_ADDR	  0x3800
#define	NRWW_STOP_ADDR	  0x3fff
#define	RWW_START_ADDR	  0x0
#define	RWW_STOP_ADDR	  0x37ff
#define	PAGESIZE	  64
#define	FIRSTBOOTSTART	  0x3f00
#define	SECONDBOOTSTART	  0x3e00
#define	THIRDBOOTSTART	  0x3c00
#define	FOURTHBOOTSTART	  0x3800
#define	SMALLBOOTSTART	  FIRSTBOOTSTART
#define	LARGEBOOTSTART	  FOURTHBOOTSTART



; ***** INTERRUPT VECTORS ************************************************
#define	INT0addr	  0x0002	; External Interrupt Request 0
#define	INT1addr	  0x0004	; External Interrupt Request 1
#define	PCI0addr	  0x0006	; Pin Change Interrupt Request 0
#define	PCI1addr	  0x0008	; Pin Change Interrupt Request 0
#define	PCI2addr	  0x000a	; Pin Change Interrupt Request 1
#define	WDTaddr	  0x000c	; Watchdog Time-out Interrupt
#define	OC2Aaddr	  0x000e	; Timer/Counter2 Compare Match A
#define	OC2Baddr	  0x0010	; Timer/Counter2 Compare Match A
#define	OVF2addr	  0x0012	; Timer/Counter2 Overflow
#define	ICP1addr	  0x0014	; Timer/Counter1 Capture Event
#define	OC1Aaddr	  0x0016	; Timer/Counter1 Compare Match A
#define	OC1Baddr	  0x0018	; Timer/Counter1 Compare Match B
#define	OVF1addr	  0x001a	; Timer/Counter1 Overflow
#define	OC0Aaddr	  0x001c	; TimerCounter0 Compare Match A
#define	OC0Baddr	  0x001e	; TimerCounter0 Compare Match B
#define	OVF0addr	  0x0020	; Timer/Couner0 Overflow
#define	SPIaddr	  0x0022	; SPI Serial Transfer Complete
#define	URXCaddr	  0x0024	; USART Rx Complete
#define	UDREaddr	  0x0026	; USART, Data Register Empty
#define	UTXCaddr	  0x0028	; USART Tx Complete
#define	ADCCaddr	  0x002a	; ADC Conversion Complete
#define	ERDYaddr	  0x002c	; EEPROM Ready
#define	ACIaddr	  0x002e	; Analog Comparator
#define	TWIaddr	  0x0030	; Two-wire Serial Interface
#define	SPMRaddr	  0x0032	; Store Program Memory Read

#define	INT_VECTORS_SIZE	  52	; size in words

#endif  /* _M328PDEF_INC_ */

; ***** END OF FILE ******************************************************

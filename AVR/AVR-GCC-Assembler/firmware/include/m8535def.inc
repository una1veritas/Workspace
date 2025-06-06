;***** THIS IS A MACHINE GENERATED FILE - DO NOT EDIT ********************
;***** Created: 2006-10-03 09:16 ******* Source: ATmega8535.xml **********
;*************************************************************************
;* A P P L I C A T I O N   N O T E   F O R   T H E   A V R   F A M I L Y
;* 
;* Number            : AVR000
;* File Name         : "m8535def.inc"
;* Title             : Register/Bit Definitions for the ATmega8535
;* Date              : 2006-10-03
;* Version           : 2.24
;* Support E-mail    : avr@atmel.com
;* Target MCU        : ATmega8535
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

#ifndef _M8535DEF_INC_
#define _M8535DEF_INC_


#pragma partinc 0

; ***** SPECIFY DEVICE ***************************************************
.device ATmega8535
#pragma AVRPART ADMIN PART_NAME ATmega8535
.equ	SIGNATURE_000	= 0x1e
.equ	SIGNATURE_001	= 0x93
.equ	SIGNATURE_002	= 0x08

#pragma AVRPART CORE CORE_VERSION V2E


; ***** I/O REGISTER DEFINITIONS *****************************************
; NOTE:
; Definitions marked "MEMORY MAPPED"are extended I/O ports
; and cannot be used with IN/OUT instructions
.equ	SREG	= 0x3f
.equ	SPL	= 0x3d
.equ	SPH	= 0x3e
.equ	OCR0	= 0x3c
.equ	GICR	= 0x3b
.equ	GIFR	= 0x3a
.equ	TIMSK	= 0x39
.equ	TIFR	= 0x38
.equ	SPMCR	= 0x37
.equ	TWCR	= 0x36
.equ	MCUCR	= 0x35
.equ	MCUCSR	= 0x34
.equ	TCCR0	= 0x33
.equ	TCNT0	= 0x32
.equ	OSCCAL	= 0x31
.equ	SFIOR	= 0x30
.equ	TCCR1A	= 0x2f
.equ	TCCR1B	= 0x2e
.equ	TCNT1L	= 0x2c
.equ	TCNT1H	= 0x2d
.equ	OCR1AL	= 0x2a
.equ	OCR1AH	= 0x2b
.equ	OCR1BL	= 0x28
.equ	OCR1BH	= 0x29
.equ	ICR1L	= 0x26
.equ	ICR1H	= 0x27
.equ	TCCR2	= 0x25
.equ	TCNT2	= 0x24
.equ	OCR2	= 0x23
.equ	ASSR	= 0x22
.equ	WDTCR	= 0x21
.equ	UBRRH	= 0x20
.equ	UCSRC	= 0x20
.equ	EEARL	= 0x1e
.equ	EEARH	= 0x1f
.equ	EEDR	= 0x1d
.equ	EECR	= 0x1c
.equ	PORTA	= 0x1b
.equ	DDRA	= 0x1a
.equ	PINA	= 0x19
.equ	PORTB	= 0x18
.equ	DDRB	= 0x17
.equ	PINB	= 0x16
.equ	PORTC	= 0x15
.equ	DDRC	= 0x14
.equ	PINC	= 0x13
.equ	PORTD	= 0x12
.equ	DDRD	= 0x11
.equ	PIND	= 0x10
.equ	SPDR	= 0x0f
.equ	SPSR	= 0x0e
.equ	SPCR	= 0x0d
.equ	UDR	= 0x0c
.equ	UCSRA	= 0x0b
.equ	UCSRB	= 0x0a
.equ	UBRRL	= 0x09
.equ	ACSR	= 0x08
.equ	ADMUX	= 0x07
.equ	ADCSRA	= 0x06
.equ	ADCH	= 0x05
.equ	ADCL	= 0x04
.equ	TWDR	= 0x03
.equ	TWAR	= 0x02
.equ	TWSR	= 0x01
.equ	TWBR	= 0x00


; ***** BIT DEFINITIONS **************************************************

; ***** AD_CONVERTER *****************
; ADMUX - The ADC multiplexer Selection Register
.equ	MUX0	= 0	; Analog Channel and Gain Selection Bits
.equ	MUX1	= 1	; Analog Channel and Gain Selection Bits
.equ	MUX2	= 2	; Analog Channel and Gain Selection Bits
.equ	MUX3	= 3	; Analog Channel and Gain Selection Bits
.equ	MUX4	= 4	; Analog Channel and Gain Selection Bits
.equ	ADLAR	= 5	; Left Adjust Result
.equ	REFS0	= 6	; Reference Selection Bit 0
.equ	REFS1	= 7	; Reference Selection Bit 1

; ADCSRA - The ADC Control and Status register
.equ	ADCSR	= ADCSRA	; For compatibility
.equ	ADPS0	= 0	; ADC  Prescaler Select Bits
.equ	ADPS1	= 1	; ADC  Prescaler Select Bits
.equ	ADPS2	= 2	; ADC  Prescaler Select Bits
.equ	ADIE	= 3	; ADC Interrupt Enable
.equ	ADIF	= 4	; ADC Interrupt Flag
.equ	ADATE	= 5	; When this bit is written to one,the Timer/Counter2 prescaler will be reset.The bit will be cleared by hardware after the operation is performed.Writing a zero to this bit will have no effect.This bit will always be read as zero if Timer/Counter2 is clocked by the internal CPU clock.If this bit is written when Timer/Counter2 is operating in asynchronous mode,the bit will remain one until the prescaler has been reset.
.equ	ADFR	= ADATE	; For compatibility
.equ	ADSC	= 6	; ADC Start Conversion
.equ	ADEN	= 7	; ADC Enable

; ADCH - ADC Data Register High Byte
.equ	ADCH0	= 0	; ADC Data Register High Byte Bit 0
.equ	ADCH1	= 1	; ADC Data Register High Byte Bit 1
.equ	ADCH2	= 2	; ADC Data Register High Byte Bit 2
.equ	ADCH3	= 3	; ADC Data Register High Byte Bit 3
.equ	ADCH4	= 4	; ADC Data Register High Byte Bit 4
.equ	ADCH5	= 5	; ADC Data Register High Byte Bit 5
.equ	ADCH6	= 6	; ADC Data Register High Byte Bit 6
.equ	ADCH7	= 7	; ADC Data Register High Byte Bit 7

; ADCL - ADC Data Register Low Byte
.equ	ADCL0	= 0	; ADC Data Register Low Byte Bit 0
.equ	ADCL1	= 1	; ADC Data Register Low Byte Bit 1
.equ	ADCL2	= 2	; ADC Data Register Low Byte Bit 2
.equ	ADCL3	= 3	; ADC Data Register Low Byte Bit 3
.equ	ADCL4	= 4	; ADC Data Register Low Byte Bit 4
.equ	ADCL5	= 5	; ADC Data Register Low Byte Bit 5
.equ	ADCL6	= 6	; ADC Data Register Low Byte Bit 6
.equ	ADCL7	= 7	; ADC Data Register Low Byte Bit 7

; SFIOR - Special Function IO Register
.equ	ADTS0	= 5	; ADC Auto Trigger Source 0
.equ	ADTS1	= 6	; ADC Auto Trigger Source 1
.equ	ADTS2	= 7	; ADC Auto Trigger Source 2


; ***** ANALOG_COMPARATOR ************
; ACSR - Analog Comparator Control And Status Register
.equ	ACIS0	= 0	; Analog Comparator Interrupt Mode Select bit 0
.equ	ACIS1	= 1	; Analog Comparator Interrupt Mode Select bit 1
.equ	ACIC	= 2	; Analog Comparator Input Capture Enable
.equ	ACIE	= 3	; Analog Comparator Interrupt Enable
.equ	ACI	= 4	; Analog Comparator Interrupt Flag
.equ	ACO	= 5	; Analog Compare Output
.equ	ACBG	= 6	; Analog Comparator Bandgap Select
.equ	AINBG	= ACBG	; For compatibility
.equ	ACD	= 7	; Analog Comparator Disable


; ***** TWI **************************
; TWBR - TWI Bit Rate register
.equ	I2BR	= TWBR	; For compatibility
.equ	TWBR0	= 0	; 
.equ	TWBR1	= 1	; 
.equ	TWBR2	= 2	; 
.equ	TWBR3	= 3	; 
.equ	TWBR4	= 4	; 
.equ	TWBR5	= 5	; 
.equ	TWBR6	= 6	; 
.equ	TWBR7	= 7	; 

; TWCR - TWI Control Register
.equ	I2CR	= TWCR	; For compatibility
.equ	TWIE	= 0	; TWI Interrupt Enable
.equ	I2IE	= TWIE	; For compatibility
.equ	TWEN	= 2	; TWI Enable Bit
.equ	I2EN	= TWEN	; For compatibility
.equ	ENI2C	= TWEN	; For compatibility
.equ	TWWC	= 3	; TWI Write Collition Flag
.equ	I2WC	= TWWC	; For compatibility
.equ	TWSTO	= 4	; TWI Stop Condition Bit
.equ	I2STO	= TWSTO	; For compatibility
.equ	TWSTA	= 5	; TWI Start Condition Bit
.equ	I2STA	= TWSTA	; For compatibility
.equ	TWEA	= 6	; TWI Enable Acknowledge Bit
.equ	I2EA	= TWEA	; For compatibility
.equ	TWINT	= 7	; TWI Interrupt Flag
.equ	I2INT	= TWINT	; For compatibility

; TWSR - TWI Status Register
.equ	I2SR	= TWSR	; For compatibility
.equ	TWPS0	= 0	; TWI Prescaler
.equ	TWS0	= TWPS0	; For compatibility
.equ	I2GCE	= TWPS0	; For compatibility
.equ	TWPS1	= 1	; TWI Prescaler
.equ	TWS1	= TWPS1	; For compatibility
.equ	TWS3	= 3	; TWI Status
.equ	I2S3	= TWS3	; For compatibility
.equ	TWS4	= 4	; TWI Status
.equ	I2S4	= TWS4	; For compatibility
.equ	TWS5	= 5	; TWI Status
.equ	I2S5	= TWS5	; For compatibility
.equ	TWS6	= 6	; TWI Status
.equ	I2S6	= TWS6	; For compatibility
.equ	TWS7	= 7	; TWI Status
.equ	I2S7	= TWS7	; For compatibility

; TWDR - TWI Data register
.equ	I2DR	= TWDR	; For compatibility
.equ	TWD0	= 0	; TWI Data Register Bit 0
.equ	TWD1	= 1	; TWI Data Register Bit 1
.equ	TWD2	= 2	; TWI Data Register Bit 2
.equ	TWD3	= 3	; TWI Data Register Bit 3
.equ	TWD4	= 4	; TWI Data Register Bit 4
.equ	TWD5	= 5	; TWI Data Register Bit 5
.equ	TWD6	= 6	; TWI Data Register Bit 6
.equ	TWD7	= 7	; TWI Data Register Bit 7

; TWAR - TWI (Slave) Address register
.equ	I2AR	= TWAR	; For compatibility
.equ	TWGCE	= 0	; TWI General Call Recognition Enable Bit
.equ	TWA0	= 1	; TWI (Slave) Address register Bit 0
.equ	TWA1	= 2	; TWI (Slave) Address register Bit 1
.equ	TWA2	= 3	; TWI (Slave) Address register Bit 2
.equ	TWA3	= 4	; TWI (Slave) Address register Bit 3
.equ	TWA4	= 5	; TWI (Slave) Address register Bit 4
.equ	TWA5	= 6	; TWI (Slave) Address register Bit 5
.equ	TWA6	= 7	; TWI (Slave) Address register Bit 6


; ***** USART ************************
; UDR - USART I/O Data Register
.equ	UDR0	= 0	; USART I/O Data Register bit 0
.equ	UDR1	= 1	; USART I/O Data Register bit 1
.equ	UDR2	= 2	; USART I/O Data Register bit 2
.equ	UDR3	= 3	; USART I/O Data Register bit 3
.equ	UDR4	= 4	; USART I/O Data Register bit 4
.equ	UDR5	= 5	; USART I/O Data Register bit 5
.equ	UDR6	= 6	; USART I/O Data Register bit 6
.equ	UDR7	= 7	; USART I/O Data Register bit 7

; UCSRA - USART Control and Status Register A
.equ	MPCM	= 0	; Multi-processor Communication Mode
.equ	U2X	= 1	; Double the USART transmission speed
.equ	UPE	= 2	; Parity Error
.equ	PE	= UPE	; For compatibility
.equ	DOR	= 3	; Data overRun
.equ	FE	= 4	; Framing Error
.equ	UDRE	= 5	; USART Data Register Empty
.equ	TXC	= 6	; USART Transmitt Complete
.equ	RXC	= 7	; USART Receive Complete

; UCSRB - USART Control and Status Register B
.equ	TXB8	= 0	; Transmit Data Bit 8
.equ	RXB8	= 1	; Receive Data Bit 8
.equ	UCSZ2	= 2	; Character Size Bit 2
.equ	CHR9	= UCSZ2	; For compatibility
.equ	TXEN	= 3	; Transmitter Enable
.equ	RXEN	= 4	; Receiver Enable
.equ	UDRIE	= 5	; USART Data register Empty Interrupt Enable
.equ	TXCIE	= 6	; TX Complete Interrupt Enable
.equ	RXCIE	= 7	; RX Complete Interrupt Enable

; UCSRC - USART Control and Status Register C
.equ	UCPOL	= 0	; Clock Polarity
.equ	UCSZ0	= 1	; Character Size Bit 0
.equ	UCSZ1	= 2	; Character Size Bit 1
.equ	USBS	= 3	; Stop Bit Select
.equ	UPM0	= 4	; Parity Mode Bit 0
.equ	UPM1	= 5	; Parity Mode Bit 1
.equ	UMSEL	= 6	; USART Mode Select
.equ	URSEL	= 7	; Register Select

; UBRRH - USART Baud Rate Register High Byte
.equ	UBRR8	= 0	; USART Baud Rate Register bit 8
.equ	UBRR9	= 1	; USART Baud Rate Register bit 9
.equ	UBRR10	= 2	; USART Baud Rate Register bit 10
.equ	UBRR11	= 3	; USART Baud Rate Register bit 11
;.equ	URSEL	= 7	; Register Select


; ***** PORTA ************************
; PORTA - Port A Data Register
.equ	PORTA0	= 0	; Port A Data Register bit 0
.equ	PA0	= 0	; For compatibility
.equ	PORTA1	= 1	; Port A Data Register bit 1
.equ	PA1	= 1	; For compatibility
.equ	PORTA2	= 2	; Port A Data Register bit 2
.equ	PA2	= 2	; For compatibility
.equ	PORTA3	= 3	; Port A Data Register bit 3
.equ	PA3	= 3	; For compatibility
.equ	PORTA4	= 4	; Port A Data Register bit 4
.equ	PA4	= 4	; For compatibility
.equ	PORTA5	= 5	; Port A Data Register bit 5
.equ	PA5	= 5	; For compatibility
.equ	PORTA6	= 6	; Port A Data Register bit 6
.equ	PA6	= 6	; For compatibility
.equ	PORTA7	= 7	; Port A Data Register bit 7
.equ	PA7	= 7	; For compatibility

; DDRA - Port A Data Direction Register
.equ	DDA0	= 0	; Data Direction Register, Port A, bit 0
.equ	DDA1	= 1	; Data Direction Register, Port A, bit 1
.equ	DDA2	= 2	; Data Direction Register, Port A, bit 2
.equ	DDA3	= 3	; Data Direction Register, Port A, bit 3
.equ	DDA4	= 4	; Data Direction Register, Port A, bit 4
.equ	DDA5	= 5	; Data Direction Register, Port A, bit 5
.equ	DDA6	= 6	; Data Direction Register, Port A, bit 6
.equ	DDA7	= 7	; Data Direction Register, Port A, bit 7

; PINA - Port A Input Pins
.equ	PINA0	= 0	; Input Pins, Port A bit 0
.equ	PINA1	= 1	; Input Pins, Port A bit 1
.equ	PINA2	= 2	; Input Pins, Port A bit 2
.equ	PINA3	= 3	; Input Pins, Port A bit 3
.equ	PINA4	= 4	; Input Pins, Port A bit 4
.equ	PINA5	= 5	; Input Pins, Port A bit 5
.equ	PINA6	= 6	; Input Pins, Port A bit 6
.equ	PINA7	= 7	; Input Pins, Port A bit 7


; ***** PORTB ************************
; PORTB - Port B Data Register
.equ	PORTB0	= 0	; Port B Data Register bit 0
.equ	PB0	= 0	; For compatibility
.equ	PORTB1	= 1	; Port B Data Register bit 1
.equ	PB1	= 1	; For compatibility
.equ	PORTB2	= 2	; Port B Data Register bit 2
.equ	PB2	= 2	; For compatibility
.equ	PORTB3	= 3	; Port B Data Register bit 3
.equ	PB3	= 3	; For compatibility
.equ	PORTB4	= 4	; Port B Data Register bit 4
.equ	PB4	= 4	; For compatibility
.equ	PORTB5	= 5	; Port B Data Register bit 5
.equ	PB5	= 5	; For compatibility
.equ	PORTB6	= 6	; Port B Data Register bit 6
.equ	PB6	= 6	; For compatibility
.equ	PORTB7	= 7	; Port B Data Register bit 7
.equ	PB7	= 7	; For compatibility

; DDRB - Port B Data Direction Register
.equ	DDB0	= 0	; Port B Data Direction Register bit 0
.equ	DDB1	= 1	; Port B Data Direction Register bit 1
.equ	DDB2	= 2	; Port B Data Direction Register bit 2
.equ	DDB3	= 3	; Port B Data Direction Register bit 3
.equ	DDB4	= 4	; Port B Data Direction Register bit 4
.equ	DDB5	= 5	; Port B Data Direction Register bit 5
.equ	DDB6	= 6	; Port B Data Direction Register bit 6
.equ	DDB7	= 7	; Port B Data Direction Register bit 7

; PINB - Port B Input Pins
.equ	PINB0	= 0	; Port B Input Pins bit 0
.equ	PINB1	= 1	; Port B Input Pins bit 1
.equ	PINB2	= 2	; Port B Input Pins bit 2
.equ	PINB3	= 3	; Port B Input Pins bit 3
.equ	PINB4	= 4	; Port B Input Pins bit 4
.equ	PINB5	= 5	; Port B Input Pins bit 5
.equ	PINB6	= 6	; Port B Input Pins bit 6
.equ	PINB7	= 7	; Port B Input Pins bit 7


; ***** PORTC ************************
; PORTC - Port C Data Register
.equ	PORTC0	= 0	; Port C Data Register bit 0
.equ	PC0	= 0	; For compatibility
.equ	PORTC1	= 1	; Port C Data Register bit 1
.equ	PC1	= 1	; For compatibility
.equ	PORTC2	= 2	; Port C Data Register bit 2
.equ	PC2	= 2	; For compatibility
.equ	PORTC3	= 3	; Port C Data Register bit 3
.equ	PC3	= 3	; For compatibility
.equ	PORTC4	= 4	; Port C Data Register bit 4
.equ	PC4	= 4	; For compatibility
.equ	PORTC5	= 5	; Port C Data Register bit 5
.equ	PC5	= 5	; For compatibility
.equ	PORTC6	= 6	; Port C Data Register bit 6
.equ	PC6	= 6	; For compatibility
.equ	PORTC7	= 7	; Port C Data Register bit 7
.equ	PC7	= 7	; For compatibility

; DDRC - Port C Data Direction Register
.equ	DDC0	= 0	; Port C Data Direction Register bit 0
.equ	DDC1	= 1	; Port C Data Direction Register bit 1
.equ	DDC2	= 2	; Port C Data Direction Register bit 2
.equ	DDC3	= 3	; Port C Data Direction Register bit 3
.equ	DDC4	= 4	; Port C Data Direction Register bit 4
.equ	DDC5	= 5	; Port C Data Direction Register bit 5
.equ	DDC6	= 6	; Port C Data Direction Register bit 6
.equ	DDC7	= 7	; Port C Data Direction Register bit 7

; PINC - Port C Input Pins
.equ	PINC0	= 0	; Port C Input Pins bit 0
.equ	PINC1	= 1	; Port C Input Pins bit 1
.equ	PINC2	= 2	; Port C Input Pins bit 2
.equ	PINC3	= 3	; Port C Input Pins bit 3
.equ	PINC4	= 4	; Port C Input Pins bit 4
.equ	PINC5	= 5	; Port C Input Pins bit 5
.equ	PINC6	= 6	; Port C Input Pins bit 6
.equ	PINC7	= 7	; Port C Input Pins bit 7


; ***** PORTD ************************
; PORTD - Port D Data Register
.equ	PORTD0	= 0	; Port D Data Register bit 0
.equ	PD0	= 0	; For compatibility
.equ	PORTD1	= 1	; Port D Data Register bit 1
.equ	PD1	= 1	; For compatibility
.equ	PORTD2	= 2	; Port D Data Register bit 2
.equ	PD2	= 2	; For compatibility
.equ	PORTD3	= 3	; Port D Data Register bit 3
.equ	PD3	= 3	; For compatibility
.equ	PORTD4	= 4	; Port D Data Register bit 4
.equ	PD4	= 4	; For compatibility
.equ	PORTD5	= 5	; Port D Data Register bit 5
.equ	PD5	= 5	; For compatibility
.equ	PORTD6	= 6	; Port D Data Register bit 6
.equ	PD6	= 6	; For compatibility
.equ	PORTD7	= 7	; Port D Data Register bit 7
.equ	PD7	= 7	; For compatibility

; DDRD - Port D Data Direction Register
.equ	DDD0	= 0	; Port D Data Direction Register bit 0
.equ	DDD1	= 1	; Port D Data Direction Register bit 1
.equ	DDD2	= 2	; Port D Data Direction Register bit 2
.equ	DDD3	= 3	; Port D Data Direction Register bit 3
.equ	DDD4	= 4	; Port D Data Direction Register bit 4
.equ	DDD5	= 5	; Port D Data Direction Register bit 5
.equ	DDD6	= 6	; Port D Data Direction Register bit 6
.equ	DDD7	= 7	; Port D Data Direction Register bit 7

; PIND - Port D Input Pins
.equ	PIND0	= 0	; Port D Input Pins bit 0
.equ	PIND1	= 1	; Port D Input Pins bit 1
.equ	PIND2	= 2	; Port D Input Pins bit 2
.equ	PIND3	= 3	; Port D Input Pins bit 3
.equ	PIND4	= 4	; Port D Input Pins bit 4
.equ	PIND5	= 5	; Port D Input Pins bit 5
.equ	PIND6	= 6	; Port D Input Pins bit 6
.equ	PIND7	= 7	; Port D Input Pins bit 7


; ***** SPI **************************
; SPDR - SPI Data Register
.equ	SPDR0	= 0	; SPI Data Register bit 0
.equ	SPDR1	= 1	; SPI Data Register bit 1
.equ	SPDR2	= 2	; SPI Data Register bit 2
.equ	SPDR3	= 3	; SPI Data Register bit 3
.equ	SPDR4	= 4	; SPI Data Register bit 4
.equ	SPDR5	= 5	; SPI Data Register bit 5
.equ	SPDR6	= 6	; SPI Data Register bit 6
.equ	SPDR7	= 7	; SPI Data Register bit 7

; SPSR - SPI Status Register
.equ	SPI2X	= 0	; Double SPI Speed Bit
.equ	WCOL	= 6	; Write Collision Flag
.equ	SPIF	= 7	; SPI Interrupt Flag

; SPCR - SPI Control Register
.equ	SPR0	= 0	; SPI Clock Rate Select 0
.equ	SPR1	= 1	; SPI Clock Rate Select 1
.equ	CPHA	= 2	; Clock Phase
.equ	CPOL	= 3	; Clock polarity
.equ	MSTR	= 4	; Master/Slave Select
.equ	DORD	= 5	; Data Order
.equ	SPE	= 6	; SPI Enable
.equ	SPIE	= 7	; SPI Interrupt Enable


; ***** EEPROM ***********************
; EEDR - EEPROM Data Register
.equ	EEDR0	= 0	; EEPROM Data Register bit 0
.equ	EEDR1	= 1	; EEPROM Data Register bit 1
.equ	EEDR2	= 2	; EEPROM Data Register bit 2
.equ	EEDR3	= 3	; EEPROM Data Register bit 3
.equ	EEDR4	= 4	; EEPROM Data Register bit 4
.equ	EEDR5	= 5	; EEPROM Data Register bit 5
.equ	EEDR6	= 6	; EEPROM Data Register bit 6
.equ	EEDR7	= 7	; EEPROM Data Register bit 7

; EECR - EEPROM Control Register
.equ	EERE	= 0	; EEPROM Read Enable
.equ	EEWE	= 1	; EEPROM Write Enable
.equ	EEMWE	= 2	; EEPROM Master Write Enable
.equ	EEWEE	= EEMWE	; For compatibility
.equ	EERIE	= 3	; EEPROM Ready Interrupt Enable


; ***** TIMER_COUNTER_0 **************
; TCCR0 - Timer/Counter Control Register
.equ	CS00	= 0	; Clock Select 1
.equ	CS01	= 1	; Clock Select 1
.equ	CS02	= 2	; Clock Select 2
.equ	WGM01	= 3	; Waveform Generation Mode 1
.equ	CTC0	= WGM01	; For compatibility
.equ	COM00	= 4	; Compare match Output Mode 0
.equ	COM01	= 5	; Compare Match Output Mode 1
.equ	WGM00	= 6	; Waveform Generation Mode 0
.equ	PWM0	= WGM00	; For compatibility
.equ	FOC0	= 7	; Force Output Compare

; TCNT0 - Timer/Counter Register
.equ	TCNT0_0	= 0	; 
.equ	TCNT0_1	= 1	; 
.equ	TCNT0_2	= 2	; 
.equ	TCNT0_3	= 3	; 
.equ	TCNT0_4	= 4	; 
.equ	TCNT0_5	= 5	; 
.equ	TCNT0_6	= 6	; 
.equ	TCNT0_7	= 7	; 

; OCR0 - Output Compare Register
.equ	OCR0_0	= 0	; 
.equ	OCR0_1	= 1	; 
.equ	OCR0_2	= 2	; 
.equ	OCR0_3	= 3	; 
.equ	OCR0_4	= 4	; 
.equ	OCR0_5	= 5	; 
.equ	OCR0_6	= 6	; 
.equ	OCR0_7	= 7	; 

; TIMSK - Timer/Counter Interrupt Mask Register
.equ	TOIE0	= 0	; Timer/Counter0 Overflow Interrupt Enable
.equ	OCIE0	= 1	; Timer/Counter0 Output Compare Match Interrupt register

; TIFR - Timer/Counter Interrupt Flag register
.equ	TOV0	= 0	; Timer/Counter0 Overflow Flag
.equ	OCF0	= 1	; Output Compare Flag 0

; SFIOR - Special Function IO Register
.equ	PSR10	= 0	; Prescaler Reset Timer/Counter1 and Timer/Counter0


; ***** TIMER_COUNTER_1 **************
; TIMSK - Timer/Counter Interrupt Mask Register
.equ	TOIE1	= 2	; Timer/Counter1 Overflow Interrupt Enable
.equ	OCIE1B	= 3	; Timer/Counter1 Output CompareB Match Interrupt Enable
.equ	OCIE1A	= 4	; Timer/Counter1 Output CompareA Match Interrupt Enable
.equ	TICIE1	= 5	; Timer/Counter1 Input Capture Interrupt Enable

; TIFR - Timer/Counter Interrupt Flag register
.equ	TOV1	= 2	; Timer/Counter1 Overflow Flag
.equ	OCF1B	= 3	; Output Compare Flag 1B
.equ	OCF1A	= 4	; Output Compare Flag 1A
.equ	ICF1	= 5	; Input Capture Flag 1

; TCCR1A - Timer/Counter1 Control Register A
.equ	WGM10	= 0	; Waveform Generation Mode
.equ	PWM10	= WGM10	; For compatibility
.equ	WGM11	= 1	; Waveform Generation Mode
.equ	PWM11	= WGM11	; For compatibility
.equ	FOC1B	= 2	; Force Output Compare 1B
.equ	FOC1A	= 3	; Force Output Compare 1A
.equ	COM1B0	= 4	; Compare Output Mode 1B, bit 0
.equ	COM1B1	= 5	; Compare Output Mode 1B, bit 1
.equ	COM1A0	= 6	; Comparet Ouput Mode 1A, bit 0
.equ	COM1A1	= 7	; Compare Output Mode 1A, bit 1

; TCCR1B - Timer/Counter1 Control Register B
.equ	CS10	= 0	; Prescaler source of Timer/Counter 1
.equ	CS11	= 1	; Prescaler source of Timer/Counter 1
.equ	CS12	= 2	; Prescaler source of Timer/Counter 1
.equ	WGM12	= 3	; Waveform Generation Mode
.equ	CTC10	= WGM12	; For compatibility
.equ	CTC1	= WGM12	; For compatibility
.equ	WGM13	= 4	; Waveform Generation Mode
.equ	CTC11	= WGM13	; For compatibility
.equ	ICES1	= 6	; Input Capture 1 Edge Select
.equ	ICNC1	= 7	; Input Capture 1 Noise Canceler


; ***** TIMER_COUNTER_2 **************
; TIMSK - Timer/Counter Interrupt Mask register
.equ	TOIE2	= 6	; Timer/Counter2 Overflow Interrupt Enable
.equ	OCIE2	= 7	; Timer/Counter2 Output Compare Match Interrupt Enable

; TIFR - Timer/Counter Interrupt Flag Register
.equ	TOV2	= 6	; Timer/Counter2 Overflow Flag
.equ	OCF2	= 7	; Output Compare Flag 2

; TCCR2 - Timer/Counter2 Control Register
.equ	CS20	= 0	; Clock Select bit 0
.equ	CS21	= 1	; Clock Select bit 1
.equ	CS22	= 2	; Clock Select bit 2
.equ	WGM21	= 3	; Waveform Generation Mode
.equ	CTC2	= WGM21	; For compatibility
.equ	COM20	= 4	; Compare Output Mode bit 0
.equ	COM21	= 5	; Compare Output Mode bit 1
.equ	WGM20	= 6	; Waveform Genration Mode
.equ	PWM2	= WGM20	; For compatibility
.equ	FOC2	= 7	; Force Output Compare

; TCNT2 - Timer/Counter2
.equ	TCNT2_0	= 0	; Timer/Counter 2 bit 0
.equ	TCNT2_1	= 1	; Timer/Counter 2 bit 1
.equ	TCNT2_2	= 2	; Timer/Counter 2 bit 2
.equ	TCNT2_3	= 3	; Timer/Counter 2 bit 3
.equ	TCNT2_4	= 4	; Timer/Counter 2 bit 4
.equ	TCNT2_5	= 5	; Timer/Counter 2 bit 5
.equ	TCNT2_6	= 6	; Timer/Counter 2 bit 6
.equ	TCNT2_7	= 7	; Timer/Counter 2 bit 7

; OCR2 - Timer/Counter2 Output Compare Register
.equ	OCR2_0	= 0	; Timer/Counter2 Output Compare Register Bit 0
.equ	OCR2_1	= 1	; Timer/Counter2 Output Compare Register Bit 1
.equ	OCR2_2	= 2	; Timer/Counter2 Output Compare Register Bit 2
.equ	OCR2_3	= 3	; Timer/Counter2 Output Compare Register Bit 3
.equ	OCR2_4	= 4	; Timer/Counter2 Output Compare Register Bit 4
.equ	OCR2_5	= 5	; Timer/Counter2 Output Compare Register Bit 5
.equ	OCR2_6	= 6	; Timer/Counter2 Output Compare Register Bit 6
.equ	OCR2_7	= 7	; Timer/Counter2 Output Compare Register Bit 7

; ASSR - Asynchronous Status Register
.equ	TCR2UB	= 0	; Timer/counter Control Register2 Update Busy
.equ	OCR2UB	= 1	; Output Compare Register2 Update Busy
.equ	TCN2UB	= 2	; Timer/Counter2 Update Busy
.equ	AS2	= 3	; Asynchronous Timer/counter2

; SFIOR - Special Function IO Register
.equ	PSR2	= 1	; Prescaler Reset Timer/Counter2


; ***** EXTERNAL_INTERRUPT ***********
; GICR - General Interrupt Control Register
.equ	GIMSK	= GICR	; For compatibility
.equ	IVCE	= 0	; Interrupt Vector Change Enable
.equ	IVSEL	= 1	; Interrupt Vector Select
.equ	INT2	= 5	; External Interrupt Request 2 Enable
.equ	INT0	= 6	; External Interrupt Request 0 Enable
.equ	INT1	= 7	; External Interrupt Request 1 Enable

; GIFR - General Interrupt Flag Register
.equ	INTF2	= 5	; External Interrupt Flag 2
.equ	INTF0	= 6	; External Interrupt Flag 0
.equ	INTF1	= 7	; External Interrupt Flag 1

; MCUCR - General Interrupt Control Register
.equ	ISC00	= 0	; Interrupt Sense Control 0 Bit 0
.equ	ISC01	= 1	; Interrupt Sense Control 0 Bit 1
.equ	ISC10	= 2	; Interrupt Sense Control 1 Bit 0
.equ	ISC11	= 3	; Interrupt Sense Control 1 Bit 1

; MCUCSR - MCU Control And Status Register
.equ	ISC2	= 6	; Interrupt Sense Control 2


; ***** WATCHDOG *********************
; WDTCR - Watchdog Timer Control Register
.equ	WDTCSR	= WDTCR	; For compatibility
.equ	WDP0	= 0	; Watch Dog Timer Prescaler bit 0
.equ	WDP1	= 1	; Watch Dog Timer Prescaler bit 1
.equ	WDP2	= 2	; Watch Dog Timer Prescaler bit 2
.equ	WDE	= 3	; Watch Dog Enable
.equ	WDCE	= 4	; Watchdog Change Enable
.equ	WDTOE	= WDCE	; For compatibility


; ***** CPU **************************
; SREG - Status Register
.equ	SREG_C	= 0	; Carry Flag
.equ	SREG_Z	= 1	; Zero Flag
.equ	SREG_N	= 2	; Negative Flag
.equ	SREG_V	= 3	; Two's Complement Overflow Flag
.equ	SREG_S	= 4	; Sign Bit
.equ	SREG_H	= 5	; Half Carry Flag
.equ	SREG_T	= 6	; Bit Copy Storage
.equ	SREG_I	= 7	; Global Interrupt Enable

; MCUCR - MCU Control Register
;.equ	ISC00	= 0	; Interrupt Sense Control 0 Bit 0
;.equ	ISC01	= 1	; Interrupt Sense Control 0 Bit 1
;.equ	ISC10	= 2	; Interrupt Sense Control 1 Bit 0
;.equ	ISC11	= 3	; Interrupt Sense Control 1 Bit 1
.equ	SM0	= 4	; Sleep Mode Select
.equ	SM1	= 5	; Sleep Mode Select
.equ	SE	= 6	; Sleep Enable
.equ	SM2	= 7	; Sleep Mode Select

; MCUCSR - MCU Control And Status Register
.equ	MCUSR	= MCUCSR	; For compatibility
.equ	PORF	= 0	; Power-on reset flag
.equ	EXTRF	= 1	; External Reset Flag
.equ	BORF	= 2	; Brown-out Reset Flag
.equ	WDRF	= 3	; Watchdog Reset Flag

; OSCCAL - Oscillator Calibration Value
.equ	CAL0	= 0	; Oscillator Calibration Value Bit0
.equ	CAL1	= 1	; Oscillator Calibration Value Bit1
.equ	CAL2	= 2	; Oscillator Calibration Value Bit2
.equ	CAL3	= 3	; Oscillator Calibration Value Bit3
.equ	CAL4	= 4	; Oscillator Calibration Value Bit4
.equ	CAL5	= 5	; Oscillator Calibration Value Bit5
.equ	CAL6	= 6	; Oscillator Calibration Value Bit6
.equ	CAL7	= 7	; Oscillator Calibration Value Bit7

; SFIOR - Special Function IO Register
.equ	PUD	= 2	; Pull-up Disable
.equ	ACME	= 3	; Anlog Comparator Multiplexer Enable

; SPMCR - 
.equ	SPMEN	= 0	; Store Program Memory Enable
.equ	PGERS	= 1	; Page Erase
.equ	PGWRT	= 2	; Page Write
.equ	BLBSET	= 3	; Boot Lock Bit Set
.equ	RWWSRE	= 4	; Read-While-Write Section Read Enable
.equ	RWWSB	= 6	; Read-While-Write Section Busy
.equ	SPMIE	= 7	; SPM Interrupt Enable



; ***** LOCKSBITS ********************************************************
.equ	LB1	= 0	; Lock bit
.equ	LB2	= 1	; Lock bit
.equ	BLB01	= 2	; Boot Lock bit
.equ	BLB02	= 3	; Boot Lock bit
.equ	BLB11	= 4	; Boot lock bit
.equ	BLB12	= 5	; Boot lock bit


; ***** FUSES ************************************************************
; LOW fuse bits
.equ	CKSEL0	= 0	; Select Clock Source
.equ	CKSEL1	= 1	; Select Clock Source
.equ	CKSEL2	= 2	; Select Clock Source
.equ	CKSEL3	= 3	; Select Clock Source
.equ	SUT0	= 4	; Select start-up time
.equ	SUT1	= 5	; Select start-up time
.equ	BODEN	= 6	; Brown out detector enable
.equ	BODLEVEL	= 7	; Brown out detector trigger level

; HIGH fuse bits
.equ	BOOTRST	= 0	; Select Reset Vector
.equ	BOOTSZ0	= 1	; Select Boot Size
.equ	BOOTSZ1	= 2	; Select Boot Size
.equ	EESAVE	= 3	; EEPROM memory is preserved through chip erase
.equ	CKOPT	= 4	; Oscillator Options
.equ	SPIEN	= 5	; Enable Serial programming and Data Downloading
.equ	WDTON	= 6	; Watchdog timer always on
.equ	S8535C	= 7	; AT90S4434/8535 compabillity mode



; ***** CPU REGISTER DEFINITIONS *****************************************
.def	XH	= r27
.def	XL	= r26
.def	YH	= r29
.def	YL	= r28
.def	ZH	= r31
.def	ZL	= r30



; ***** DATA MEMORY DECLARATIONS *****************************************
.equ	FLASHEND	= 0x0fff	; Note: Word address
.equ	IOEND	= 0x003f
.equ	SRAM_START	= 0x0060
.equ	SRAM_SIZE	= 512
.equ	RAMEND	= 0x025f
.equ	XRAMEND	= 0x0000
.equ	E2END	= 0x01ff
.equ	EEPROMEND	= 0x01ff
.equ	EEADRBITS	= 9
#pragma AVRPART MEMORY PROG_FLASH 8192
#pragma AVRPART MEMORY EEPROM 512
#pragma AVRPART MEMORY INT_SRAM SIZE 512
#pragma AVRPART MEMORY INT_SRAM START_ADDR 0x60



; ***** BOOTLOADER DECLARATIONS ******************************************
.equ	NRWW_START_ADDR	= 0xc00
.equ	NRWW_STOP_ADDR	= 0xfff
.equ	RWW_START_ADDR	= 0x0
.equ	RWW_STOP_ADDR	= 0xbff
.equ	PAGESIZE	= 32
.equ	FIRSTBOOTSTART	= 0xf80
.equ	SECONDBOOTSTART	= 0xf00
.equ	THIRDBOOTSTART	= 0xe00
.equ	FOURTHBOOTSTART	= 0xc00
.equ	SMALLBOOTSTART	= FIRSTBOOTSTART
.equ	LARGEBOOTSTART	= FOURTHBOOTSTART



; ***** INTERRUPT VECTORS ************************************************
.equ	INT0addr	= 0x0001	; External Interrupt 0
.equ	INT1addr	= 0x0002	; External Interrupt 1
.equ	OC2addr	= 0x0003	; Timer/Counter2 Compare Match
.equ	OVF2addr	= 0x0004	; Timer/Counter2 Overflow
.equ	ICP1addr	= 0x0005	; Timer/Counter1 Capture Event
.equ	OC1Aaddr	= 0x0006	; Timer/Counter1 Compare Match A
.equ	OC1Baddr	= 0x0007	; Timer/Counter1 Compare Match B
.equ	OVF1addr	= 0x0008	; Timer/Counter1 Overflow
.equ	OVF0addr	= 0x0009	; Timer/Counter0 Overflow
.equ	SPIaddr	= 0x000a	; SPI Serial Transfer Complete
.equ	URXCaddr	= 0x000b	; USART, RX Complete
.equ	UDREaddr	= 0x000c	; USART Data Register Empty
.equ	UTXCaddr	= 0x000d	; USART, TX Complete
.equ	ADCCaddr	= 0x000e	; ADC Conversion Complete
.equ	ERDYaddr	= 0x000f	; EEPROM Ready
.equ	ACIaddr	= 0x0010	; Analog Comparator
.equ	TWIaddr	= 0x0011	; Two-wire Serial Interface
.equ	INT2addr	= 0x0012	; External Interrupt Request 2
.equ	OC0addr	= 0x0013	; TimerCounter0 Compare Match
.equ	SPMRaddr	= 0x0014	; Store Program Memory Read

.equ	INT_VECTORS_SIZE	= 21	; size in words

#pragma AVRPART CORE INSTRUCTIONS_NOT_SUPPORTED break

#endif  /* _M8535DEF_INC_ */

; ***** END OF FILE ******************************************************

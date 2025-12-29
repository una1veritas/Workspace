/*!
 * PIC18F47Q43/PIC18F47Q83/PIC18F47Q84 ROM image uploader and UART emulation firmware
 * This single source file contains all code
 *
 * Target: EMUZ80 with 6809+RAM
 * Compiler: MPLAB XC8 v2.40
 *
 * Modified by Satoshi Okue https://twitter.com/S_Okue
 * Version 0.1 2022/12/13
 * Version 0.2 2023/4/19
 * Version 0.3 2023/5/29
 */

/*
	PIC18F47Q43 ROM RAM and UART emulation firmware
	This single source file contains all code

	Target: EMUZ80 - The computer with only Z80 and PIC18F47Q43
	Compiler: MPLAB XC8 v2.36
	Written by Tetsuya Suzuki
*/


// CONFIG1
#pragma config FEXTOSC = OFF	// External Oscillator Selection (Oscillator not enabled)
#pragma config RSTOSC = HFINTOSC_64MHZ// Reset Oscillator Selection (HFINTOSC with HFFRQ = 64 MHz and CDIV = 1:1)

// CONFIG2
#pragma config CLKOUTEN = OFF	// Clock out Enable bit (CLKOUT function is disabled)
#pragma config PR1WAY = ON		// PRLOCKED One-Way Set Enable bit (PRLOCKED bit can be cleared and set only once)
#pragma config CSWEN = ON		// Clock Switch Enable bit (Writing to NOSC and NDIV is allowed)
#pragma config FCMEN = ON		// Fail-Safe Clock Monitor Enable bit (Fail-Safe Clock Monitor enabled)
#ifndef _18F47Q43
#pragma config JTAGEN = OFF
#pragma config FCMENP = OFF
#pragma config FCMENS = OFF
#endif

// CONFIG3
#pragma config MCLRE = EXTMCLR	// MCLR Enable bit (If LVP = 0, MCLR pin is MCLR; If LVP = 1, RE3 pin function is MCLR )
#pragma config PWRTS = PWRT_OFF // Power-up timer selection bits (PWRT is disabled)
#pragma config MVECEN = ON		// Multi-vector enable bit (Multi-vector enabled, Vector table used for interrupts)
#pragma config IVT1WAY = ON		// IVTLOCK bit One-way set enable bit (IVTLOCKED bit can be cleared and set only once)
#pragma config LPBOREN = OFF	// Low Power BOR Enable bit (Low-Power BOR disabled)
#pragma config BOREN = SBORDIS	// Brown-out Reset Enable bits (Brown-out Reset enabled , SBOREN bit is ignored)

// CONFIG4
#pragma config BORV = VBOR_1P9	// Brown-out Reset Voltage Selection bits (Brown-out Reset Voltage (VBOR) set to 1.9V)
#pragma config ZCD = OFF		// ZCD Disable bit (ZCD module is disabled. ZCD can be enabled by setting the ZCDSEN bit of ZCDCON)
#pragma config PPS1WAY = OFF	// PPSLOCK bit One-Way Set Enable bit (PPSLOCKED bit can be set and cleared repeatedly (subject to the unlock sequence))
#pragma config STVREN = ON		// Stack Full/Underflow Reset Enable bit (Stack full/underflow will cause Reset)
#pragma config LVP = ON			// Low Voltage Programming Enable bit (Low voltage programming enabled. MCLR/VPP pin function is MCLR. MCLRE configuration bit is ignored)
#pragma config XINST = OFF		// Extended Instruction Set Enable bit (Extended Instruction Set and Indexed Addressing Mode disabled)

// CONFIG5
#pragma config WDTCPS = WDTCPS_31// WDT Period selection bits (Divider ratio 1:65536; software control of WDTPS)
#pragma config WDTE = OFF		// WDT operating mode (WDT Disabled; SWDTEN is ignored)

// CONFIG6
#pragma config WDTCWS = WDTCWS_7// WDT Window Select bits (window always open (100%); software control; keyed access not required)
#pragma config WDTCCS = SC		// WDT input clock selector (Software Control)

// CONFIG7
#pragma config BBSIZE = BBSIZE_512// Boot Block Size selection bits (Boot Block size is 512 words)
#pragma config BBEN = OFF		// Boot Block enable bit (Boot block disabled)
#pragma config SAFEN = OFF		// Storage Area Flash enable bit (SAF disabled)
#ifdef _18F47Q43
#pragma config DEBUG = OFF
#endif

// CONFIG8
#pragma config WRTB = OFF		// Boot Block Write Protection bit (Boot Block not Write protected)
#pragma config WRTC = OFF		// Configuration Register Write Protection bit (Configuration registers not Write protected)
#pragma config WRTD = OFF		// Data EEPROM Write Protection bit (Data EEPROM not Write protected)
#pragma config WRTSAF = OFF	 	// SAF Write protection bit (SAF not Write Protected)
#pragma config WRTAPP = OFF	 	// Application Block write protection bit (Application Block not write protected)

// CONFIG10
#pragma config CP = OFF			// PFM and Data EEPROM Code Protection bit (PFM and Data EEPROM code protection disabled)

#ifndef _18F47Q43
// CONFIG9
#pragma config BOOTPINSEL = RC5	// CRC on boot output pin selection (CRC on boot output pin is RC5)
#pragma config BPEN = OFF		// CRC on boot output pin enable bit (CRC on boot output pin disabled)
#pragma config ODCON = OFF		// CRC on boot output pin open drain bit (Pin drives both high-going and low-going signals)

// CONFIG11
#pragma config BOOTSCEN = OFF	// CRC on boot scan enable for boot area (CRC on boot will not include the boot area of program memory in its calculation)
#pragma config BOOTCOE = HALT	// CRC on boot Continue on Error for boot areas bit (CRC on boot will stop device if error is detected in boot areas)
#pragma config APPSCEN = OFF	// CRC on boot application code scan enable (CRC on boot will not include the application area of program memory in its calculation)
#pragma config SAFSCEN = OFF	// CRC on boot SAF area scan enable (CRC on boot will not include the SAF area of program memory in its calculation)
#pragma config DATASCEN = OFF	// CRC on boot Data EEPROM scan enable (CRC on boot will not include data EEPROM in its calculation)
#pragma config CFGSCEN = OFF	// CRC on boot Config fuses scan enable (CRC on boot will not include the configuration fuses in its calculation)
#pragma config COE = HALT		// CRC on boot Continue on Error for non-boot areas bit (CRC on boot will stop device if error is detected in non-boot areas)
#pragma config BOOTPOR = OFF	// Boot on CRC Enable bit (CRC on boot will not run)

// CONFIG12
#pragma config BCRCPOLT = hFF	// Boot Sector Polynomial for CRC on boot bits 31-24 (Bits 31:24 of BCRCPOL are 0xFF)

// CONFIG13
#pragma config BCRCPOLU = hFF	// Boot Sector Polynomial for CRC on boot bits 23-16 (Bits 23:16 of BCRCPOL are 0xFF)

// CONFIG14
#pragma config BCRCPOLH = hFF	// Boot Sector Polynomial for CRC on boot bits 15-8 (Bits 15:8 of BCRCPOL are 0xFF)

// CONFIG15
#pragma config BCRCPOLL = hFF	// Boot Sector Polynomial for CRC on boot bits 7-0 (Bits 7:0 of BCRCPOL are 0xFF)

// CONFIG16
#pragma config BCRCSEEDT = hFF	// Boot Sector Seed for CRC on boot bits 31-24 (Bits 31:24 of BCRCSEED are 0xFF)

// CONFIG17
#pragma config BCRCSEEDU = hFF	// Boot Sector Seed for CRC on boot bits 23-16 (Bits 23:16 of BCRCSEED are 0xFF)

// CONFIG18
#pragma config BCRCSEEDH = hFF	// Boot Sector Seed for CRC on boot bits 15-8 (Bits 15:8 of BCRCSEED are 0xFF)

// CONFIG19
#pragma config BCRCSEEDL = hFF	// Boot Sector Seed for CRC on boot bits 7-0 (Bits 7:0 of BCRCSEED are 0xFF)

// CONFIG20
#pragma config BCRCEREST = hFF	// Boot Sector Expected Result for CRC on boot bits 31-24 (Bits 31:24 of BCRCERES are 0xFF)

// CONFIG21
#pragma config BCRCERESU = hFF	// Boot Sector Expected Result for CRC on boot bits 23-16 (Bits 23:16 of BCRCERES are 0xFF)

// CONFIG22
#pragma config BCRCERESH = hFF	// Boot Sector Expected Result for CRC on boot bits 15-8 (Bits 15:8 of BCRCERES are 0xFF)

// CONFIG23
#pragma config BCRCERESL = hFF	// Boot Sector Expected Result for CRC on boot bits 7-0 (Bits 7:0 of BCRCERES are 0xFF)

// CONFIG24
#pragma config CRCPOLT = hFF	// Non-Boot Sector Polynomial for CRC on boot bits 31-24 (Bits 31:24 of CRCPOL are 0xFF)

// CONFIG25
#pragma config CRCPOLU = hFF	// Non-Boot Sector Polynomial for CRC on boot bits 23-16 (Bits 23:16 of CRCPOL are 0xFF)

// CONFIG26
#pragma config CRCPOLH = hFF	// Non-Boot Sector Polynomial for CRC on boot bits 15-8 (Bits 15:8 of CRCPOL are 0xFF)

// CONFIG27
#pragma config CRCPOLL = hFF	// Non-Boot Sector Polynomial for CRC on boot bits 7-0 (Bits 7:0 of CRCPOL are 0xFF)

// CONFIG28
#pragma config CRCSEEDT = hFF	// Non-Boot Sector Seed for CRC on boot bits 31-24 (Bits 31:24 of CRCSEED are 0xFF)

// CONFIG29
#pragma config CRCSEEDU = hFF	// Non-Boot Sector Seed for CRC on boot bits 23-16 (Bits 23:16 of CRCSEED are 0xFF)

// CONFIG30
#pragma config CRCSEEDH = hFF	// Non-Boot Sector Seed for CRC on boot bits 15-8 (Bits 15:8 of CRCSEED are 0xFF)

// CONFIG31
#pragma config CRCSEEDL = hFF	// Non-Boot Sector Seed for CRC on boot bits 7-0 (Bits 7:0 of CRCSEED are 0xFF)

// CONFIG32
#pragma config CRCEREST = hFF	// Non-Boot Sector Expected Result for CRC on boot bits 31-24 (Bits 31:24 of CRCERES are 0xFF)

// CONFIG33
#pragma config CRCERESU = hFF	// Non-Boot Sector Expected Result for CRC on boot bits 23-16 (Bits 23:16 of CRCERES are 0xFF)

// CONFIG34
#pragma config CRCERESH = hFF	// Non-Boot Sector Expected Result for CRC on boot bits 15-8 (Bits 15:8 of CRCERES are 0xFF)

// CONFIG35
#pragma config CRCERESL = hFF	// Non-Boot Sector Expected Result for CRC on boot bits 7-0 (Bits 7:0 of CRCERES are 0xFF)
#endif
// #pragma config statements should precede project file includes.
// Use project enums instead of #define for ON and OFF.

#include <xc.h>
#include <stdio.h>

#include "rom.h"

#define CLK_6809 12000000UL	// 6809 clock frequency(Max 16MHz) 1MHz=1000000UL
// HD63C09 CPU Maximum Freq. is 3MHz, CLK_6809 is divided 4

#define ROM_TOP 0xC000		// ROM TOP Address
#define ROM_SIZE 0x4000		// 16K bytes
#define UART_DREG 0x8019	// Data REG
#define UART_CREG 0x8018	// Control REG

#define _XTAL_FREQ 64000000UL

#define UART3_BAUD 57600UL


//Address Bus
union {
	unsigned int w; //16 bits Address
	struct {
		unsigned char l;	//Address low
		unsigned char h;	//Address high
	};
} ab;

// UART3 Transmit
// Called by printf
void putch(char c) {
	while(!U3TXIF);		// Wait or Tx interrupt flag set
	U3TXB = c;			// Write data
}

/*
// UART3 Recive
char getch(void) {
	while(!U3RXIF);		// Wait for Rx interrupt flag set
	return U3RXB;		// Read data
}
*/

// Never called, logically
void __interrupt(irq(default),base(8)) Default_ISR(){}

// Called at MRDY falling edge(Immediately after CLK rasing)
void __interrupt(irq(CLC5),base(8)) CLC5_ISR(){
	CLC5IF = 0;					// Clear interrupt flag
	ab.h = PORTD;				// Read address high
	ab.l = PORTB;				// Read address low

	//6809 IO write cycle
	if(!RA4) {
		if(ab.w == UART_DREG)	// U3TXB
		U3TXB = PORTC;			// Write into	U3TXB

	//Release MRDY (D-FF reset)
	G3POL = 1;
	G3POL = 0;
	return;
	}

	//6809 IO read cycle
	TRISC = 0x00; 				// Set as output
	if(ab.w == UART_CREG)		// PIR9
		LATC = PIR9;			// Out PIR9
	else if(ab.w == UART_DREG)	// U3RXB
		LATC = U3RXB;			// Out U3RXB
	else						// Empty
		LATC = 0xff;			// Invalid address

	// Detect CLK raising edge
	while(!RA3);
	NOP();				// wait 62.5ns

	//Release MRDY (D-FF reset)
	G3POL = 1;
	G3POL = 0;

	//while(RA1);			// Detect E falling edge <2.75MHz (11MHz)
	TRISC = 0xff;		// Set data bus as input
}

// main routine
void main(void) {

	unsigned int i,j;

	// System initialize
	OSCFRQ = 0x08;	// 64MHz internal OSC

	// /RESET (RE2) output pin
	ANSELE2 = 0;	// Disable analog function
	LATE2 = 0;		// /Reset = Low
	TRISE2 = 0;		// Set as output

	// DMA/BREQ (RE0) output pin
	ANSELE0 = 0;	// Disable analog function
	LATE0 = 1;		// DMA/BREQ = High
	TRISE0 = 0;		// Set as output

	// Address bus A15-A8 pin
	ANSELD = 0x00;	// Disable analog function
	WPUD = 0xff;	// Week pull up
	TRISD = 0xff;	// Set as input

	// Address bus A7-A0 pin
	ANSELB = 0x00;	// Disable analog function
	WPUB = 0xff;	// Week pull up
	TRISB = 0xff;	// Set as input

	// Data bus D7-D0 pin
	ANSELC = 0x00;	// Disable analog function
	LATC = 0x00;
	TRISC = 0x00;	// Set as output

	// 6809 clock (RA3) by NCO FDC mode
	RA3PPS = 0x3f;	// RA1 assign NCO1
	ANSELA3 = 0;	// Disable analog function
	TRISA3 = 0;		// NCO output pin
	NCO1INC = (unsigned int)(400000UL / 30.5175781);
	NCO1CLK = 0x00; // Clock source Fosc
	NCO1PFM = 0;	// FDC mode
	NCO1OUT = 1;	// NCO output enable
	NCO1EN = 1;		// NCO enable

	// MRDY (RA0) output pin Low = Halt
	ANSELA0 = 0;	// Disable analog function
	RA0PPS = 0x00;	// LATA0 -> RA0
	LATA0 = 1;		// RDY = High
	TRISA0 = 0;		// Set as output

	// E (RA1) input pin
	ANSELA1 = 0;	// Disable analog function
	WPUA1 = 1;		// Week pull up
	TRISA1 = 1;		// Set as input

	// R/W (RA4) input pin
	ANSELA4 = 0;	// Disable analog function
	WPUA4 = 1;		// Week pull up
	TRISA4 = 1;		// Set as input

	// /WE (RA2) output pin
	ANSELA2 = 0;	// Disable analog function
	RA2PPS = 0x00;	// LATA2 -> RA2
	LATA2 = 1;		// /WE = High
	TRISA2 = 0;		// Set as output

	// /OE (RA5) output pin
	ANSELA5 = 0;	// Disable analog function
	RA5PPS = 0x00;	// LATA5 -> RA5
	LATA5 = 1;		// /OE = High
	TRISA5 = 0;		// Set as output

	// UART3 initialize, Fosc = 64 MHz, U3BRG = round(64,000,000 / (16 * 9600) - 1) = 416
	U3BRG = (_XTAL_FREQ/(16UL * UART3_BAUD)) - 1; //U3BRG = 416;	// 9600bps @ 64MHz
	U3RXEN = 1;		// Receiver enable
	U3TXEN = 1;		// Transmitter enable

	// UART3 Receiver
	ANSELA7 = 0;	// Disable analog function
	TRISA7 = 1;		// RX set as input
	U3RXPPS = 0x07;	//RA7->UART3:RX3;

	// UART3 Transmitter
	ANSELA6 = 0;	// Disable analog function
	LATA6 = 1;		// Default level
	TRISA6 = 0;		// TX set as output
	RA6PPS = 0x26;	//RA6->UART3:TX3;

	U3ON = 1;		// Serial port enable

	printf("\r\nMEZ6809RAM, UART baud %lu, ", UART3_BAUD);

	i = 0;
	do {
		while(RA1);
		while(!RA1);
		LATE0 = 0;		// DMA/BREQ = Low
		while(RA1);
		while(!RA1);
		while(RA1);
		TRISD = 0x00;	// A15-A8:Set as output
		TRISB = 0x00;	// A7-A0 :Set as output

		for(j = 0; j < 32; j++) {
			ab.w = i+ROM_TOP;
			LATD = ab.h;
			LATB = ab.l;
			LATA2 = 0;		// /WE=0
			LATC = rom[i];
			i++;
			LATA2 = 1;		// /WE=1
		}
		TRISD = 0xff;	// A15-A8:Set as input
		TRISB = 0xff;	// A7-A0 :Set as input

	while(RA1);
	while(!RA1);
	LATE0 = 1;			// DMA/BREQ = High
	__delay_us(30);
	} while (i < ROM_SIZE);


	// Address bus A15-A8 pin
	ANSELD = 0x00;	// Disable analog function
	WPUD = 0xff;	// Week pull up
	TRISD = 0xff;	// Set as input

	// Address bus A7-A0 pin
	ANSELB = 0x00;	// Disable analog function
	WPUB = 0xff;	// Week pull up
	TRISB = 0xff;	// Set as input

	// Data bus D7-D0 pin
	ANSELC = 0x00;	// Disable analog function
	WPUC = 0xff;	// Week pull up
	TRISC = 0xff;	// Set as input(default)

	NCO1EN = 0;		// NCO disable
	NCO1INC = (unsigned int)(CLK_6809 / 30.5175781);
	NCO1EN = 1;		// NCO enable

	printf("%2.3fMHz\r\n",NCO1INC * 30.5175781 / 4 / 1000000);

	//========== CLC input pin assign ===========
	// 0,1,4,5 = Port A, C
	// 2,3,6,7 = Port B, D
	CLCIN0PPS = 0x01;	// RA1 <- E
	CLCIN2PPS = 0x1f;	// RD7 <- A15
	CLCIN3PPS = 0x1e;	// RD6 <- A14
	CLCIN4PPS = 0x04;	// RA4 <- R/W
	CLCIN6PPS = 0x1d;	// RD5 <- A13
	CLCIN7PPS = 0x1c;	// RD4 <- A12

	//========== CLC1 /OE ==========
	CLCSELECT = 0;		// CLC1 select

	CLCnSEL0 = 53;		// CLC3 (/IORQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 0;		// CLCIN0PPS <- E
	CLCnSEL3 = 55;		// CLC5 (MRDY)

	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x08;	// R/W noninverted
	CLCnGLS2 = 0x20;	// E noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC1 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC2 /WE ==========
	CLCSELECT = 1;		// CLC2 select

	CLCnSEL0 = 53;		// CLC3 (/IOREQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 0;		// CLCIN0PPS <- E
	CLCnSEL3 = 55;		// CLC5 (MRDY)

	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x04;	// R/W inverted
	CLCnGLS2 = 0x20;	// E noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC2 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC3 /IORQ :IO area 0x8000 - 0x8FFF ==========
	CLCSELECT = 2;		// CLC3 select

	CLCnSEL0 = 2;		// CLCIN2PPS <- A15
	CLCnSEL1 = 3;		// CLCIN3PPS <- A14
	CLCnSEL2 = 6;		// CLCIN6PPS <- A13
	CLCnSEL3 = 7;		// CLCIN7PPS <- A12

	CLCnGLS0 = 0x02;	// A15 noninverted
	CLCnGLS1 = 0x04;	// A14 inverted
	CLCnGLS2 = 0x10;	// A13 inverted
	CLCnGLS3 = 0x40;	// A12 inverted

	CLCnPOL = 0x80;		// inverted the CLC3 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC5 MRDY ==========
	CLCSELECT = 4;		// CLC5 select

	CLCnSEL0 = 0;		// D-FF CLK CLCIN0PPS <- E
	CLCnSEL1 = 53;		// D-FF D CLC3 (/IORQ)
	CLCnSEL2 = 127;		// D-FF S NC
	CLCnSEL3 = 127;		// D-FF R NC

	CLCnGLS0 = 0x01;	// LCG1D1N
	CLCnGLS1 = 0x04;	// LCG2D1N
	CLCnGLS2 = 0x00;	// Connect none
	CLCnGLS3 = 0x00;	// Connect none

	CLCnPOL = 0x81;		// inverted the CLC5 output, G1 inverted
	CLCnCON = 0x8c;		// Select D-FF, falling edge inturrupt

	//========== CLC output pin assign ===========
	// 1,2,5,6 = Port A, C
	// 3,4,7,8 = Port B, D
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> MRDY

	// Unlock IVT
	IVTLOCK = 0x55;
	IVTLOCK = 0xAA;
	IVTLOCKbits.IVTLOCKED = 0x00;

	// Default IVT base address
	IVTBASE = 0x000008;

	// Lock IVT
	IVTLOCK = 0x55;
	IVTLOCK = 0xAA;
	IVTLOCKbits.IVTLOCKED = 0x01;

	// CLC VI enable
	CLC5IF = 0;			// Clear the CLC5 interrupt flag
	CLC5IE = 1;			// Enabling CLC5 interrupt

	// 6809 start
	GIE = 1;			// Global interrupt enable
	LATE2 = 1;			// Release reset

	while(1);			// All things come to those who wait
}

/*!
 * PIC18F47Q43/PIC18F47Q83/PIC18F47Q84 ROM image uploader and UART emulation firmware
 * This single source file contains all code
 *
 * Target: EMUZ80 with W65C02S+RAM
 * Compiler: MPLAB XC8 v2.40
 *
 * Modified by Satoshi Okue https://twitter.com/S_Okue
 * Version 0.1 2022/12/9
 * Version 0.2 2023/4/19
 */

/*
	PIC18F47Q43 ROM RAM and UART emulation firmware
	This single source file contains all code

	Target: EMUZ80 - The computer with only Z80 and PIC18F47Q43
	Compiler: MPLAB XC8 v2.36
	Written by Tetsuya Suzuki
*/

#include "config.h"

#include <xc.h>

#include <stdio.h>
#include <inttypes.h>

#include "uart.h"


#define _XTAL_FREQ 64000000UL

#define CLK_6502 2000000UL	// 6502 clock frequency(Max 16MHz) 1MHz=1000000UL
#define ROM_TOP 0xC000		// ROM TOP Address
#define ROM_SIZE 0x4000		// 16K bytes
#define UART_DREG 0xB019	// Data REG
#define UART_CREG 0xB018	// Control REG


//6502 ROM equivalent, see end of this file
extern const unsigned char rom[];

//Address Bus
union {
	unsigned int w; //16 bits Address
	struct {
		unsigned char l;	//Address low
		unsigned char h;	//Address high
	};
} ab;

#define DATABUS_MODE_INPUT      (WPUC = 0xff, TRISC = 0xff)	
#define DATABUS_MODE_OUTPUT     (WPUC = 0x00, TRISC = 0x00)
#define DATABUS_INPUT           PORTC
#define DATABUS_OUTPUT          LATC
// Set as input(default)
#define ADDRBUS_MODE_INTPUT 	(WPUD = 0xff, WPUB = 0xff, TRISD = 0xff, LTRISB = 0xff)
// Set as output
#define ADDRBUS_MODE_OUTPUT 	(WPUD = 0x00, WPUB = 0x00, TRISD = 0x00, TRISB = 0x00)
// Set as output
#define ADDRBUS_HIGH     LATD
#define ADDRBUS_LOW      LATB

#define W65C02_RW  RA4


// UART3 Transmit
void putch(char c) {
    while(!U3TXIF);		// Wait or Tx interrupt flag set
    U3TXB = c;			// Write data
}


// UART3 Recive
int getch(void) {
    while(!U3RXIF);		// Wait for Rx interrupt flag set
    return U3RXB;		// Read data
}


// Never called, logically
void __interrupt(irq(default),base(8)) Default_ISR(){}

void setup_clock() {
    /* set HFINTOSC Oscillator */
    //OSCCON1bits.NOSC = 6;
    /* set HFFRQ to 1 MHz */
    //OSCFRQbits.HFFRQ = 0;
	// System initialize
	OSCFRQ = 0x08;	// 64MHz internal OSC
}

void setup_6502_interface() {

    // 6502 clock (RA3) by NCO FDC mode
	RA3PPS = 0x3f;	// RA1 assign NCO1
	//ANSELA3 = 0;	// Disable analog function
    ANSELAbits.ANSELA3 = 0;
	//TRISA3 = 0;		// NCO output pin
    TRISAbits.TRISA3 = 0;
	NCO1INC = (unsigned int)(CLK_6502 / 30.5175781);
	NCO1CLK = 0x00; // Clock source Fosc
	NCO1PFM = 0;	// FDC mode
	NCO1OUT = 1;	// NCO output enable
	NCO1EN = 1;		// NCO enable

	// /RESET (RE2) output pin
	ANSELE2 = 0;	// Disable analog function
	LATE2 = 0;		// /Reset = Low
	TRISE2 = 0;		// Set as output

	// BE (RE0) output pin
	ANSELE0 = 0;	// Disable analog function
	LATE0 = 0;		// BE = Low
	TRISE0 = 0;		// Set as output

	// Address bus A15-A8 pin
    // setting the whole 8 bits on port
	ANSELD = 0x00;	// Disable analog function
	LATD = 0x00;
	TRISD = 0x00;	// Set as output

	// Address bus A7-A0 pin
	ANSELB = 0x00;	// Disable analog function
	LATB = 0x00;
	TRISB = 0x00;	// Set as output

	// Data bus D7-D0 pin
	ANSELC = 0x00;	// Disable analog function
	LATC = 0x00;
	TRISC = 0x00;	// Set as output


	// RDY (RA0) output pin Low = Halt
	ANSELA0 = 0;	// Disable analog function
	RA0PPS = 0x00;	// LATA0 -> RA0
	LATA0 = 1;		// RDY = High
	TRISA0 = 0;		// Set as output

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

	// UART3 initialize
	U3BRG = 68; // 19200  //416;	// Console Serial Baud rate 9600bps @ 64MHz
	U3RXEN = 1;		// Receiver enable
	U3TXEN = 1;		// Transmitter enable

	// UART3 Receiver
	ANSELA7 = 0;	// Disable analog function
	TRISA7 = 1;		// RX set as input
	U3RXPPS = 0x07;     //RA7->UART3:RX3;

	// UART3 Transmitter
	ANSELA6 = 0;	// Disable analog function
	LATA6 = 1;		// Default level
	TRISA6 = 0;		// TX set as output
	RA6PPS = 0x26;	//RA6->UART3:TX3;

	U3ON = 1;		// Serial port enable

    printf("\r\nSystem initialized. \r\nHello, UART enabled.\r\n");
}

void setup_CLC() {
   	//========== CLC input pin assign ===========
	// 0,1,4,5 = Port A, C
	// 2,3,6,7 = Port B, D
	CLCIN2PPS = 0x1f;	// RD7 <- A15
	CLCIN3PPS = 0x1e;	// RD6 <- A14
	CLCIN4PPS = 0x04;	// RA4 <- R/W
	CLCIN6PPS = 0x1d;	// RD5 <- A13
	CLCIN7PPS = 0x1c;	// RD4 <- A12

	//========== CLC1 /OE ==========
	CLCSELECT = 0;		// CLC1 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// CLC3 (/IORQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 42;		// NCO1
	CLCnSEL3 = 55;		// CLC5 (RDY)
 
	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x08;	// R/W noninverted
	CLCnGLS2 = 0x20;	// NCO1 noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC1 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC2 /WE ==========
	CLCSELECT = 1;		// CLC2 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// CLC3 (/IOREQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 42;		// NCO1
	CLCnSEL3 = 55;		// CLC5 (RDY)
 
	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x04;	// R/W inverted
	CLCnGLS2 = 0x20;	// NCO1 noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC2 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC3 /IORQ :IO area 0xB000 - 0xBFFF ==========
	CLCSELECT = 2;		// CLC3 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 2;		// CLCIN2PPS <- A15
	CLCnSEL1 = 3;		// CLCIN3PPS <- A14
	CLCnSEL2 = 6;		// CLCIN6PPS <- A13
	CLCnSEL3 = 7;		// CLCIN7PPS <- A12
 
	CLCnGLS0 = 0x02;	// A15 noninverted
	CLCnGLS1 = 0x04;	// A14 inverted
	CLCnGLS2 = 0x20;	// A13 noninverted
	CLCnGLS3 = 0x80;	// A12 noninverted

	CLCnPOL = 0x80;		// inverted the CLC3 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC5 RDY ==========
	CLCSELECT = 4;		// CLC5 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// D-FF CLK <- CLC3 (/IORQ)
	CLCnSEL1 = 127;		// D-FF D NC
	CLCnSEL2 = 127;		// D-FF S NC
	CLCnSEL3 = 127;		// D-FF R NC

	CLCnGLS0 = 0x1;		// LCG1D1N
	CLCnGLS1 = 0x0;		// Connect none
	CLCnGLS2 = 0x0;		// Connect none
	CLCnGLS3 = 0x0;		// Connect none

	CLCnPOL = 0x82;		// inverted the CLC5 output, G2 inverted
	CLCnCON = 0x84;		// Select D-FF

	//========== CLC output pin assign ===========
	// 1,2,5,6 = Port A, C
	// 3,4,7,8 = Port B, D
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> RDY

    printf("CLC setup finished.\r\n");
}

void setup_busmode_6502() {
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
}

void setup_InterruptVectorTable() {
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
}

uint32_t memory_check(uint32_t startaddr, uint32_t endaddr) {
    uint32_t stopaddr = endaddr;
    uint8_t val, wval;
    ADDRBUS_MODE_OUTPUT;
	for(uint32_t i = startaddr; i < endaddr; i++) {
        ab.w = (uint16_t) (startaddr+i);
		LATD = ab.h;
		LATB = ab.l;
        LATA5 = 0;		// _OE=0
        DATABUS_MODE_INPUT;
        val = DATABUS_INPUT;
		LATA5 = 1;		// _OE=1
        
        wval = val^0x55;
        DATABUS_MODE_OUTPUT;
        DATABUS_OUTPUT = wval;
        LATA2 = 0;		// /WE=0
        __delay_us(1);
		LATA2 = 1;		// /WE=1
        
        DATABUS_MODE_INPUT;
        LATA5 = 0;		// _OE=0
        val = DATABUS_INPUT;
		LATA5 = 1;		// _OE=1
        
        if (wval != val) {
            printf("error at %04lx: written %02x, read %02x.\r\n", i+ROM_TOP, rom[i],val);
            stopaddr = startaddr+i;
            break;
        }
        
        wval ^= 0x55;
        DATABUS_MODE_OUTPUT;
        DATABUS_OUTPUT = wval;
        LATA2 = 0;		// /WE=0
        __delay_us(1);
		LATA2 = 1;		// /WE=1
        
	}
    return stopaddr;
}

uint32_t transfer_to_sram(const uint8_t arr[], uint32_t startaddr, uint32_t size) {
    printf("Transferring data (%luk bytes) to SRAM...\r\n",size/1024);
    
    ADDRBUS_MODE_OUTPUT;
    DATABUS_MODE_OUTPUT;
	for(uint32_t i = 0; i < size; i++) {
		ab.w = (uint16_t) (startaddr + i);
		ADDRBUS_HIGH = ab.h;
		ADDRBUS_LOW  = ab.l;
        DATABUS_OUTPUT = arr[i];
		LATA2 = 0;		// /WE=0
        __delay_us(1);
		LATA2 = 1;		// /WE=1
    }
    
    // verify
    uint8_t val;
    uint32_t errcount = 0;
    DATABUS_MODE_INPUT;
	for(uint32_t i = 0; i < size; i++) {
		ab.w = (uint16_t) (startaddr + i);
		ADDRBUS_HIGH = ab.h;
		ADDRBUS_LOW  = ab.l;
		LATA5 = 0;		// _OE=0
        __delay_us(1);
        val = DATABUS_INPUT;
		LATA5 = 1;		// _OE=1
        if (arr[i] != val) {
            errcount += 1;
        }
    }
    if ( errcount == 0 ) {
        printf("transfer and verify done.\r\n");
    } else {
        printf("%lu errors detected.\r\n", errcount);
    }
    return errcount;
}

// main routine
void main(void) {

	//unsigned int i;
    
    setup_clock();
    setup_6502_interface(); // 	// BE (RE0) output pin LOW
    
    uint32_t stopaddr = memory_check(0, 0x10000);
    printf("stopaddr = %04lx.\r\n", stopaddr);
    transfer_to_sram(rom, ROM_TOP, ROM_SIZE);
    
    setup_busmode_6502();
    
	printf("\r\nMEZ6502RAM %2.3fMHz\r\n",NCO1INC * 30.5175781 / 1000000);

    setup_CLC();
    setup_InterruptVectorTable();
	
	// 6502 start
    printf("\r\nStarting 65C02...\r\n");
    
	GIE = 1;			// Global interrupt enable
	LATE0 = 1;			// BE = High
	LATE2 = 1;			// Release reset

	while(1){
		while(CLC5OUT); //RDY == 1
		ab.h = PORTD;				// Read address high
		ab.l = PORTB;				// Read address low

		//6502 IO write cycle
		if (!W65C02_RW) /*(!RA4)*/ {
			if(ab.w == UART_DREG) {	// U3TXB
                while(!U3TXIF);
                U3TXB = PORTC;			// Write into	U3TXB
            }
            
			//Release RDY (D-FF reset)
			G3POL = 1;
			G3POL = 0;
		} else {
		//6502 IO read cycle
			DATABUS_MODE_OUTPUT; //TRISC = 0x00;				// Set Data Bus as output
			if(ab.w == UART_CREG) {		// PIR9
				LATC = PIR9;			// Out PIR9
			} else if(ab.w == UART_DREG) {	// U3RXB
                while(!U3RXIF);
				LATC = U3RXB;			// Out U3RXB
			} else {						// Empty
				LATC = 0xff;			// Invalid address
            }
			// Detect CLK falling edge
			while(RA3);
			//Release RDY (D-FF reset)
			G3POL = 1;
			DATABUS_MODE_INPUT; //TRISC = 0xff;				// Set Data Bus as input
			G3POL = 0;
		}
	}
}

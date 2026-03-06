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


#include "config.h"

#include <xc.h>
#include <stdio.h>

#include "pic18common.h"

#include "rom.h"

#define CLK_6809 12000000UL	// 6809 clock frequency(Max 16MHz) 1MHz=1000000UL
// HD63C09 CPU Maximum Freq. is 3MHz, CLK_6809 / 4

#define ROM_TOP 0xC000		// ROM TOP Address
#define ROM_SIZE 0x4000		// 16K bytes
#define UART_CREG 0x8018	// Control REG
#define UART_DREG 0x8019	// Data REG

#define _XTAL_FREQ 64000000UL

#define UART3_BAUD 57600UL

#define MPU_RESET   E2
#define MPU_BREQ    E0
#define MPU_E       A1
#define MPU_MRDY    A0
#define MPU_RW      A4
#define MPU_CLK     A3

#define SRAM_WE     A2
#define SRAM_OE     A5

#define ACIA_STA_PERR   (1<<0)
#define ACIA_STA_FERR   (1<<1)
#define ACIA_STA_OVRUN  (1<<2)
#define ACIA_STA_RDRF   (1<<3)
#define ACIA_STA_TDRE   (1<<4)


#define DATABUS_MODE_INPUT      (WPUC = 0xff, TRISC = 0xff)
#define DATABUS_MODE_OUTPUT     (WPUC = 0x00, TRISC = 0x00)
#define DATABUS_RD              PORTC
#define DATABUS_WR              LATC
// Set as input(default)
#define ADDRBUS_MODE_INPUT      (WPUD = 0xff, WPUB = 0xff, TRISD = 0xff, TRISB = 0xff)
// Set as output
#define ADDRBUS_MODE_OUTPUT 	(WPUD = 0x00, WPUB = 0x00, TRISD = 0x00, TRISB = 0x00)
// Set as output
#define ADDRBUS_HIGH_WR    LATD
#define ADDRBUS_LOW_WR     LATB
#define ADDRBUS_HIGH_RD    PORTD
#define ADDRBUS_LOW_RD     PORTB

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
    
	//6809 Write cycle
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

void setup(void) {
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
}


void set_busmode_DMA() {
    // ensure to avoid bus conflict
    OUT(W65C02_BE) = LOW;
    !!! check 6809 bus enable sequence !!!
/*
 * 		while(RA1);
		while(!RA1);
		LATE0 = 0;		// DMA/BREQ = Low
		while(RA1);
		while(!RA1);
		while(RA1);
 */
    OUT(SRAM_OE) = HIGH;
    OUT(SRAM_WE) = HIGH; 
    
    // set PPS OUTPUT selection to LATxy
    RA5PPS = 0x00;		// LATxy
	RA2PPS = 0x00;
	RA0PPS = 0x00;
    
    DATABUS_MODE_INPUT;
    ADDRBUS_MODE_OUTPUT;
}

void set_busmode_6502() {
    DATABUS_MODE_INPUT;
    ADDRBUS_MODE_INPUT;
    
    //========== CLC output pin assign ===========
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> RDY
}

inline void set_addr_bus(const uint16_t addr) {
    uint8_t * ptr = (uint8_t *) & addr;
    //ADDRBUS_MODE_OUTPUT;
    ADDRBUS_LOW_WR  = *ptr++;   //(uint8_t) addr; //LATB = ab.l;
    ADDRBUS_HIGH_WR = *ptr;     //((uint8_t *)&addr)[1]; // *(((uint8_t *) & addr) + 1); //LATD = ab.h;
}

inline uint16_t get_addr_bus() {
    uint16_t addr;
    uint8_t * ptr = (uint8_t *) &addr;
    //ADDRBUS_MODE_INPUT;
    *ptr++ = ADDRBUS_LOW_RD;
    *ptr   = ADDRBUS_HIGH_RD;
    return addr;
}

uint8_t sram_read(const uint16_t addr) {
    uint8_t data;
    OUT(SRAM_WE) = HIGH;
    DATABUS_MODE_INPUT;
    set_addr_bus(addr);
    OUT(SRAM_OE) = LOW; //LATA5 = 0;		// _OE=0
    NOP(); //__delay_us(1);
    data = DATABUS_RD;
	OUT(SRAM_OE) = HIGH; //LATA5 = 1;		// _OE=1
    return data;
}
/*
void sram_write(const uint16_t addr, const uint8_t data) {
	OUT(SRAM_OE) = HIGH; 
    DATABUS_MODE_OUTPUT;
    set_addr_bus(addr);
    DATABUS_WR = data;
    OUT(SRAM_WE) = LOW; //LATA2 = 0;		// /WE=0
    NOP(); //__delay_us(1);
    OUT(SRAM_WE) = HIGH; //LATA2 = 1;		// /WE=1    
}
*/

void sram_block_write(uint16_t addr, const uint8_t data[], const uint16_t bytes) {
	OUT(SRAM_OE) = HIGH; 
    DATABUS_MODE_OUTPUT;
    for(uint16_t bytecount = 0; bytecount < bytes; ++bytecount, ++addr) {
        set_addr_bus(addr);
        DATABUS_WR = data[bytecount];
        OUT(SRAM_WE) = LOW; //LATA2 = 0;		// /WE=0
        NOP(); //__delay_us(1);
        OUT(SRAM_WE) = HIGH; //LATA2 = 1;		// /WE=1    
    }
}

uint16_t load_rom(const uint8_t rom[], const uint16_t dst_addr, const uint16_t bytesize) {
    // returns the number of read/write check failures
    uint16_t errors = 0;
    uint8_t onebyte;
    /*
    for(uint16_t bytecount = 0; bytecount < bytesize; ++bytecount) {
        sram_write(dst_addr + bytecount, rom[bytecount]);
    }
     * */
    sram_block_write(dst_addr, rom, bytesize);
    for(uint16_t bytecount = 0; bytecount < bytesize; ++bytecount) {
        onebyte = sram_read(dst_addr + bytecount);
        if ( onebyte != rom[bytecount] ) {
            if ( errors == 0 ) {
                printf("the first error occurred at %04x.\n", dst_addr + bytecount);
            }
            ++errors;
        }
    }
    return errors;
}

// main routine
void main(void) {

	unsigned int i,j;

    setup();

	printf("\r\nMEZ6809RAM, UART baud %lu, \r\n", UART3_BAUD);

    set_busmode_DMA();
    load_rom(rom, ROM_TOP, ROM_SIZE);
    /*
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
    */
    set_busmode_6809();
    
	printf("Finished loading rom contents.\r\n");
    
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

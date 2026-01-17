 /*
 * MAIN Generated Driver File
 * 
 * @file main.c
 * 
 * @defgroup main MAIN
 * 
 * @brief This is the generated driver implementation file for the MAIN driver.
 *
 * @version MAIN Driver Version 1.0.2
 *
 * @version Package Version: 3.1.2
*/

#include <xc.h>
#include "config_bits.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "pic18common.h"
#include "system.h"

#ifndef _XTAL_FREQ
#define _XTAL_FREQ 64000000UL
#endif

#define CLK_6502_FREQ   2000000UL	// 6502 clock frequency(Max 16MHz) 1MHz=1000000UL

#define UART_CREG   0xB018	// Control REG
#define UART_DREG   0xB019	// Data REG

//6551 style
#define ACIA_DAT    0xB098
#define ACIA_STA    0xB099 
#define ACIA_CMD    0xB09A
//#define ACIA_CTL    0xB09B

/*
 * ACIA (6551) Status Register
 * bits
 * 0 -- parity err, 1 -- framing err, 2 -- overrun,
 * 3 -- Receive Data Reg. full
 * 4 -- Transmit Data Reg. empty
 * 5, 6, 7 --- /DCD, /DSR, /IRQ
 */

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

#define W65C02_CLK  RA3
#define W65C02_RW   RA4
#define W65C02_RDY  LATA0
#define W65C02_RST  LATE2
#define W65C02_BE   LATE0

#define SRAM_WE     LATA2
#define SRAM_OE     LATA5


#define ROM_TOP         0xC000		// ROM TOP Address
#define ROM_SIZE        0x4000		// 16K bytes

//6502 ROM equivalent, see end of this file
//extern const unsigned char rom_EhBASIC[];
extern const unsigned char rom_ehbasic_acia[];
#define ROM rom_ehbasic_acia


void W65C02_interface_init() {
	// /RESET (RE2) output pin
	LATE2 = 0;		// /Reset = Low
	TRISE2 = 0;		// Set as output

	// BE (RE0) output pin
	LATE0 = 0;		// BE = Low
	TRISE0 = 0;		// Set as output

	// Address bus A15-A8 pin
    // setting the whole 8 bits on port
	LATD = 0x00;
	TRISD = 0x00;	// Set as output

	// Address bus A7-A0 pin
	LATB = 0x00;
	TRISB = 0x00;	// Set as output

	// Data bus D7-D0 pin
	LATC = 0x00;
	TRISC = 0x00;	// Set as output


	// RDY (RA0) output pin Low = Halt
	RA0PPS = 0x00;	// LATA0 -> RA0
	LATA0 = 1;		// RDY = High
	TRISA0 = 0;		// Set as output

	// R/W (RA4) input pin
	WPUA4 = 1;		// Week pull up
	TRISA4 = 1;		// Set as input

	// /WE (RA2) output pin
	RA2PPS = 0x00;	// LATA2 -> RA2
	LATA2 = 1;		// /WE = High
	TRISA2 = 0;		// Set as output

	// /OE (RA5) output pin
	RA5PPS = 0x00;	// LATA5 -> RA5
	LATA5 = 1;		// /OE = High
	TRISA5 = 0;		// Set as output

}

void set_busmode_DMA() {
    // ensure to avoid bus conflict
    W65C02_BE = LOW;
    SRAM_OE = HIGH;
    SRAM_WE = HIGH; 
    
    DATABUS_MODE_INPUT;
    ADDRBUS_MODE_OUTPUT;
}

void set_busmode_6502() {
    DATABUS_MODE_INPUT;
    ADDRBUS_MODE_INPUT;
    
    //========== CLC output pin assign ===========
	// 1,2,5,6 = Port A, C
	// 3,4,7,8 = Port B, D
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> RDY
}

inline void set_addr_bus(const uint16_t addr) {
    //ADDRBUS_MODE_OUTPUT;
    ADDRBUS_HIGH_WR = ((uint8_t *)&addr)[1]; //*(((uint8_t *) & addr) + 1); //LATD = ab.h;
    ADDRBUS_LOW_WR  = (uint8_t) addr; //LATB = ab.l;
}

uint8_t sram_read(const uint16_t addr) {
    uint8_t data;
    SRAM_WE = HIGH;
    DATABUS_MODE_INPUT;
    set_addr_bus(addr);
    SRAM_OE = LOW; //LATA5 = 0;		// _OE=0
    NOP(); //__delay_us(1);
    data = DATABUS_RD;
	SRAM_OE = HIGH; //LATA5 = 1;		// _OE=1
    return data;
}

void sram_write(const uint16_t addr, const uint8_t data) {
	SRAM_OE = HIGH; 
    DATABUS_MODE_OUTPUT;
    set_addr_bus(addr);
    DATABUS_WR = data;
    SRAM_WE = LOW; //LATA2 = 0;		// /WE=0
    NOP(); //__delay_us(1);
    SRAM_WE = HIGH; //LATA2 = 1;		// /WE=1    
}

uint32_t memory_check(uint32_t startaddr, uint32_t endaddr) {
    uint32_t stopaddr = endaddr;
    uint8_t val, wval;
    uint16_t addr16;
    
    ADDRBUS_MODE_OUTPUT;
	for(uint32_t i = startaddr; i < endaddr; i++) {
        addr16 = (uint16_t) (startaddr+i);
        DATABUS_MODE_INPUT;
        val = sram_read(addr16);
        
        wval = val^0x55;
        DATABUS_MODE_OUTPUT;
        sram_write(addr16, wval);

        DATABUS_MODE_INPUT;
        val = sram_read(addr16);
        if (wval != val) {
            printf("error at %04lx: written %02x, read %02x.\r\n", startaddr+i, wval,val);
            stopaddr = startaddr+i;
            break;
        }
        
        wval ^= 0x55;
        DATABUS_MODE_OUTPUT;
        sram_write(addr16, wval);
	}
    return stopaddr;
}

uint16_t transfer_to_sram(const uint8_t arr[], uint16_t startaddr, uint32_t size) {
    printf("Transferring %luk bytes data to SRAM...\r\n",size/1024);
    
    ADDRBUS_MODE_OUTPUT;
    DATABUS_MODE_OUTPUT;
	for(uint32_t i = 0; i < size; i++) {
        sram_write((startaddr + (uint16_t) i), arr[i]);
    }
    
    // verify
    uint8_t val;
    uint16_t errcount = 0;
    DATABUS_MODE_INPUT;
	for(uint32_t i = 0; i < size; i++) {
        val = sram_read( startaddr + (uint16_t) i );
        if (arr[i] != val) {
            errcount += 1;
        }
    }
    if ( errcount == 0 ) {
        printf("transfer and verify done.\r\n");
    } else {
        printf("%u errors detected.\r\n", errcount);
    }
    return errcount;
}

void NCO1_init(void){
    // NCO1 pin
    RA3PPS = 0x3F;  //RA3->NCO1:NCO1;
    pinanalog(A3, DISABLE); //ANSELA3 = 0;	// Disable analog function
    pinmode(A3, OUTPUT);
    
    //NPWS 1_clk; NCKS HFINTOSC; 
    // (0<<5 | 0x1 ) NCO output is active for 1 input clock periods, Clock source HFINTOSC
    NCO1CLK = 0x1;
    //NCOACC 0x0; 
    NCO1ACCU = 0x0;
    //NCOACC 0x0; 
    NCO1ACCH = 0x0;
    //NCOACC 0x0; 
    NCO1ACCL = 0x0;
    // NCO1INC = (unsigned int)(CLK_6502_FREQ / 30.5175781);
    // 1MHz --> 0x008000
    //NCOINC 0; 
    NCO1INCU = 0x0;
    //NCOINC 128; 
    NCO1INCH = 0x80;
    //NCOINC 0; 
    NCO1INCL = 0x0;
    
    //NEN enabled; NPOL active_hi; NPFM FDC_mode; 
    NCO1CON = 0x80;
}

/*
void __interrupt(irq(NCO1),base(8)) NCO1_ISR()
{
   // Clear the NCO interrupt flag
    PIR6bits.NCO1IF = 0;
}

bool NCO1_GetOutputStatus(void) 
{
	return (NCO1CONbits.OUT);
}
*/

void  Interrupt_init(void)
{
    INTCON0bits.IPEN = 1; // interrupt priorities are enabled

    bool state = (unsigned char) GlobalInterruptHigh; // backup
    GlobalInterruptHigh = DISABLE;
    IVTLOCK = 0x55;
    IVTLOCK = 0xAA;
    IVTLOCKbits.IVTLOCKED = 0x00; // unlock IVT

    IVTBASEU = 0;
    IVTBASEH = 0;
    IVTBASEL = 8;

    IVTLOCK = 0x55;
    IVTLOCK = 0xAA;
    IVTLOCKbits.IVTLOCKED = 0x01; // lock IVT

    GlobalInterruptHigh = state;
    // Assign peripheral interrupt priority vectors
    IPR9bits.U3RXIP = 1; //UART3 Receive Interrupt Priority

}

void __interrupt(irq(default),base(8)) Default_ISR()
{
}

/* UART3 ISR is defined in uart3.c */

void system_init(void) {
    
    // HFINTOSC Clock initialize
    // Set the CLOCK CONTROL module to the options selected in the user interface.
    OSCCON1 = (0 << _OSCCON1_NDIV_POSN)   // NDIV 1
        | (6 << _OSCCON1_NOSC_POSN);  // NOSC HFINTOSC    
    OSCFRQ = (8 << _OSCFRQ_HFFRQ_POSN);  // HFFRQ 64_MHz
    
    pins_default();
    
    W65C02_interface_init();
    NCO1_init();
    UART3_init();
    
    //CLC_init();
    Interrupt_init();
}

void UART_ProcessInput(void) {
    char c;
    
    c = UART3_Read();
    if ( isprint(c) ) {
        putch(c);
    } else {
        printf("<%d>", c);
        putch(c);
        if ( c == '\r' ) {
            putch('\n');
        }
    }
}

uint16_t load_rom(const uint8_t rom[], const uint16_t dst_addr, const uint16_t bytesize) {
    // returns the number of read/write check failures
    uint16_t errors = 0;
    uint8_t onebyte;
    for(uint16_t bytecount = 0; bytecount < bytesize; ++bytecount) {
        sram_write(dst_addr + bytecount, rom[bytecount]);
    }
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

int main(void) {
    uint16_t errcount;
    
    system_init();
    
    // Enable the Global High Interrupts 
    GlobalInterruptHigh = ENABLE; 
    
    printf("\e[H\e[2JHello, System initialized. UART3 enabled.\r\n");
    printf("CLC init is skipped.\r\n");

    set_busmode_DMA();
    printf("loading rom... ");
    errcount = load_rom(ROM, ROM_TOP, ROM_SIZE);
    if (! errcount) {
        printf("\r%u kb loaded from %04x with no errors.\r\n", ROM_SIZE/1024, ROM_TOP);
    } else {
        printf("\r%u errors occurred during loading rom.\r\n", errcount);
    }
    printf("Now this program simply echos back.\r\n");

    while(1)
    {
        if ( UART3_IsRxReady() ) {
            UART_ProcessInput();
        }
    }
}


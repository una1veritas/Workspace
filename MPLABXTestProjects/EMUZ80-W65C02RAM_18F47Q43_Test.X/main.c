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

// PORT name definition
#define ADDRBUS_HIGH    D
#define ADDRBUS_LOW     B
#define DATABUS         C

// PIN name definition
#define W65C02_CLK  A3
#define W65C02_RWB  A4
#define W65C02_RESB E2
// assigned CLC outputs
#define W65C02_RDY  A0
#define W65C02_BE   E0
#define SRAM_WE     A2
#define SRAM_OE     A5
// _CS1 and CS2 are tied with GND and VCC, resp.
// A16 of SRAM is tied with GND

//6551 style
#define ACIA_DAT    0xB098
#define ACIA_STA    0xB099 
#define ACIA_CMD    0xB09A
//#define ACIA_CTL    0xB09B

#define ACIA_STA_PERR   (1<<0)
#define ACIA_STA_FERR   (1<<1)
#define ACIA_STA_OVRUN  (1<<2)
#define ACIA_STA_RDRF   (1<<3)
#define ACIA_STA_TDRE   (1<<4)

/*
 * ACIA (6551) Status Register
 * bits
 * 0 -- parity err, 1 -- framing err, 2 -- overrun,
 * 3 -- Receive Data Reg. full
 * 4 -- Transmit Data Reg. empty
 * 5, 6, 7 --- /DCD, /DSR, /IRQ
 */
//6850 style
// CS0, CS1, /CS2, RS, R/W
// xxx,RS=1,R/W -> Rx/Tx
// xxx,RS=0,R/W -> Status/Control
// from bit 0 to 7 ... RDRF, TDRE, /DTD, /CTS, FE, OVRN, /IRQ

#define ROM_TOP         0xC000		// ROM TOP Address
#define ROM_SIZE        0x4000		// 16K bytes
//6502 ROM equivalent, see end of this file
//extern const unsigned char rom_EhBASIC[];
extern const unsigned char rom_ehbasic_acia[];
#define ROM rom_ehbasic_acia

void system_init(void) {
    // PIC18 System Clock
    // HFINTOSC Clock initialize
    // Set the CLOCK CONTROL module to the options selected in the user interface.
    OSCCON1 = (0 << _OSCCON1_NDIV_POSN)   // NDIV 1
        | (6 << _OSCCON1_NOSC_POSN);  // NOSC HFINTOSC    
    OSCFRQ = (8 << _OSCFRQ_HFFRQ_POSN);  // HFFRQ 64_MHz
    
    pins_default();
    
    W65C02_interface_init();
    NCO1_init();    // 65C02 Clock
    UART3_init();
    printf("system clock, W65C02 clock, UART3 init finished.\r\n");
    
    CLC_init();
    printf("CLC init finished.\r\n");
    Interrupt_init();
}

void W65C02_interface_init() {
    // specific pins & ports
    // bus and bidirectional port/pins are 
    // set input until bus request is acknowledged.
    
    // /RESET (RE2) PIC output pin 
	pinmode(W65C02_RESB, OUTPUT);    //TRISE2 = 0;		// Set as output
	pinwrite(W65C02_RESB, LOW);      //LATE2 = 0;		// /Reset = Low

	// BE (RE0) PIC output pin
	pinmode(W65C02_BE, OUTPUT); //TRISE0 = 0;		// Set as output
	pinwrite(W65C02_BE, LOW); //LATE0 = 0;		// BE = Low, set 6502 address/data/RWB ports/pin to be High-Z

    // memory and I/O buses */
	// Address bus A15-A8 pin
    // setting the whole 8 bits on port
	portmode(ADDRBUS_HIGH, PORT_INPUT); //TRISD 
	portwrite(ADDRBUS_HIGH, 0); //LATD = 0x00;

	// Address bus A7-A0 pin
	portmode(ADDRBUS_LOW, PORT_INPUT); //TRISB
	portwrite(ADDRBUS_LOW, 0); //LATB = 0x00;

	// Data bus D7-D0 pin
	portmode(DATABUS, PORT_INPUT); //TRISC
	portwrite(DATABUS, 0); //LATC = 0x00;

	// R/W (RA4) PIC CLC input pin
	pinmodewpu(W65C02_RWB,INPUT); //WPUA4 = 1;		// set weak pull up and mode input since this is input for CLC
	//TRISA4 = 1;		// Set as input

    /* 
     * pins controlled by CLC output
     * PPS are set in CLC_init
     * 
     */
	// RDY/_HALT (RA0) output pin Low = Halt
	//RA0PPS = 0x00;	// LATA0 -> RA0
	pinwrite(W65C02_RDY, HIGH); //LATA0 = 1;		// RDY = High
	pinmode(W65C02_RDY, OUTPUT); //TRISA0 = 0;		// Set as output

	// /WE (RA2) output pin
	//RA2PPS = 0x00;	// LATA2 -> RA2
	pinwrite(SRAM_WE, HIGH); //LATA2 = 1;		// /WE = High
	pinmode(SRAM_WE, OUTPUT); //TRISA2 = 0;		// Set as output

	// /OE (RA5) output pin
	//RA5PPS = 0x00;	// LATA5 -> RA5
	pinwrite(SRAM_OE, HIGH); //LATA5 = 1;		// /OE = High
	pinmode(SRAM_OE, OUTPUT); //TRISA5 = 0;		// Set as output
}

void NCO1_init(void){
    // NCO1 pin
    RA3PPS = 0x3F;  //RA3->NCO1:NCO1;
    pinanalogmode(A3, DISABLE); //ANSELA3 = 0;	// Disable analog function
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

// PIC becomes the bus master
void set_busmode_DMA() {
    pinwrite(W65C02_BE, LOW);
    // disable CLC output for SRAM_WE (RA2) and SRAM_OE(RA5)
    // by assign LATAx 0x00 by PPS
    PPS(SRAM_WE) = 0x00;
    PPS(SRAM_OE) = 0x00; 

    portmode(ADDRBUS_LOW, OUTPUT);
    portmode(ADDRBUS_HIGH, OUTPUT);
    portmodewpu(DATABUS, INPUT);
}

// W65C02 becomes the bus master
void set_busmode_W65C02() {
    // Address bus A15-A8 pin
    portmodewpu(ADDRBUS_HIGH, INPUT);
	// Address bus A7-A0 pin
    portmodewpu(ADDRBUS_LOW, INPUT);
	// Data bus D7-D0 pin
    portmodewpu(DATABUS, INPUT);
    
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> RDY}
    
    pinwrite(W65C02_BE, LOW);
}

inline void set_addr_bus(const uint16_t addr) {
    //ADDRBUS_MODE_OUTPUT;
    portwrite(ADDRBUS_HIGH, ((uint8_t *)&addr)[1]); //*(((uint8_t *) & addr) + 1); //LATD = ab.h;
    portwrite(ADDRBUS_LOW, (uint8_t) addr); //LATB = ab.l;
}

uint8_t sram_read(const uint16_t addr) {
    uint8_t data;
    pinmodewpu(DATABUS, INPUT);
    pinwrite(SRAM_OE, LOW); //LATA5 = 0;		// _OE=0
    set_addr_bus(addr);
    NOP(); //__delay_us(1);
    data = portread(DATABUS);
	pinwrite(SRAM_OE, HIGH); //LATA5 = 1;		// _OE=1
    return data;
}

void sram_write(const uint16_t addr, const uint8_t data) {
    pinmodewpu(DATABUS, OUTPUT);
    set_addr_bus(addr);
    portwrite(DATABUS, data);
    pinwrite(SRAM_WE, LOW); //LATA2 = 0;		// /WE=0
    NOP(); //__delay_us(1);
    pinwrite(SRAM_WE, HIGH); //LATA2 = 1;		// /WE=1    
}

const static uint8_t rndata[256] = {
    0x2d, 0xa8, 0x15, 0x58, 0xcd, 0xb0, 0x1c, 0xae, 0xe7, 0x63, 0x11, 0xd2, 0x6c, 0x69, 0xec, 0x90, 
    0x4c, 0x41, 0x2a, 0xc1, 0x70, 0x2b, 0x76, 0x55, 0x8b, 0xcc, 0x46, 0xc8, 0x94, 0xed, 0x13, 0x44, 
    0x68, 0xb6, 0xdd, 0x67, 0xa9, 0x33, 0x03, 0x38, 0xea, 0x81, 0x0a, 0x34, 0x65, 0x8e, 0x4e, 0x5c, 
    0xf2, 0xee, 0x7c, 0xbb, 0x72, 0x39, 0x17, 0x27, 0x51, 0x3b, 0x79, 0xa3, 0x26, 0x6a, 0x59, 0xe3, 
    0x6e, 0x1a, 0x95, 0x7e, 0xb2, 0xc5, 0x0c, 0x3c, 0xa4, 0x57, 0x8a, 0xdc, 0xf6, 0x8c, 0xf1, 0x35, 
    0x49, 0x50, 0xef, 0xda, 0x2c, 0x02, 0x97, 0x31, 0x16, 0x75, 0xce, 0x6f, 0x2f, 0xa5, 0x74, 0xdb, 
    0xb4, 0xd3, 0x6d, 0x77, 0x92, 0xff, 0x4d, 0x85, 0xc2, 0x73, 0x04, 0x1f, 0xaa, 0xb5, 0x61, 0x14, 
    0xfe, 0x20, 0x9f, 0x8d, 0x96, 0x87, 0x9a, 0x54, 0x52, 0x5d, 0x00, 0x32, 0xcb, 0x9c, 0x5b, 0x18, 
    0x07, 0xe5, 0x7b, 0xd6, 0x4a, 0x5e, 0x4f, 0xac, 0xa6, 0xab, 0xf9, 0xc3, 0x21, 0xf3, 0x83, 0xf8, 
    0x09, 0x3e, 0xe9, 0xbe, 0x25, 0x2e, 0x91, 0x9d, 0xcf, 0x62, 0xd8, 0x0f, 0x1b, 0x06, 0x3a, 0xc0, 
    0x5f, 0xc6, 0x60, 0x29, 0xbd, 0x08, 0x40, 0x30, 0xfb, 0x99, 0xc4, 0x89, 0xe1, 0xf0, 0xd5, 0x3f, 
    0xb8, 0x9e, 0xd7, 0x88, 0x7d, 0x93, 0xaf, 0x3d, 0xe6, 0x4b, 0x9b, 0x71, 0xfa, 0xf7, 0xf5, 0x7a, 
    0xc7, 0xc9, 0xad, 0x48, 0x36, 0x0b, 0x84, 0xb9, 0xfc, 0xe2, 0x19, 0xbc, 0x0d, 0x47, 0xdf, 0x12, 
    0x01, 0x82, 0xd1, 0xb3, 0xa1, 0x98, 0x6b, 0xde, 0x28, 0xba, 0xe4, 0xa2, 0x78, 0x05, 0x22, 0x86, 
    0xd0, 0x24, 0x5a, 0x64, 0x1e, 0xb7, 0x37, 0x8f, 0xfd, 0x10, 0x1d, 0x45, 0xf4, 0x56, 0xd9, 0x66, 
    0xa0, 0x23, 0x53, 0x80, 0xbf, 0xe8, 0x0e, 0xeb, 0xd4, 0xb1, 0x42, 0x7f, 0xe0, 0xca, 0x43, 0xa7, 
};

uint32_t memory_check(uint32_t startaddr, uint32_t endaddr) {
    uint32_t stopaddr = endaddr;
    uint8_t val;
    uint16_t addr16;
    uint8_t buf[256];
    const char testbytes[] = "All work and no play makes Jack a dull boy.\r]n";
    unsigned int len = strlen(testbytes);
    
    set_busmode_DMA();
    
	for(uint32_t baseaddr = startaddr; baseaddr < endaddr; baseaddr += 256) {
        for(uint32_t i = 0; i < 256; ++i) {
            addr16 = (uint16_t) (startaddr + rndata[i]);
            buf[(uint16_t)i] = sram_read(addr16);
            sram_write(addr16, testbytes[i % len]);
        }

        for(uint32_t i = 0; i < 256; ++i) {
            addr16 = (uint16_t) (startaddr + rndata[i]);
            val = sram_read(addr16);
            sram_write(addr16, buf[i]);
            if (val != testbytes[i % len]) {
                printf("error at %04lx: written %02x, read %02x.\r\n", startaddr+i, testbytes[i % len], val);
                stopaddr = startaddr+i;
                return stopaddr;                
            }
        }
	}
    return stopaddr;
}

uint16_t transfer_to_sram(const uint8_t arr[], uint32_t startaddr, uint32_t size) {
    printf("Transferring %luk bytes data to SRAM...\r\n",size/1024);

    set_busmode_DMA();
	for(uint32_t i = 0; i < size; i++) {
        sram_write( (uint16_t) (startaddr + i), arr[i]);
    }
    
    // verify
    uint8_t val;
    uint16_t errcount = 0;
	for(uint32_t i = 0; i < size; i++) {
        val = sram_read( (uint16_t) (startaddr +  i) );
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
/*
int main(void) {
    
    system_init();
    
    // Enable the Global High Interrupts 
    GlobalInterruptHigh = ENABLE; 
    printf("GIE Interrupt High enabled.\r\n");
    
    printf("Type keys then echos and outs by LED_out");
    LED_out(0);
    while(1)
    {
        if ( UART3_IsRxReady() ) {
            UART_ProcessInput();
        }
    }
}
*/

// main routine
void main(void) {
    union {
        uint16_t u16;
        uint8_t  u8[2];
        struct {
            uint8_t low8, hi8;
        };
    } addr;
    
	system_init();
    
    // Enable the Global High Interrupts 
    GlobalInterruptHigh = ENABLE; 
    
    
    printf("\e[H\e[2JHello, System initialized. UART3 enabled.\r\n");
    
    set_busmode_DMA();
    uint32_t stopaddr = memory_check(0, 0x10000);
    printf("read/write check... %luk bytes succeeded.\r\n", stopaddr/1024);
    transfer_to_sram(ROM, ROM_TOP, ROM_SIZE);
    set_busmode_W65C02();
    
	printf("\r\nMEZ6502RAM %2.3fMHz\r\n",NCO1INC * 30.5175781 / 1000000);

    Interrupt_init();
	
	// 6502 start
    printf("\r\nStarting 65C02CPU.\r\n");
    
	//GIE = 1;			// Global interrupt enable

	pinwrite(W65C02_RESB, HIGH); //LATE2 = 1;			// Release reset

	while(1){
		while(CLC5OUT); //RDY == 1  // waiting for $Bxxx is on address bus.
		addr.u8[1] = portread(ADDRBUS_HIGH); //PORTD;				// Read address high
		addr.u8[0] = portread(ADDRBUS_LOW); //PORTB;				// Read address low
		//6502 -> PIC IO write cycle
		if ( ! pinread(W65C02_RWB) ) /*(!RA4)*/ {
            // 6502 Write then PIC Read and Out
			if ( addr.u16 == ACIA_DAT ) {
                //putch(PORTC);
                UART3_Write(portread(DATABUS));
            }
			//Release RDY (D-FF reset)
			G3POL = 1;
			G3POL = 0;
		} else {
    		//PIC In and Write then 6502 Read
			portmode(DATABUS, OUTPUT); //TRISC = 0x00;				// Set Data Bus as output
			if ( addr.u16 == ACIA_STA ) {
                portwrite(DATABUS, (UART3_IsRxReady() ? ACIA_STA_RDRF : 0 ) | (UART3_IsTxReady() ? ACIA_STA_TDRE : 0 ));
			} else if ( addr.u16 == ACIA_DAT ) {
                if (UART3_IsRxReady()) {
                    portwrite(DATABUS, UART3_Read());
                } else {
                    portwrite(DATABUS, 0x00);
                }
			} else {
				portwrite(DATABUS, 0xff);			// Invalid address
            }
			// Detect CLK falling edge
			while( pinread(W65C02_CLK) ); //RA3);
			//Release RDY (D-FF reset)
			G3POL = 1;
			portmodewpu(DATABUS, INPUT); //TRISC = 0xff;				// Set Data Bus as input
			G3POL = 0;
		}
	}
}
  
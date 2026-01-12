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

#include <ctype.h>
#include <string.h>

#include "pic18common.h"

#include "system.h"
#include "uart3.h"
/*
    Main application
*/

#define MC68K8_RESET_HALT   E0
#define MC68K8_BR           E1
#define MC68K8_BG           C2
#define MC68K8_DTACK        C5
#define MC68K8_RW           A4
#define MC68K8_AS           C1
#define MC68K8_DS           C0

#define ADBUS       B
#define ABUS_MID    D
#define ABUS_HIGH4  A
#define ALE         E2
#define ALE_OE      C3
#define SRAM_CE     C0
#define SRAM_WE     C4

#define SPI_SS      C7
#define SPI_SCK     B1
#define SPI_MOSI    B0
#define SPI_MISO    C6

#define MC68K8_RESET_OUT    LATE0
#define MC68K8_RESET_MODE   TRISE0
#define MC68K8_BR_OUT       LATE1
#define MC68K8_BR_MODE      TRISE1
#define MC68K8_BG_OUT       LATC2
#define MC68K8_DTACK_OUT    LATC5

#define ALE_OUT             LATE2
#define ALE_OE_OUT          LATC3
#define ALE_MODE            TRISE2
#define ALE_OE_MODE         TRISC3

#define ADBUS_OUT    LATB
#define ADBUS_IN     PORTB
#define ADBUS_MODE   TRISB
#define ADBUS_WPU    WPUB
#define ABUS_MID_OUT    LATD
#define ABUS_MID_IN     PORTD
#define ABUS_MID_MODE   TRISD
#define ABUS_MID_WPU    WPUD
#define ABUS_HIGH4_OUT  LATA
#define ABUS_HIGH4_IN   PORTA
#define ABUS_HIGH4_MODE TRISA
#define ABUS_HIGH4_WPU  WPUA

#define MC68K8_RW_IN        PORTA4
#define MC68K8_RW_MODE      TRISA4

#define _SPI_SS_OUT         LATC7

#define M68K8_AS_IN         PORTC1
#define M68K8_DS_IN         PORTC0
#define M68K8_DS_OUT        LATC0
#define SRAM_CE_OUT         LATC0
#define SRAM_CE_MODE        TRISC0
#define SRAM_WE_OUT         LATC4
#define SRAM_WE_MODE        TRISC4

void mc68k8_interface_setup() {
    // /RESET /HALT (RE0) output pin, starts LOW
	pinmode(MC68K8_RESET_HALT, OUTPUT); // MC68K8_RESET_MODE = OUTPUT;		// Set as output
	pinwrite(MC68K8_RESET_HALT, HIGH); // MC68K8_RESET_OUT  = LOW;		// /Reset = Low

	// /BR (RE1) _BR BUSREQUEST output pin, starts LOW
	pinmode(MC68K8_BR, OUTPUT); // MC68K8_BR_MODE  = OUTPUT;		// Set as output
	pinwrite(MC68K8_BR, HIGH); //MC68K8_BR_OUT   = LOW;		// BR = Low

    // /BG bus grant RC2
    pinmodewpu(MC68K8_BG, INPUT); // TRISC2  = INPUT; WPUC2   = HIGH;
    pinwrite(MC68K8_BG, LOW); //LATC2   = 0;
    
    // /DTACK RC5
    pinmode(MC68K8_DTACK, OUTPUT); 
    //TRISC5  = OUTPUT;
    //WPUC5   = HIGH;
    pinwrite(MC68K8_DTACK, HIGH); //LATC5   = HIGH;
    
	// ALE 74LS373 C (LE) RE2
	pinmode(ALE, OUTPUT); //TRISE2  = OUTPUT;	// Set as input
	pinwrite(ALE, HIGH); //LATE2   = HIGH;     // pass thru
    
	// ALE_OE 74LS373 /LAT_OE (/OC) RC3
    pinmode(ALE_OE, OUTPUT); //TRISC3  = OUTPUT;
    pinwrite(ALE_OE, HIGH);  //LATC3   = HIGH;     // output disable

	// D0 - D7/A0 - A7 pin
	//LATB   = 0x00;
	portmodewpu(ADBUS, INPUT); //TRISB  = 0xff;	// Set as input
    //WPUD   = 0xff;  // weak pull up
    portwrite(ADBUS, 0x00);
    
	// Address bus A8 - A15 pin
    // setting the whole 8 bits on port
	//LATD    = 0x00;
	//TRISD   = 0xff; // Set as input
    //WPUD    = 0xff; // weak pull up
    portmodewpu(ABUS_MID, INPUT); 
    portwrite(ABUS_MID, 0x00);
    
    // A16 - A19 (RA0 - RA3, only A19 (RA3) is pull-downed by hardware))
    TRIS(ABUS_HIGH4) |= 0x0f; // RA0 -- RA
    WPU(ABUS_HIGH4) |= 0x07;
    
    // CPU R/W RA4
    pinmodewpu(MC68K8_RW, INPUT); //TRISA4  = INPUT;    
    // AS (RC1), DS (RC0))
    pinmodewpu(MC68K8_AS, INPUT); //TRISC0  = INPUT;
    pinmodewpu(MC68K8_DS, INPUT); //TRISC1  = INPUT;
    
    // SPI_SS/ /LED (RC7)
    pinmode(SPI_SS, OUTPUT); //TRISC7  = OUTPUT;
    pinwrite(SPI_SS, HIGH);  //LATC7   = HIGH;
    
}

void UART_echoCharacters(void)
{
    uint8_t rxbyte;  
    rxbyte = UART3_Read();
    if ( isprint(rxbyte) ) {
        UART3_Write(rxbyte);
    } else {
        printf("\r\n<%02x>\r\n", rxbyte);
    }
}

void busmode_DMA(void) {
    // Ensure BUSREQ ACKed.
    SRAM_CE_MODE = OUTPUT;
    SRAM_WE_MODE = OUTPUT;
    ADBUS_MODE   = PORT_OUTPUT;
    ABUS_MID_MODE = PORT_OUTPUT;
    ABUS_MID_WPU = PORT_DISABLE;
    ABUS_HIGH4_MODE = (ABUS_HIGH4_MODE & 0xf0) | (PORT_OUTPUT & 0x0f);
    ABUS_HIGH4_WPU &= 0xf0;
    ALE_OE_OUT = LOW; // enable
}
/*
void setup_busmode_6502() {
    // Address bus A15-A8 pin
	ANSELD = 0x00;	// Disable analog function
	WPUD = 0xff;	// Week pull up
	TRISD = 0xff;	// Set as input

	// Address bus A7-A0 pin
	ANSELB = 0x00;	// Disable analog function
	WPUB =  0xff;	// Week pull up
	TRISB = 0xff;	// Set as input

	// Data bus D7-D0 pin
	ANSELC = 0x00;	// Disable analog function
	WPUC =  0xff;	// Week pull up
	TRISC = 0xff;	// Set as input(default)
}

inline void set_addr_bus(const uint16_t addr) {
    //ADDRBUS_MODE_OUTPUT;
    ADDRBUS_HIGH_WR = ((uint8_t *)&addr)[1]; // *(((uint8_t *) & addr) + 1); //LATD = ab.h;
    ADDRBUS_LOW_WR  = (uint8_t) addr; //LATB = ab.l;
}

uint8_t sram_read(const uint16_t addr) {
    uint8_t data;
    set_addr_bus(addr);
    SRAM_OE = LOW; //LATA5 = 0;		// _OE=0
    NOP(); //__delay_us(1);
    data = DATABUS_RD;
	SRAM_OE = HIGH; //LATA5 = 1;		// _OE=1
    return data;
}

void sram_write(const uint16_t addr, const uint8_t data) {
    set_addr_bus(addr);
    DATABUS_WR = data;
    SRAM_WE = LOW; //LATA2 = 0;		// /WE=0
    NOP(); //__delay_us(1);
    SRAM_WE = HIGH; //LATA2 = 1;		// /WE=1    
}
*/
void set_addr_bus(uint32_t addr20) {
    ADBUS_MODE = PORT_OUTPUT;
    ADBUS_WPU = PORT_DISABLE;
    ALE_OUT = HIGH; // latch enable
    ADBUS_OUT = addr20 & 0xff;
    ALE_OUT = LOW;
    addr20 >>= 8;
    ABUS_MID_OUT = addr20 & 0xff;
    // the highest 4 bits
    addr20 >>= 8;
    ABUS_HIGH4_OUT = (ABUS_HIGH4_OUT & 0xf0) | (addr20 & 0x0f); 
}

uint8_t sram_read(uint32_t addr) {
    uint8_t data;
    SRAM_WE_OUT = HIGH;
    set_addr_bus(addr);
    SRAM_CE_OUT = LOW;
    ADBUS_MODE = PORT_INPUT;
    ADBUS_WPU = PORT_ENABLE;
    NOP();
    data = ADBUS_IN;
    SRAM_CE_OUT = HIGH;
    return data;
}

void sram_write(uint32_t addr, uint8_t data) {
    SRAM_CE_OUT = LOW;
    set_addr_bus(addr);
    ADBUS_MODE = PORT_OUTPUT;
    ADBUS_WPU = PORT_DISABLE;
    ADBUS_OUT = data;
    SRAM_WE_OUT = LOW;
    NOP();
    SRAM_WE_OUT = HIGH;
    SRAM_CE_OUT = HIGH;
    return;    
}

#define BUFFSIZE 256
void sram_check() {
    static const char proverbs[] = "All work and no play makes Jack a dull boy\r\n"
    "Give to Caesar what belongs to Caesar, and to God what belongs to God\r\n";
    uint32_t len = strlen(proverbs);
    const uint32_t bsize = BUFFSIZE;
    uint8_t rval[BUFFSIZE];
    uint32_t errcount = 0;
    printf("start checking with data:\r\n%s\r\n",proverbs);
    for(uint32_t addr = 0; addr < 0x80000; addr += bsize) {
        for(uint32_t i = 0; i < BUFFSIZE; ++i) {
            rval[i] = sram_read(addr + i);
            sram_write(addr + i, proverbs[(addr+i) % len]);
        }
        for(uint32_t i = 0; i < BUFFSIZE; ++i) {
            if ( proverbs[(addr+i) % len] == sram_read(addr+i) ) {
                sram_write(addr+i, rval[i]);
            } else {
                errcount += 1;
            }
        }
        if ( addr > 0 && (addr & ((bsize<<6)-1)) == 0 ) {
            printf("%05lx -- %05lx [% 6ld] ", addr - (bsize<<6), addr, errcount);
            for(uint32_t i = 0; i < 16; ++i) {
                printf("%02x ", sram_read(addr - (bsize<<6) +i));
            }
            printf("\r\n");
        }
    }
    printf("Finished. \r\n");
}

int main(void)
{
    uint32_t addr32 = 0;
    uint8_t rxbyte, prevbyte = 0;
    
    SYSTEM_Initialize();
    mc68k8_interface_setup();
    
    printf("\e[H\e[2J");
    printf("Hello World!\r\n");
    printf("Type characters in the terminal, to have them echoed back ...\r\n");

    INTERRUPT_GlobalInterruptEnable(); //INTCON0bits.GIE = 1
    
    pinwrite(MC68K8_BR, LOW);
    while (pinread(MC68K8_BG) == HIGH);
    if (pinread(MC68K8_BG) == LOW) {
        printf("MC68K8 Bus Request Granted.\r\n");

        busmode_DMA();
        sram_check();
        for(;;) {}
        
        for(;;) {
            while ( UART3_IsRxReady() ) {
                rxbyte = UART3_Read();
                sram_write(addr32++, rxbyte);
                if (rxbyte == 0x0d && prevbyte != 0x0a) {
                    UART3_Write(0x0d);
                    UART3_Write(0x0a);
                } else {
                    UART3_Write(rxbyte);
                }
                prevbyte = rxbyte;            
            }

            if (addr32 > 8) {
                printf("Buffer limit reached. reading memory...\r\n");
                for (uint32_t i = 0; i < addr32; ++i) {
                    printf("%lu %c ", i, sram_read(i));
                }
                printf("\r\n");
                addr32 = 0;
            }

        }
    }
    return 0;
}
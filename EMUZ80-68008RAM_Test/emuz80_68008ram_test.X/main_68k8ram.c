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
#define MC68K8_RESET_OUT    LATE0
#define MC68K8_RESET_MODE   TRISE0
#define MC68K8_BR_OUT       LATE1
#define MC68K8_BR_MODE      TRISE1
#define MC68K8_BG_OUT       LATC2
#define MC68K8_DTACK_OUT    LATC5

#define ALE                 LATE2
#define ALE_OE              LATC3
#define ALE_MODE            TRISE2
#define ALE_OE_MODE         TRISC3

#define ADBUS_OUT    LATB
#define ADBUS_IN     PORTB
#define ADBUS_MODE   TRISB
#define ADBUS_WPU   WPUB
#define ABUS_MID_OUT    LATD
#define ABUS_MID_IN     PORTD
#define ABUS_MID_MODE   TRISD
#define ABUS_HIGH4_OUT  LATA
#define ABUS_HIGH4_IN   PORTA
#define ABUS_HIGH4_MODE TRISA

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

void board_pin_setup() {
        // /RESET /HALT (RE0) output pin, starts LOW
	MC68K8_RESET_MODE = OUTPUT;		// Set as output
	MC68K8_RESET_OUT   = LOW;		// /Reset = Low

	// /BR (RE1) _BR BUSREQUEST output pin, starts LOW
	MC68K8_BR_MODE  = OUTPUT;		// Set as output
	MC68K8_BR_OUT   = LOW;		// BR = Low

    // /BG bus grant RC2
    LATC2   = 0;
    TRISC2  = INPUT;
    WPUC2   = HIGH;

    // /DTACK RC5
    LATC5   = HIGH;
    TRISC5  = OUTPUT;
    WPUC5   = HIGH;
    
	// ALE 74LS373 C (LE) RE2
	LATE2   = HIGH;     // pass thru
	TRISE2  = OUTPUT;	// Set as input
    
	// ALE_OE 74LS373 /LAT_OE (/OC) RC3
    LATC3   = HIGH;     // output disable
    TRISC3  = OUTPUT;

	// D0 - D7/A0 - A7 pin
	LATB   = 0x00;
	TRISB  = 0xff;	// Set as input
    WPUD   = 0xff;  // weak pull up

	// Address bus A8 - A15 pin
    // setting the whole 8 bits on port
	LATD    = 0x00;
	TRISD   = 0xff; // Set as input
    WPUD    = 0xff; // weak pull up
    
    // A16 - A19 (RA0 - RA4, only A19 (RA3) is pull-downed by hardware))
    LATA    = 0x00;
    TRISA   |= 0x0f; // OUTPUT
    
    // CPU R/W RA4
    TRISA4  = INPUT;
    
    
    // SPI_SS/ /LED (RC7)
    LATC7   = HIGH;
    TRISC7  = OUTPUT;
    
    // AS (RC1), DS (RC0))
    TRISC0  = INPUT;
    TRISC1  = INPUT;
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
    ABUS_HIGH4_MODE = (ABUS_HIGH4_MODE & 0xf0) | PORT_OUTPUT & 0x0f;
    ALE_OE_MODE = OUTPUT; 
    ALE_MODE = OUTPUT; 
    ALE_OE = LOW; // enable
    ALE = LOW;    // disable
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
    ALE = HIGH; // latch enable
    ADBUS_OUT = addr20 & 0xff;
    ALE = LOW;
    ABUS_MID_OUT = (addr20 >> 8) & 0xff;

    // the highest 4 bits are always cleared
    ABUS_HIGH4_OUT &= 0xf0; 
}

uint8_t sram_read(uint32_t addr) {
    uint8_t data;
    SRAM_WE_OUT = HIGH;
    set_addr_bus(addr);
    SRAM_CE_OUT = LOW;
    ADBUS_MODE = PORT_INPUT;
    NOP();
    data = ADBUS_IN;
    SRAM_CE_OUT = HIGH;
    return data;
}

void sram_write(uint32_t addr, uint8_t data) {
    SRAM_CE_OUT = LOW;
    set_addr_bus(addr);
    ADBUS_MODE = PORT_OUTPUT;
    ADBUS_OUT = data;
    SRAM_WE_OUT = LOW;
    NOP();
    SRAM_WE_OUT = HIGH;
    SRAM_CE_OUT = HIGH;
    return;    
}

int main(void)
{
    uint32_t addr32 = 0;
    uint8_t rxbyte;
    
    SYSTEM_Initialize();
    board_pin_setup();
    printf("\e[H\e[2J");
    printf("Hello World!\r\n");
    printf("Type characters in the terminal, to have them echoed back ...\r\n");

    INTERRUPT_GlobalInterruptEnable(); //INTCON0bits.GIE = 1
    
    busmode_DMA();
    
    for(;;) {
        //printf("\r\n");
        sram_write(13,rxbyte);
        printf("%02x ", sram_read(13));
        rxbyte++;
        printf("\r\n");
        __delay_ms(2000);
    }
    
    for(;;) {
        while ( UART3_IsRxReady() ) {
            rxbyte = UART3_Read();
            sram_write(addr32++, rxbyte);
            UART3_Write(rxbyte);
        }
        if (addr32 > 8) {
            for (uint32_t i = 0; i < addr32; ++i) {
                printf("\r\n%lu <%02x>\r\n", i, sram_read(i));
            }
            addr32 = 0;
        }
    }
}
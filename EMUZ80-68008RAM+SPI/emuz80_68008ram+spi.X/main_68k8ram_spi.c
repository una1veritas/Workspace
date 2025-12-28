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

#include "system.h"
#include "uart3.h"
/*
    Main application
*/

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
    MC68K8_RW_MODE = OUTPUT;
    ADBUS_MODE = PORT_INPUT;
    ABUS_MID_MODE = PORT_INPUT;
    ABUS_HIGH4_MODE |= PORT_INPUT & 0x0f;
    ALE_OE = LOW; // enable
    ALE = LOW;    // disable
}

void set_addr_bus(uint32_t addr20) {
    ADBUS_MODE = PORT_OUTPUT;
    ALE = HIGH; // latch enable
    ADBUS_OUT = addr20 & 0xff;
    ALE = LOW;
    addr20 >>= 8;
    ABUS_MID_OUT = addr20 & 0xff;
    addr20 >>= 8;
    ABUS_HIGH4_OUT = (ABUS_HIGH4_OUT & 0xF0) | addr20 & 0x0f;
}

void set_data_bus(const uint8_t data) {
    ADBUS_WPU = 0x00;
    ADBUS_MODE = PORT_OUTPUT;
    ADBUS_OUT = data;
}

uint8_t get_data_bus(void) {
    uint8_t data;
    ALE = LOW;
    ADBUS_MODE = PORT_INPUT;
    ADBUS_WPU = 0xff;
    data = ADBUS_IN;
    return data;
}

uint8_t sram_read(uint32_t addr) {
    uint8_t data;
    set_addr_bus(addr);
    SRAM_WE_OUT = HIGH;
    SRAM_CE_OUT = LOW;
    asm("nop");
    data = get_data_bus();
    SRAM_CE_OUT = HIGH;
    return data;
}

void sram_write(uint32_t addr, uint8_t data) {
    set_addr_bus(addr);
    set_data_bus(data);
    SRAM_WE_OUT = LOW;
    SRAM_CE_OUT = LOW;
    asm("nop");
    SRAM_CE_OUT = HIGH;
    SRAM_WE_OUT = HIGH;
    return;    
}

int main(void)
{
    uint32_t addr32 = 0;
    uint8_t rxbyte;
    
    SYSTEM_Initialize();

    printf("\e[H\e[2J");
    printf("Hello World!\r\n");
    printf("Type characters in the terminal, to have them echoed back ...\r\n");

    INTERRUPT_GlobalInterruptEnable(); //INTCON0bits.GIE = 1
    
    busmode_DMA();
    
    for (uint32_t addr = 0; addr < 64; ++addr) {
        if ( (addr & 0x0f) == 0 ) {
            printf("\r\n");
        } 
        printf("%02x ", sram_read(addr));
    }
    printf("\r\n");
    for(;;);
    
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
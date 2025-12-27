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

int main(void)
{
    SYSTEM_Initialize();

    printf("\e[H\e[2J");
    printf("Hello World!\r\n");
    printf("Type characters in the terminal, to have them echoed back ...\r\n");

    INTERRUPT_GlobalInterruptEnable(); //INTCON0bits.GIE = 1
    
    for(;;) {
        while ( UART3_IsRxReady() ) {
            LATC7 = LOW;
            UART_echoCharacters();
            LATC7 = HIGH;
        }
    }
}
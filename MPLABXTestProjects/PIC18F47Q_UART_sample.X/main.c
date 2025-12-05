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
    
    printf("\r\nHello World!\r\n");
    printf("Type characters in the terminal, to have them echoed back ...\r\n");

    //UART3_RxCompleteCallbackRegister(&UART_echoCharacters);
    INTERRUPT_GlobalInterruptEnable();
    
    while(1) {
        while ( UART3_IsRxReady() ) {
            UART_echoCharacters();
        }
    }
}
/**
 * Interrupt Manager Generated Driver File.
 *
 * @file interrupt.c
 * 
 * @ingroup interrupt 
 * 
 * @brief This file contains the API prototypes for the Interrupt Manager driver.
 * 
 * @version Interrupt Manager Driver Version 2.0.4
*/

#include "system.h"
#include "uart3.h"

void  INTERRUPT_Initialize (void) {
    // Disable Interrupt Priority Vectors (16CXXX Compatibility Mode)
    INTCON0bits.IPEN = 0;
}

void __interrupt() INTERRUPT_InterruptManager (void)
{
    // interrupt handler
    if(PIE9bits.U3RXIE == 1 && PIR9bits.U3RXIF == 1)
    {
        UART3_RxInterruptHandler();
    }
}

/**
 End of File
*/
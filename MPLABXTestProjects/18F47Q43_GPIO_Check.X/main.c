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

/*
� [2026] Microchip Technology Inc. and its subsidiaries.

    Subject to your compliance with these terms, you may use Microchip 
    software and any derivatives exclusively with Microchip products. 
    You are responsible for complying with 3rd party license terms  
    applicable to your use of 3rd party software (including open source  
    software) that may accompany Microchip software. SOFTWARE IS ?AS IS.? 
    NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS 
    SOFTWARE, INCLUDING ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT,  
    MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT 
    WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL, PUNITIVE, 
    INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY 
    KIND WHATSOEVER RELATED TO THE SOFTWARE, HOWEVER CAUSED, EVEN IF 
    MICROCHIP HAS BEEN ADVISED OF THE POSSIBILITY OR THE DAMAGES ARE 
    FORESEEABLE. TO THE FULLEST EXTENT ALLOWED BY LAW, MICROCHIP?S 
    TOTAL LIABILITY ON ALL CLAIMS RELATED TO THE SOFTWARE WILL NOT 
    EXCEED AMOUNT OF FEES, IF ANY, YOU PAID DIRECTLY TO MICROCHIP FOR 
    THIS SOFTWARE.
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

/*
    Main application
*/

#define MAX_COMMAND_LEN         (8U)
#define LINEFEED_CHAR           ((uint8_t)'\n')
#define CARRIAGERETURN_CHAR     ((uint8_t)'\r')

static char command[MAX_COMMAND_LEN];
static uint8_t index = 0;
static uint8_t readMessage;

void LED_out(char c);
void UART_ProcessInput(void);

void LED_out(char c) {
    portwrite(B, c);
}

void UART_ProcessInput(void) {
    char c;
    
    if(UART3_IsRxReady()) {
        c = UART3_Read();
        if ( isprint(c) ) {
            putch(c);
        }
        LED_out(c);
    }
}

void io_init() {
    
    // UART3
    // RX
    pinmode(A7, INPUT);
    pinADmode(A7, DIGITAL);
    U3RXPPS = 0x7; //RA7->UART3:RX3;
    // TX
    pinmode(A6, OUTPUT);
    RA6PPS = 0x26;  //RA6->UART3:TX3;
    
    //NCO1
    RA3PPS = 0x3F;  //RA3->NCO1:NCO1;

    LATB = 0;
    portmode(B, PORT_OUTPUT);
}

void NCO1_init(void){

    //NPWS 1_clk; NCKS HFINTOSC; 
    NCO1CLK = 0x1;
    //NCOACC 0x0; 
    NCO1ACCU = 0x0;
    //NCOACC 0x0; 
    NCO1ACCH = 0x0;
    //NCOACC 0x0; 
    NCO1ACCL = 0x0;
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

void system_init(void) {
    
    // Clock initialize
    // Set the CLOCK CONTROL module to the options selected in the user interface.
    OSCCON1 = (0 << _OSCCON1_NDIV_POSN)   // NDIV 1
        | (6 << _OSCCON1_NOSC_POSN);  // NOSC HFINTOSC    
    OSCFRQ = (8 << _OSCFRQ_HFFRQ_POSN);  // HFFRQ 64_MHz
    
    pins_default();
    io_init();
    NCO1_init();
    
    UART3_init();
    
    INTERRUPT_Initialize();
}

void  INTERRUPT_Initialize (void)
{
    INTCON0bits.IPEN = 1; // interrupt priorities are enabled

    bool state = (unsigned char) GlobalInterruptHigh;
    GlobalInterruptHigh = INT_DISABLE;
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


int main(void) {
    
    system_init();
    
    // Enable the Global High Interrupts 
    GlobalInterruptHigh = INT_ENABLE; 


    
    printf("In the terminal, send 'ON' to turn the LED on, and 'OFF' to turn it off.\r\n");
    printf("Note: commands 'ON' and 'OFF' are case sensitive.\r\n");
    
    LED_out(0);
    while(1)
    {
        UART_ProcessInput();
    }
}
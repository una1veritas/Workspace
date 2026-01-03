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

void UART_ExecuteCommand(char *command);
void UART_ProcessCommand(void);

void IO_LED_SetLow() {
    pinwrite(B0, LOW);
    pinwrite(B1, LOW);
    pinwrite(B2, LOW);
    pinwrite(B3, LOW);
    pinwrite(B4, LOW);
    pinwrite(B5, LOW);
    pinwrite(B6, LOW);
    pinwrite(B7, LOW);
}

void IO_LED_SetHigh() {
    pinwrite(B0, HIGH);
    pinwrite(B1, HIGH);
    pinwrite(B2, HIGH);
    pinwrite(B3, HIGH);
    pinwrite(B4, HIGH);
    pinwrite(B5, HIGH);
    pinwrite(B6, HIGH);
    pinwrite(B7, HIGH);
}

void UART_ExecuteCommand(char *command)
{
    if(strcmp(command, "ON") == 0)
    {
        IO_LED_SetHigh();
        (void)printf("OK, LED ON.\r\n");
    }
    else if (strcmp(command, "OFF") == 0)
    {
        IO_LED_SetLow();
        (void)printf("OK, LED OFF.\r\n");
    }
    else
    {
        (void)printf("Incorrect command.\r\n");
    }
}

void UART_ProcessCommand(void)
{
    if(UART3_IsRxReady())
    {
        readMessage = UART3_Read();
        if ( (readMessage != LINEFEED_CHAR) && (readMessage != CARRIAGERETURN_CHAR) ) 
        {
            command[index++] = readMessage;
            UART3_Write(readMessage);
            if (index > MAX_COMMAND_LEN) 
            {
                (index) = 0;
            }
        }
    
        if (readMessage == CARRIAGERETURN_CHAR) 
        {
            UART3_Write('\r');
             command[index] = '\0';
             index = 0;
             UART_ExecuteCommand(command);
         }
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
 
void __interrupt(irq(NCO1),base(8)) NCO1_ISR()
{
   // Clear the NCO interrupt flag
    PIR6bits.NCO1IF = 0;
}

bool NCO1_GetOutputStatus(void) 
{
	return (NCO1CONbits.OUT);
}

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


int main(void) {
    
    system_init();

    // If using interrupts in PIC18 High/Low Priority Mode you need to enable the Global High and Low Interrupts 
    // If using interrupts in PIC Mid-Range Compatibility Mode you need to enable the Global Interrupts 
    // Use the following macros to: 

    // Enable the Global High Interrupts 
    INTERRUPT_GlobalInterruptHighEnable(); 

    // Disable the Global High Interrupts 
    //INTERRUPT_GlobalInterruptHighDisable(); 

    // Enable the Global Low Interrupts 
    //INTERRUPT_GlobalInterruptLowEnable(); 

    // Disable the Global Low Interrupts 
    //INTERRUPT_GlobalInterruptLowDisable(); 

    
    printf("In the terminal, send 'ON' to turn the LED on, and 'OFF' to turn it off.\r\n");
    printf("Note: commands 'ON' and 'OFF' are case sensitive.\r\n");
    
    UART_ExecuteCommand("OFF");
    while(1)
    {
        UART_ProcessCommand();
    }
}
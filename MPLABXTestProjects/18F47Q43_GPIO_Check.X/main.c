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
#include "mcc_generated_files/system/system.h"

/*
    Main application
*/
#include <stdio.h>
#include <string.h>

#include "pic18common.h"

#define MAX_COMMAND_LEN         (8U)
#define LINEFEED_CHAR           ((uint8_t)'\n')
#define CARRIAGERETURN_CHAR     ((uint8_t)'\r')

static char command[MAX_COMMAND_LEN];
static uint8_t index = 0;
static uint8_t readMessage;

void UART_ExecuteCommand(char *command);
void UART_ProcessCommand(void);

void IO_LED_SetLow() {
    pinmode(B0, OUTPUT);
    pinmode(B1, INPUT);
    pinwrite(B0, LOW);
    pinwrite(B1, LOW);
    pinwrite(B2, LOW);
    pinwrite(B3, LOW);
    pinwrite(B4, LOW);
    pinwrite(B5, LOW);
    LATA0 = HIGH;
    LATA1 = LOW;
}

void IO_LED_SetHigh() {
    pinmode(B0, OUTPUT);
    pinmode(B1, INPUT);
    pinwrite(B0, HIGH);
    pinwrite(B1, HIGH);
    pinwrite(B2, HIGH);
    pinwrite(B3, HIGH);
    pinwrite(B4, HIGH);
    pinwrite(B5, HIGH);
    LATA0 = LOW;
    LATA1 = HIGH;
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

int main(void)
{
    SYSTEM_Initialize();
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

    SYSTEM_Initialize();
    
    printf("In the terminal, send 'ON' to turn the LED on, and 'OFF' to turn it off.\r\n");
    printf("Note: commands 'ON' and 'OFF' are case sensitive.\r\n");
    
    UART_ExecuteCommand("OFF");
    while(1)
    {
        UART_ProcessCommand();
    }
}
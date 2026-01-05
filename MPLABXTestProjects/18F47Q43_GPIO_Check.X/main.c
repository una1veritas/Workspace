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
    pinanalog(A7, DISABLE);
    U3RXPPS = 0x7; //RA7->UART3:RX3; default value 
    // TX
    pinmode(A6, OUTPUT);
    RA6PPS = 0x26;  //RA6->UART3:TX3;
    
    //NCO1
    RA3PPS = 0x3F;  //RA3->NCO1:NCO1;
    pinanalog(A3, DISABLE); //ANSELA3 = 0;	// Disable analog function
    pinmode(A3, OUTPUT);
    
    portwrite(B, 0);
    portmode(B, PORT_OUTPUT);

    // specific pins & ports
    // only to change/to ensure from default setting in pins_default
    
    // /RESET (RE2) output pin    
	pinwrite(E2, LOW);      //LATE2 = 0;		// /Reset = Low
	pinmode(E2, OUTPUT);    //TRISE2 = 0;		// Set as output

	// BE (RE0) output pin
	pinwrite(E0, LOW); //LATE0 = 0;		// BE = Low, bus request
	pinmode(E0, OUTPUT); //TRISE0 = 0;		// Set as output

    // memory and I/O buses */
	// Address bus A15-A8 pin
    // setting the whole 8 bits on port
	portwrite(D, 0); //LATD = 0x00;
	portmode(D, PORT_OUTPUT); //TRISD = 0x00;	// Set as output

	// Address bus A7-A0 pin
	portwrite(B, 0); //LATB = 0x00;
	portmode(B, OUTPUT); //TRISB = 0x00;	// Set as output

	// Data bus D7-D0 pin
	portwrite(C, 0); //LATC = 0x00;
	portmode(C, OUTPUT); //TRISC = 0x00;	// Set as output

	// R/W (RA4) input pin
	pinmodewpu(A4,INPUT); //WPUA4 = 1;		// set weak pull up and mode input since this is input for CLC
	//TRISA4 = 1;		// Set as input

    // pins controlled by CLC output
    // PPS are set in CLC_init
	// RDY (RA0) output pin Low = Halt
	//RA0PPS = 0x00;	// LATA0 -> RA0
	pinwrite(A0, HIGH); //LATA0 = 1;		// RDY = High
	pinmode(A0, OUTPUT); //TRISA0 = 0;		// Set as output

	// /WE (RA2) output pin
	//RA2PPS = 0x00;	// LATA2 -> RA2
	pinwrite(A2, HIGH); //LATA2 = 1;		// /WE = High
	pinmode(A2, OUTPUT); //TRISA2 = 0;		// Set as output

	// /OE (RA5) output pin
	//RA5PPS = 0x00;	// LATA5 -> RA5
	pinwrite(A5, HIGH); //LATA5 = 1;		// /OE = High
	pinmode(A5, OUTPUT); //TRISA5 = 0;		// Set as output
}

void NCO1_init(void){

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
    
    CLC_init();
    Interrupt_init();
}

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

int main(void) {
    
    system_init();
    
    // Enable the Global High Interrupts 
    GlobalInterruptHigh = ENABLE; 

    printf("In the terminal, send 'ON' to turn the LED on, and 'OFF' to turn it off.\r\n");
    printf("Note: commands 'ON' and 'OFF' are case sensitive.\r\n");
    
    LED_out(0);
    while(1)
    {
        UART_ProcessInput();
    }
}


void CLC_init(void) {
   	//========== CLC input pin assign ===========
	// 0,1,4,5 = Port A, C
	// 2,3,6,7 = Port B, D
	CLCIN2PPS = 0x1f;	// RD7 = A15
	CLCIN3PPS = 0x1e;	// RD6 = A14
	CLCIN4PPS = 0x04;	// RA4 = R/W
	CLCIN6PPS = 0x1d;	// RD5 = A13 (CLCx Input 7))
	CLCIN7PPS = 0x1c;	// RD4 = A12

	//========== CLC1 /OE ==========
	CLCSELECT = 0;		// CLC1 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// CLC3 (/IORQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 42;		// NCO1
	CLCnSEL3 = 55;		// CLC5 (RDY)
 
	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x08;	// R/W noninverted
	CLCnGLS2 = 0x20;	// NCO1 noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC1 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC2 /WE ==========
	CLCSELECT = 1;		// CLC2 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// CLC3 (/IOREQ)
	CLCnSEL1 = 4;		// CLCIN4PPS <- R/W
	CLCnSEL2 = 42;		// NCO1
	CLCnSEL3 = 55;		// CLC5 (RDY)
 
	CLCnGLS0 = 0x02;	// CLC3 noninverted
	CLCnGLS1 = 0x04;	// R/W inverted
	CLCnGLS2 = 0x20;	// NCO1 noninverted
	CLCnGLS3 = 0x80;	// CLC5 noninverted

	CLCnPOL = 0x80;		// inverted the CLC2 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC3 /IORQ :IO area 0xB000 - 0xBFFF ==========
	CLCSELECT = 2;		// CLC3 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 2;		// CLCIN2PPS <- A15
	CLCnSEL1 = 3;		// CLCIN3PPS <- A14
	CLCnSEL2 = 6;		// CLCIN6PPS <- A13
	CLCnSEL3 = 7;		// CLCIN7PPS <- A12
 
	CLCnGLS0 = 0x02;	// A15 noninverted
	CLCnGLS1 = 0x04;	// A14 inverted
	CLCnGLS2 = 0x20;	// A13 noninverted
	CLCnGLS3 = 0x80;	// A12 noninverted

	CLCnPOL = 0x80;		// inverted the CLC3 output
	CLCnCON = 0x82;		// 4 input AND

	//========== CLC5 RDY ==========
	CLCSELECT = 4;		// CLC5 select
	CLCnCON = 0x00;		// Disable CLC

	CLCnSEL0 = 53;		// D-FF CLK <- CLC3 (/IORQ)
	CLCnSEL1 = 127;		// D-FF D NC
	CLCnSEL2 = 127;		// D-FF S NC
	CLCnSEL3 = 127;		// D-FF R NC

	CLCnGLS0 = 0x1;		// LCG1D1N
	CLCnGLS1 = 0x0;		// Connect none
	CLCnGLS2 = 0x0;		// Connect none
	CLCnGLS3 = 0x0;		// Connect none

	CLCnPOL = 0x82;		// inverted the CLC5 output, G2 inverted
	CLCnCON = 0x84;		// Select D-FF

	//========== CLC output pin assign ===========
	// 1,2,5,6 = Port A, C
	// 3,4,7,8 = Port B, D
	RA5PPS = 0x01;		// CLC1OUT -> RA5 -> /OE
	RA2PPS = 0x02;		// CLC2OUT -> RA2 -> /WE
	RA0PPS = 0x05;		// CLC5OUT -> RA0 -> RDY

    printf("CLC setup finished.\r\n");
}


#include <xc.h>

#include "uart3.h"

void setup_UART3() {
    // UART3 initialize
	U3BRG = 68; // 19200  //416;	// Console Serial Baud rate 9600bps @ 64MHz
	U3RXEN = 1;		// Receiver enable
	U3TXEN = 1;		// Transmitter enable

	// UART3 Receiver
	ANSELA7 = 0;	// Disable analog function
	TRISA7 = 1;		// RX set as input
	U3RXPPS = 0x07;     //RA7->UART3:RX3;

	// UART3 Transmitter
	ANSELA6 = 0;	// Disable analog function
	LATA6 = 1;		// Default level
	TRISA6 = 0;		// TX set as output
	RA6PPS = 0x26;	//RA6->UART3:TX3;

	U3ON = 1;		// Serial port enable
}

uint8_t UART3_IR_status(void) {
    // from bit 0 to 3 ... U3RXIF, U3TXIF, U3EIF (framing err), U3IF, 
    // bit 5 ... CLC4IF
    return PIR9;
}

uint8_t UART3_Read(void){
    return U3RXB;
}

void UART3_Write(uint8_t txData){
    U3TXB = txData; 
}

int getch(void){
    while(!(UART3_IsRxReady())) { }
    return UART3_Read();
}

void putch(char txData) {
    while(!(UART3_IsTxReady())) { }
    return UART3_Write(txData);   
}


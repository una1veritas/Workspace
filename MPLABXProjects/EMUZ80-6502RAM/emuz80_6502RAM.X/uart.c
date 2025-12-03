/*
 *
 */

#include <xc.h>
#include "uart.h"

// console input buffers
#define U3B_SIZE 128
unsigned char rx_buf[U3B_SIZE];	//UART Rx ring buffer
unsigned int rx_wp, rx_rp, rx_cnt;

////////////// UART3 Receive interrupt ////////////////////////////
// UART3 Rx interrupt
// PIR9 (bit0:U3RXIF bit1:U3TXIF)
/////////////////////////////////////////////////////////////////
void __interrupt(irq(U3RX),base(8)) URT3Rx_ISR(){

	unsigned char rx_data;

	rx_data = U3RXB;			// get rx data

	if (rx_cnt < U3B_SIZE) {
		rx_buf[rx_wp] = rx_data;
		rx_wp = (rx_wp + 1) & (U3B_SIZE - 1);
		rx_cnt++;
	}
}

// UART3 Transmit
/*
 * non-buffered TX
 */
void uart_tx(char c) {
    while(!U3TXIF);             // Wait or Tx interrupt flag set
    U3TXB = c;                  // Write data
}

// UART3 Recive
int uart_rx(void) {
	char c;

	while(!rx_cnt);             // Wait for Rx interrupt flag set
	GIE = 0;                // Disable interrupt
	c = rx_buf[rx_rp];
	rx_rp = (rx_rp + 1) & ( U3B_SIZE - 1);
	rx_cnt--;
	GIE = 1;                // enable interrupt
    return c;               // Read data
}

void devio_init(void) {
	rx_wp = 0;
	rx_rp = 0;
	rx_cnt = 0;
    U3RXIE = 1;          // Receiver interrupt enable
}

/*
 * common system initialization for UART
 * 
 */
/*

	// UART3 initialize
    U3BRG = 416;			// 9600bps @ 64MHz
    U3RXEN = 1;				// Receiver enable
    U3TXEN = 1;				// Transmitter enable

    // UART3 Receiver
    TRISA7 = 1;				// RX set as input
    U3RXPPS = 0x07;			// RA7->UART3:RXD;

    // UART3 Transmitter
    LATA6 = 1;				// Default level
    TRISA6 = 0;				// TX set as output
    RA6PPS = 0x26;			// UART3:TXD -> RA6;

    U3ON = 1;				// Serial port enable
*/
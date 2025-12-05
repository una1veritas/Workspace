/*
 *
 */

#include <xc.h>
#include "uart.h"

// console input buffers
#define U3B_SIZE 128
#define UART3_RX_BUFFER_SIZE (128U) //buffer size should be 2^n
#define UART3_RX_BUFFER_MASK (UART3_RX_BUFFER_SIZE - 1U)

unsigned char rx_buf[U3B_SIZE];	//UART Rx ring buffer
unsigned int rx_wp, rx_rp, rx_cnt;
static volatile uint8_t uart3RxHead = 0;
static volatile uint8_t uart3RxTail = 0;
static volatile uint8_t uart3RxCount;
static volatile uint8_t uart3RxBuffer[UART3_RX_BUFFER_SIZE];

static volatile uart3_status_t uart3RxStatusBuffer[UART3_RX_BUFFER_SIZE];
static volatile uart3_status_t uart3RxLastError;

//static void (*UART3_TxCompleteInterruptHandler)(void) = NULL;
void (*UART3_RxInterruptHandler)(void);
static void (*UART3_RxCompleteInterruptHandler)(void) = NULL;

void UART3_TransmitISR (void);
void UART3_ReceiveISR(void);

void UART3_Initialize(void)
{
    PIE9bits.U3RXIE = 0;   
    UART3_RxInterruptHandler = UART3_ReceiveISR; 
    //PIE9bits.U3TXIE = 0; 
    //UART3_TxInterruptHandler = UART3_TransmitISR; 

    // Set the UART3 module to the options selected in the user interface.

    //P1L 0x0; 
    U3P1L = 0x0;
    //P2L 0x0; 
    U3P2L = 0x0;
    //P3L 0x0; 
    U3P3L = 0x0;
    //MODE Asynchronous 8-bit mode; RXEN enabled; TXEN enabled; ABDEN disabled; BRGS high speed; 
    U3CON0 = 0xB0;
    //SENDB disabled; BRKOVR disabled; RXBIMD Set RXBKIF on rising RX input; WUE disabled; ON enabled; 
    U3CON1 = 0x80;
    //FLO off; TXPOL not inverted; STP Transmit 1Stop bit, receiver verifies first Stop bit; RXPOL not inverted; RUNOVF RX input shifter stops all activity; 
    U3CON2 = 0x0;
    //BRGL 21; 
    U3BRGL = 0x15;
    //BRGH 1; 
    U3BRGH = 0x1;
    //TXBE empty; STPMD in middle of first Stop bit; TXWRE No error; 
    U3FIFO = 0x2E;
    //ABDIE disabled; ABDIF Auto-baud not enabled or not complete; WUIF WUE not enabled by software; 
    U3UIR = 0x0;
    //TXCIF equal; RXFOIF not overflowed; RXBKIF No Break detected; FERIF no error; CERIF No Checksum error; ABDOVF Not overflowed; PERIF no parity error; TXMTIF empty; 
    U3ERRIR = 0x80;
    //TXCIE disabled; RXFOIE disabled; RXBKIE disabled; FERIE disabled; CERIE disabled; ABDOVE disabled; PERIE disabled; TXMTIE disabled; 
    U3ERRIE = 0x0;

    uart3RxLastError.status = 0;  
    uart3RxHead = 0;
    uart3RxTail = 0;
    uart3RxCount = 0;
    
    PIE9bits.U3RXIE = 1;
}

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

void UART3_ReceiveInterruptEnable(void) {
    PIE9bits.U3RXIE = 1;
}

void UART3_ReceiveInterruptDisable(void) {
    PIE9bits.U3RXIE = 0;
}

bool UART3_IsRxReady(void) {
    return (uart3RxCount ? true : false);
}

bool UART3_IsTxReady(void) {
    return (uart3TxBufferRemaining ? true : false);
}

bool UART3_IsTxDone(void) {
    return U3ERRIRbits.TXMTIF;
}

size_t UART3_ErrorGet(void) {
    uart3RxLastError.status = uart3RxStatusBuffer[(uart3RxTail) & UART3_RX_BUFFER_MASK].status;

    return uart3RxLastError.status;
}

uint8_t UART3_Read(void) {
    uint8_t readValue  = 0;
    uint8_t tempRxTail;
    
    readValue = uart3RxBuffer[uart3RxTail];
    tempRxTail = (uart3RxTail + 1U) & UART3_RX_BUFFER_MASK; // Buffer size of RX should be in the 2^n  
    uart3RxTail = tempRxTail;
    PIE9bits.U3RXIE = 0; 
    if (0U != uart3RxCount) {
        uart3RxCount--;
    }
    PIE9bits.U3RXIE = 1;
    return readValue;
}


void UART3_ReceiveISR(void) {
    uint8_t regValue;
    uint8_t tempRxHead;

    // use this default receive interrupt handler code
    uart3RxStatusBuffer[uart3RxHead].status = 0;

    if(true == U3ERRIRbits.FERIF) {
        uart3RxStatusBuffer[uart3RxHead].ferr = 1;
        if(NULL != UART3_FramingErrorHandler) {
            UART3_FramingErrorHandler();
        } 
    }
    if(true == U3ERRIRbits.RXFOIF) {
        uart3RxStatusBuffer[uart3RxHead].oerr = 1;
        if(NULL != UART3_OverrunErrorHandler) {
            UART3_OverrunErrorHandler();
        }   
    }   
    if(true == U3ERRIRbits.PERIF) {
        uart3RxStatusBuffer[uart3RxHead].perr = 1;
        if (NULL != UART3_ParityErrorHandler) {
            UART3_ParityErrorHandler();
        }   
    }  
 
    regValue = U3RXB;
    
    tempRxHead = (uart3RxHead + 1U) & UART3_RX_BUFFER_MASK;
    if (tempRxHead == uart3RxTail) {
		// ERROR! Receive buffer overflow 
	} else {
        uart3RxBuffer[uart3RxHead] = regValue;
		uart3RxHead = tempRxHead;
		uart3RxCount++;
	}   
    
    if (NULL != UART3_RxCompleteInterruptHandler) {
        (*UART3_RxCompleteInterruptHandler)();
    } 
}


int getch(void) {
    while(!(UART3_IsRxReady())) { }
    return UART3_Read();
}

// printf callback
void putch(char txData) {
    while(!(UART3_IsTxReady())) { }
    return UART3_Write(txData);   
}


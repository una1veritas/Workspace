/**
 * UART3 Generated Driver API Header File
 * 
 * @file uart3.c
 * 
 * @ingroup uart3
 * 
 * @brief This is the generated driver implementation file for the UART3 driver using the Universal Asynchronous Receiver and Transmitter (UART) module.
 *
 * @version UART3 Driver Version 3.0.9
*/

/**
  Section: Included Files
*/
#include "uart3.h"

/**
  Section: Macro Declarations
*/

#define UART3_TX_BUFFER_SIZE (128U) //buffer size should be 2^n
#define UART3_TX_BUFFER_MASK (UART3_TX_BUFFER_SIZE - 1U) 

#define UART3_RX_BUFFER_SIZE (128U) //buffer size should be 2^n
#define UART3_RX_BUFFER_MASK (UART3_RX_BUFFER_SIZE - 1U)

/**
  Section: UART3 variables
*/
static volatile uint8_t uart3TxHead = 0;
static volatile uint8_t uart3TxTail = 0;
static volatile uint8_t uart3TxBufferRemaining;
static volatile uint8_t uart3TxBuffer[UART3_TX_BUFFER_SIZE];

static volatile uint8_t uart3RxHead = 0;
static volatile uint8_t uart3RxTail = 0;
static volatile uint8_t uart3RxCount;
static volatile uint8_t uart3RxBuffer[UART3_RX_BUFFER_SIZE];
/**
 * @misradeviation{@advisory,19.2}
 * The UART error status necessitates checking the bitfield and accessing the status within the group byte therefore the use of a union is essential.
 */
 /* cppcheck-suppress misra-c2012-19.2 */
//static volatile uart3_status_t uart3RxStatusBuffer[UART3_RX_BUFFER_SIZE];

 /**
 * @misradeviation{@advisory,19.2}
 * The UART error status necessitates checking the bitfield and accessing the status within the group byte therefore the use of a union is essential.
 */
 /* cppcheck-suppress misra-c2012-19.2 */
//static volatile uart3_status_t uart3RxLastError;

/**
  Section: UART3 APIs
*/

void (*UART3_TxInterruptHandler)(void);
/* cppcheck-suppress misra-c2012-8.9 */
static void (*UART3_TxCompleteInterruptHandler)(void) = NULL;
void (*UART3_RxInterruptHandler)(void);
static void (*UART3_RxCompleteInterruptHandler)(void) = NULL;

void UART3_TransmitISR (void);
void UART3_ReceiveISR(void);

/**
  Section: UART3  APIs
*/

void UART3_Initialize(void)
{
    PIE9bits.U3RXIE = 0;   
    UART3_RxInterruptHandler = UART3_ReceiveISR; 
    PIE9bits.U3TXIE = 0; 
    UART3_TxInterruptHandler = UART3_TransmitISR; 

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

//    UART3_FramingErrorCallbackRegister(UART3_DefaultFramingErrorCallback);
//    UART3_OverrunErrorCallbackRegister(UART3_DefaultOverrunErrorCallback);
//    UART3_ParityErrorCallbackRegister(UART3_DefaultParityErrorCallback);

//    uart3RxLastError.status = 0;  
    uart3TxHead = 0;
    uart3TxTail = 0;
    uart3TxBufferRemaining = sizeof(uart3TxBuffer);
    uart3RxHead = 0;
    uart3RxTail = 0;
    uart3RxCount = 0;
    PIE9bits.U3RXIE = 1;
}

void UART3_Deinitialize(void)
{
    PIE9bits.U3RXIE = 0;   
    PIE9bits.U3TXIE = 0;
    U3RXB = 0x00;
    U3TXB = 0x00;
    U3P1L = 0x00;
    U3P2L = 0x00;
    U3P3L = 0x00;
    U3CON0 = 0x00;
    U3CON1 = 0x00;
    U3CON2 = 0x00;
    U3BRGL = 0x00;
    U3BRGH = 0x00;
    U3FIFO = 0x00;
    U3UIR = 0x00;
    U3ERRIR = 0x00;
    U3ERRIE = 0x00;
}

void UART3_Enable(void) {
    U3CON1bits.ON = 1; 
}

void UART3_Disable(void) {
    U3CON1bits.ON = 0; 
}
/*
void UART3_TransmitEnable(void) {
    U3CON0bits.TXEN = 1;
}

void UART3_TransmitDisable(void) {
    U3CON0bits.TXEN = 0;
}

void UART3_ReceiveEnable(void) {
    U3CON0bits.RXEN = 1;
}

void UART3_ReceiveDisable(void) {
    U3CON0bits.RXEN = 0;
}
*/
void UART3_SendBreakControlEnable(void) {
    U3CON1bits.SENDB = 1;
}

void UART3_SendBreakControlDisable(void) {
    U3CON1bits.SENDB = 0;
}

void UART3_TransmitInterruptEnable(void)
{
    PIE9bits.U3TXIE = 1;
}

void UART3_TransmitInterruptDisable(void)
{ 
    PIE9bits.U3TXIE = 0;
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
/*
size_t UART3_ErrorGet(void) {
//    uart3RxLastError.status = uart3RxStatusBuffer[(uart3RxTail) & UART3_RX_BUFFER_MASK].status;

    return U3ERRIRbits;
}
*/

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

    /*
    // use this default receive interrupt handler code
    uart3RxLastError.status = 0;
    //uart3RxStatusBuffer[uart3RxHead].status = 0;

    if(true == U3ERRIRbits.FERIF) {
        uart3RxLastError.ferr = 1;
    }
    if(true == U3ERRIRbits.RXFOIF) {
        uart3RxLastError.oerr = 1;
    }   
    if(true == U3ERRIRbits.PERIF) {
        uart3RxLastError.perr = 1;
    }  
    */
    
    regValue = U3RXB;
    
    tempRxHead = (uart3RxHead + 1U) & UART3_RX_BUFFER_MASK;
    if (tempRxHead == uart3RxTail) {
		// ERROR! Receive buffer overflow 
	} else {
        uart3RxBuffer[uart3RxHead] = regValue;
		uart3RxHead = tempRxHead;
		uart3RxCount++;
	}   
    /*
    if (NULL != UART3_RxCompleteInterruptHandler) {
        (*UART3_RxCompleteInterruptHandler)();
    } 
    */
}

void UART3_Write(uint8_t txData) {
    uint8_t tempTxHead;
    
    if(0 == PIE9bits.U3TXIE) {
        U3TXB = txData;
    } else if(0U < uart3TxBufferRemaining) {
       // check if at least one byte place is available in TX buffer
       uart3TxBuffer[uart3TxHead] = txData;
       tempTxHead = (uart3TxHead + 1U) & UART3_TX_BUFFER_MASK;

       uart3TxHead = tempTxHead;
       PIE9bits.U3TXIE = 0; //Critical value decrement
       uart3TxBufferRemaining--; // one less byte remaining in TX buffer
    } else {
        //overflow condition; uart3TxBufferRemaining is 0 means TX buffer is full
    }
    PIE9bits.U3TXIE = 1;
}


void UART3_TransmitISR(void) {
    uint8_t tempTxTail;
    // use this default transmit interrupt handler code
    if (sizeof(uart3TxBuffer) > uart3TxBufferRemaining) {
       // check if all data is transmitted
       U3TXB = uart3TxBuffer[uart3TxTail];
       tempTxTail = (uart3TxTail + 1U) & UART3_TX_BUFFER_MASK;
       
       uart3TxTail = tempTxTail;
       uart3TxBufferRemaining++; // one byte sent, so 1 more byte place is available in TX buffer
    } else {
        PIE9bits.U3TXIE = 0;
    }
    /*
    if ( NULL != UART3_TxCompleteInterruptHandler ) {
        (*UART3_TxCompleteInterruptHandler)();
    }
    */
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



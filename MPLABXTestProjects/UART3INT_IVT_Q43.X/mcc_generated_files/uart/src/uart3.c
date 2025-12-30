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

/*
© [2025] Microchip Technology Inc. and its subsidiaries.

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

/**
  Section: Included Files
*/
#include "../uart3.h"

/**
  Section: Macro Declarations
*/

#define UART3_RX_BUFFER_SIZE (256U) //buffer size should be 2^n
#define UART3_RX_BUFFER_MASK (UART3_RX_BUFFER_SIZE - 1U)

/**
  Section: UART3 variables
*/
static volatile uint16_t uart3RxHead = 0;
static volatile uint16_t uart3RxTail = 0;
static volatile uint16_t uart3RxCount;
static volatile uint8_t uart3RxBuffer[UART3_RX_BUFFER_SIZE];
/**
 * @misradeviation{@advisory,19.2}
 * The UART error status necessitates checking the bitfield and accessing the status within the group byte therefore the use of a union is essential.
 */
 /* cppcheck-suppress misra-c2012-19.2 */
static volatile uart3_status_t uart3RxStatusBuffer[UART3_RX_BUFFER_SIZE];

 /**
 * @misradeviation{@advisory,19.2}
 * The UART error status necessitates checking the bitfield and accessing the status within the group byte therefore the use of a union is essential.
 */
 /* cppcheck-suppress misra-c2012-19.2 */
static volatile uart3_status_t uart3RxLastError;

/**
  Section: UART3 APIs
*/

static void (*UART3_FramingErrorHandler)(void);
static void (*UART3_OverrunErrorHandler)(void);
static void (*UART3_ParityErrorHandler)(void);
void (*UART3_RxInterruptHandler)(void);
static void (*UART3_RxCompleteInterruptHandler)(void) = NULL;

static void UART3_DefaultFramingErrorCallback(void);
static void UART3_DefaultOverrunErrorCallback(void);
static void UART3_DefaultParityErrorCallback(void);
void UART3_ReceiveISR(void);

/**
  Section: UART3  APIs
*/

void UART3_Initialize(void)
{
    PIE9bits.U3RXIE = 0;   
    UART3_RxInterruptHandler = UART3_ReceiveISR; 

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

    UART3_FramingErrorCallbackRegister(UART3_DefaultFramingErrorCallback);
    UART3_OverrunErrorCallbackRegister(UART3_DefaultOverrunErrorCallback);
    UART3_ParityErrorCallbackRegister(UART3_DefaultParityErrorCallback);

    uart3RxLastError.status = 0;  
    uart3RxHead = 0;
    uart3RxTail = 0;
    uart3RxCount = 0;
    PIE9bits.U3RXIE = 1;
}

void UART3_Deinitialize(void)
{
    PIE9bits.U3RXIE = 0;   
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

void UART3_Enable(void)
{
    U3CON1bits.ON = 1; 
}

void UART3_Disable(void)
{
    U3CON1bits.ON = 0; 
}

void UART3_TransmitEnable(void)
{
    U3CON0bits.TXEN = 1;
}

void UART3_TransmitDisable(void)
{
    U3CON0bits.TXEN = 0;
}

void UART3_ReceiveEnable(void)
{
    U3CON0bits.RXEN = 1;
}

void UART3_ReceiveDisable(void)
{
    U3CON0bits.RXEN = 0;
}

void UART3_SendBreakControlEnable(void)
{
    U3CON1bits.SENDB = 1;
}

void UART3_SendBreakControlDisable(void)
{
    U3CON1bits.SENDB = 0;
}

void UART3_AutoBaudSet(bool enable)
{
    if(enable)
    {
        U3CON0bits.ABDEN = 1; 
    }
    else
    {
      U3CON0bits.ABDEN = 0;  
    }
}


bool UART3_AutoBaudQuery(void)
{
    return (bool)U3UIRbits.ABDIF; 
}

void UART3_AutoBaudDetectCompleteReset(void)
{
    U3UIRbits.ABDIF = 0; 
}

bool UART3_IsAutoBaudDetectOverflow(void)
{
    return (bool)U3ERRIRbits.ABDOVF; 
}

void UART3_AutoBaudDetectOverflowReset(void)
{
    U3ERRIRbits.ABDOVF = 0; 
}

void UART3_ReceiveInterruptEnable(void)
{
    PIE9bits.U3RXIE = 1;
}
void UART3_ReceiveInterruptDisable(void)
{
    PIE9bits.U3RXIE = 0;
}

bool UART3_IsRxReady(void)
{
    return (uart3RxCount ? true : false);
}

bool UART3_IsTxReady(void)
{
    return (bool)(U3FIFObits.TXBE && U3CON0bits.TXEN);
}

bool UART3_IsTxDone(void)
{
    return U3ERRIRbits.TXMTIF;
}

size_t UART3_ErrorGet(void)
{
    uart3RxLastError.status = uart3RxStatusBuffer[(uart3RxTail) & UART3_RX_BUFFER_MASK].status;

    return uart3RxLastError.status;
}

uint8_t UART3_Read(void)
{
    uint8_t readValue  = 0;
    uint16_t tempRxTail;
    
    readValue = uart3RxBuffer[uart3RxTail];
    tempRxTail = (uart3RxTail + 1U) & UART3_RX_BUFFER_MASK; // Buffer size of RX should be in the 2^n  
    uart3RxTail = tempRxTail;  
    PIE9bits.U3RXIE = 0; 
    if(0U != uart3RxCount)
    {
        uart3RxCount--;
    }
    PIE9bits.U3RXIE = 1;
    return readValue;
}

void __interrupt(irq(IRQ_U3RX), base(8)) UART3_Receive_Vector_ISR(void)
{   
    UART3_ReceiveISR();
}

void UART3_ReceiveISR(void)
{
    uint8_t regValue;
    uint16_t tempRxHead;

    // use this default receive interrupt handler code
    uart3RxStatusBuffer[uart3RxHead].status = 0;

    if(true == U3ERRIRbits.FERIF)
    {
        uart3RxStatusBuffer[uart3RxHead].ferr = 1;
        if(NULL != UART3_FramingErrorHandler)
        {
            UART3_FramingErrorHandler();
        } 
    }
    if(true == U3ERRIRbits.RXFOIF)
    {
        uart3RxStatusBuffer[uart3RxHead].oerr = 1;
        if(NULL != UART3_OverrunErrorHandler)
        {
            UART3_OverrunErrorHandler();
        }   
    }   
    if(true == U3ERRIRbits.PERIF)
    {
        uart3RxStatusBuffer[uart3RxHead].perr = 1;
        if(NULL != UART3_ParityErrorHandler)
        {
            UART3_ParityErrorHandler();
        }   
    }  
 
    regValue = U3RXB;
    
    tempRxHead = (uart3RxHead + 1U) & UART3_RX_BUFFER_MASK;
    if (tempRxHead == uart3RxTail) 
    {
		// ERROR! Receive buffer overflow 
	} 
    else
    {
        uart3RxBuffer[uart3RxHead] = regValue;
		uart3RxHead = tempRxHead;
		uart3RxCount++;
	}   
    
    if(NULL != UART3_RxCompleteInterruptHandler)
    {
        (*UART3_RxCompleteInterruptHandler)();
    } 
}

void UART3_Write(uint8_t txData)
{
    U3TXB = txData; 
}


int getch(void)
{
    while(!(UART3_IsRxReady()))
    {

    }
    return UART3_Read();
}

void putch(char txData)
{
    while(!(UART3_IsTxReady()))
    {

    }
    return UART3_Write(txData);   
}





static void UART3_DefaultFramingErrorCallback(void)
{
    
}

static void UART3_DefaultOverrunErrorCallback(void)
{
    
}

static void UART3_DefaultParityErrorCallback(void)
{
    
}

void UART3_FramingErrorCallbackRegister(void (* callbackHandler)(void))
{
    if(NULL != callbackHandler)
    {
        UART3_FramingErrorHandler = callbackHandler;
    }
}

void UART3_OverrunErrorCallbackRegister(void (* callbackHandler)(void))
{
    if(NULL != callbackHandler)
    {
        UART3_OverrunErrorHandler = callbackHandler;
    }    
}

void UART3_ParityErrorCallbackRegister(void (* callbackHandler)(void))
{
    if(NULL != callbackHandler)
    {
        UART3_ParityErrorHandler = callbackHandler;
    } 
}
void UART3_RxCompleteCallbackRegister(void (* callbackHandler)(void))
{
    if(NULL != callbackHandler)
    {
       UART3_RxCompleteInterruptHandler = callbackHandler; 
    }   
}


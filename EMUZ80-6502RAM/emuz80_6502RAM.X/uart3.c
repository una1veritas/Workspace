
#include <xc.h>

#include "uart3.h"

#ifdef UART3RX_INTERRUPT
#define UART3_RX_BUFFER_SIZE (128U) //buffer size should be 2^n
#define UART3_RX_BUFFER_MASK (UART3_RX_BUFFER_SIZE - 1U)

/**
  Section: UART3 variables
*/
static volatile uint8_t uart3RxHead = 0;
static volatile uint8_t uart3RxTail = 0;
static volatile uint8_t uart3RxCount;
static volatile uint8_t uart3RxBuffer[UART3_RX_BUFFER_SIZE];

 /**
 * @misradeviation{@advisory,19.2}
 * The UART error status necessitates checking the bitfield and accessing the status within the group byte therefore the use of a union is essential.
 */
 /* cppcheck-suppress misra-c2012-19.2 */
static volatile uart3_status_t uart3RxLastError;
void (*UART3_RxInterruptHandler)(void);

void UART3_ReceiveISR(void);

#endif

void setup_UART3() {
    /*
    //6551 Speed setting x 8
    static uint16_t baud[] = {
        416, // dummy 16 x External CLOCK
        416, // dummy 120
        416, // dummy 300
        416, // dummy 600
        416, // dummy 1200
        416, // dummy 2400
        416, // dummy 4800
        416, // 9600
        277, // 14400
        207, // 19200
        138, // 28800
        103, // 38400
        68,  // 57600
        51,  // 76800
        34,  // 115200
        25,  // 153600
    };
     * */
    /* interrupt */
#ifdef UART3RX_INTERRUPT
    PIE9bits.U3RXIE = 0;   
    UART3_RxInterruptHandler = UART3_ReceiveISR; 
#endif
    
    // UART3 initialize
	U3BRG = 277; //68; // normal speed setting. baud[12]; // 57600 // 9600  //416;	// Console Serial Baud rate 9600bps @ 64MHz
    // U3CON0 = 0xb0 (10110000))
    U3BRGS = 1;
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

    // U3CON1 = 0x80
	U3ON = 1;		// Serial port enable
    
#ifdef UART3RX_INTERRUPT
    /* interrupt & buffering */
    uart3RxLastError.status = 0;  
    
    uart3RxHead = 0;
    uart3RxTail = 0;
    uart3RxCount = 0;
    PIE9bits.U3RXIE = 1;
#endif
}

uint8_t UART3_IR_status(void) {
    // from bit 0 to 3 ... U3RXIF, U3TXIF, U3EIF (framing err), U3IF, 
    // bit 5 ... CLC4IF
    return PIR9;
}

#ifdef UART3RX_INTERRUPT
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

void __interrupt(irq(IRQ_U3RX), base(8)) UART3_Receive_Vector_ISR(void)
{   
    UART3_ReceiveISR();
}

void UART3_ReceiveISR(void) {
    uint8_t regValue;
    uint8_t tempRxHead;
    
    regValue = U3RXB;
    
    tempRxHead = (uart3RxHead + 1U) & UART3_RX_BUFFER_MASK;
    if (tempRxHead == uart3RxTail) {
		// ERROR! Receive buffer overflow 
	} else {
        uart3RxBuffer[uart3RxHead] = regValue;
		uart3RxHead = tempRxHead;
		uart3RxCount++;
	}
}
#else
uint8_t UART3_Read(void){
    return U3RXB;
}
#endif

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


/* 
 * File:   uart3.h
 * Author: sin
 *
 * Created on December 6, 2025, 11:09 AM
 */

#ifndef UART3_H
#define	UART3_H

#ifdef	__cplusplus
extern "C" {
#endif

//#define UART3RX_INTERRUPT

#include <stdint.h>

#define UART3_IsRxReady() (! U3FIFObits.RXBE)
#define UART3_IsTxReady() (U3FIFObits.TXBE && U3CON0bits.TXEN)
#define UART3_IsTxDone()  (U3ERRIRbits.TXMTIF)

#ifdef UART3RX_INTERRUPT 
typedef union {
    struct {
        uint8_t perr : 1;     /**<This is a bit field for Parity Error status*/
        uint8_t ferr : 1;     /**<This is a bit field for Framing Error status*/
        uint8_t oerr : 1;     /**<This is a bit field for Overfrun Error status*/
        uint8_t reserved : 5; /**<Reserved*/
    };
    size_t status;            /**<Group byte for status errors*/
} uart3_status_t;

extern void (*UART3_RxInterruptHandler)(void);
void UART3_ReceiveISR(void);


void UART3_ReceiveInterruptEnable(void);
void UART3_ReceiveInterruptDisable(void);

#endif

void setup_UART3(void);
uint8_t UART3_IR_status(void);
uint8_t UART3_Read(void);
void UART3_Write(uint8_t txData);
int getch(void);
void putch(char txData);


#ifdef	__cplusplus
}
#endif

#endif	/* UART3_H */


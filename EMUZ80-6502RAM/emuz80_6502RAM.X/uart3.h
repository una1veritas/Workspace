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

#include <stdint.h>

#define UART3_IsRxReady() (!U3FIFObits.RXBE)
#define UART3_IsTxReady() (U3FIFObits.TXBE && U3CON0bits.TXEN)
#define UART3_IsTxDone()  (U3ERRIRbits.TXMTIF)

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


/* 
 * File:   uart.h
 * Author: sin
 *
 * Created on December 3, 2025, 12:06 AM
 */

#ifndef UART_H
#define	UART_H

#ifdef	__cplusplus
extern "C" {
#endif

typedef union {
    struct {
        uint8_t perr : 1;     /**<This is a bit field for Parity Error status*/
        uint8_t ferr : 1;     /**<This is a bit field for Framing Error status*/
        uint8_t oerr : 1;     /**<This is a bit field for Overfrun Error status*/
        uint8_t reserved : 5; /**<Reserved*/
    };
    size_t status;            /**<Group byte for status errors*/
}uart3_status_t;

void uart_tx(char c);
int uart_rx(void);
void devio_init(void);


#ifdef	__cplusplus
}
#endif

#endif	/* UART_H */


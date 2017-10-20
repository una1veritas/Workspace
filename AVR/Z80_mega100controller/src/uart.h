#ifndef _UART_H_
#define _UART_H_

void uart_init(unsigned long baud);
int uart_getchar(void);
unsigned int uart_available (void);
int uart_peek(void);

#endif

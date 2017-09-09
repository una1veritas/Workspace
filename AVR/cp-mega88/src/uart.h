#ifndef _UART_H_
#define _UART_H_

//#define USE_INTERRUPT_TX

#if defined (__AVR_ATmega88__) || defined (__AVR_ATmega88P__) || \
	defined (__AVR_ATmega168__) || defined (__AVR_ATmega168P__) || \
	defined (__AVR_ATmega328__) || defined (__AVR_ATmega328P__)
#define USARTn_RX_vect USART_RX_vect
#elif defined(__AVR_ATmega1280__) || defined(__AVR_ATmega2560__) || \
		defined(__AVR_ATmega644P__) || defined(__AVR_ATmega1284P__)
#define USARTn_RX_vect USART0_RX_vect
#endif


#ifdef UART_DEBUG
int debug_rxenq();
int debug_rxdeq();
unsigned char * debug_rxfifo();
#endif

void uart_init(void);
unsigned char uart_tx(unsigned char);
unsigned char uart_rx(void);

int uart_getchar();
void uart_putchar(unsigned char c);
int uart_peek();
unsigned int uart_available();

void uart_putstr (char *s);
void uart_puthex(unsigned char c);
void uart_putnum_u16(unsigned short n, int digit);
void uart_puts(char *s);
void uart_putsln(char *s);

#endif // _UART_H_

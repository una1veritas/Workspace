#define FOSC F_CPU
#define BAUD 9600

void uart_init(void);      // initializes IO
int uart_putchar(char c, FILE *stream);
uint8_t uart_getchar(void);

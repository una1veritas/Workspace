/*
 * main.c
 *
 *  Created on: 2017/08/28
 *      Author: sin
 */

#include <avr/io.h>
#include <util/delay.h>

#include <avr/interrupt.h>

volatile int i=0;
volatile uint8_t buffer[20];
volatile uint8_t StrRxFlag=0;

ISR(USART0_RX_vect)
{
    buffer[i]=UDR0;         //Read USART data register
    if(buffer[i++]=='\r')   //check for carriage return terminator and increment buffer index
    {
        // if terminator detected
        StrRxFlag=1;        //Set String received flag
        buffer[i-1]=0x00;   //Set string terminator to 0x00
        i=0;                //Reset buffer index
    }
}

#define ADDRL PORTA
#define ADDRL_DDR DDRA
#define ADDRH PORTC
#define ADDRH_DDR DDRC
#define ADDRX_DDR DDRB
#define ADDRX PORTB
#define ADDRX_MASK (1<<0)

#define DATA_OUT  PORTA
#define DATA_IN  PINA
#define DATA_DDR DDRA

#define CONTROL PORTB
#define CONTROL_DDR DDRB
#define SRAM_CS (1<<3)
#define SRAM_OE (1<<2)
#define SRAM_WE (1<<1)
#define LATCH_L (1<<7)

void sram_init(void) {
	ADDRL_DDR = 0xff;
	ADDRH_DDR = 0xff;
	ADDRX_DDR = ADDRX_MASK;

	CONTROL_DDR = ( SRAM_CS | SRAM_OE | SRAM_WE | LATCH_L );
	CONTROL |=  ( SRAM_CS | SRAM_OE | SRAM_WE | LATCH_L );
}

void sram_select(void) {
	CONTROL &= ~SRAM_CS;
}

void sram_deselect(void) {
	CONTROL |= SRAM_CS;
}

void sram_addr_out(uint32_t addr) {
	ADDRL_DDR = 0xff;
	ADDRL = addr;
	CONTROL &= ~LATCH_L;
	CONTROL |= LATCH_L;
	addr >>= 8;
	ADDRH = addr;
	addr >>= 8;
	ADDRX = (addr & ADDRX_MASK) | (ADDRX & ~ADDRX_MASK);
}

void sram_data_out(uint8_t val) {
	DATA_OUT = val;
	CONTROL &= ~SRAM_WE;
	CONTROL |= SRAM_WE;
}

uint8_t sram_data_in(void) {
	uint8_t val;
	DATA_DDR = 0x00;
	CONTROL &= ~SRAM_OE;
	__asm__ __volatile__ (
			"nop" "\n\t"
			"nop");
	val = DATA_IN;
	CONTROL |= SRAM_OE;
	return val;
}

void serial0_init(uint32_t baud) {
    cli();
    // Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet

    UBRR0 = (((F_CPU / (baud * 16UL))) - 1);                 // Set the baud rate prescale rate register
    UCSR0B = ((1<<RXEN0)|(1<<TXEN0)|(1 << RXCIE0));       // Enable receiver and transmitter and Rx interrupt
    UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));  // Set frame format: 8data, 1 stop bit. See Table 22-7 for details
    sei();
}
/*
void serial0_init(uint32_t baud) {
	// Macro to determine the baud prescale rate see table 22.1 in the Mega datasheet
	UBRR0 = (((F_CPU / (baud * 16UL))) - 1);         // Set the baud rate prescale rate register

	UCSR0C = ((0<<USBS0)|(1 << UCSZ01)|(1<<UCSZ00));   // Set frame format: 8data, 1 stop bit. See Table 22-7 for details
	UCSR0B = ((1<<RXEN0)|(1<<TXEN0));       // Enable receiver and transmitter
}
*/
void serial0_tx(uint8_t data) {
    //while the transmit buffer is not empty loop
    while(!(UCSR0A & (1<<UDRE0)));

    //when the buffer is empty write data to the transmitted
    UDR0 = data;
}

uint8_t serial0_rx(void) {
	/* Wait for data to be received */
	while (!(UCSR0A & (1<<RXC0)));
	/* Get and return received data from buffer */
	return UDR0;
}

void serial0_puts(char* StringPtr)
// sends the characters from the string one at a time to the USART
{
    while(*StringPtr != 0x00)
    {
        serial0_tx(*StringPtr);
        StringPtr++;
    }
}

int main(void) {

	sram_init();
	serial0_init(9600);

	serial0_puts("Hello there.\n");

	for(;;) {
		sram_select();
		_delay_ms(500);
		sram_deselect();
		_delay_ms(500);
	}
	return 0;
}

/*
 * main.c
 *
 *  Created on: 2017/08/28
 *      Author: sin
 */

#include <avr/io.h>
#include <util/delay.h>

#include <stdio.h>
#include <stdlib.h>

#include <avr/interrupt.h>
#include "uart_serial.h"

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

static int uart_putchar(char c, FILE *stream)
{
  uart_tx(c);
  return c;
}
static FILE uartout = FDEV_SETUP_STREAM(uart_putchar, NULL, _FDEV_SETUP_WRITE);


#define ADDR PORTA
#define ADDR_DDR DDRA

#define DATA_OUT  PORTC
#define DATA_IN  PINC
#define DATA_DDR DDRC

#define BUS_CONTROL_PORT PORTB
#define BUS_CONTROL_DIR  DDRB
#define BUS_RESET 	(1<<0)
#define BUS_AVRWAIT (1<<1)
#define BUS_BUSREQ  (1<<2)

#define SRAM_CONTROL_PORT PORTD
#define SRAM_CONTROL_DDR  DDRD
// A16
#define SRAM_A16  (1<<2)
// WR
#define SRAM_WE (1<<3)
// RD
#define SRAM_OE (1<<4)
#define RD 		(1<<4)

#define LATCH_OE (1<<5)
#define BUSAK 	(1<<5)
// E2 IOREQ
#define SRAM_EN (1<<6)
#define IOREQ 	(1<<6)
// HC574 CLK
#define LATCH_CLK (1<<7)

void sram_init(void) {
	SRAM_CONTROL_DDR  |= ( SRAM_A16 | SRAM_EN | SRAM_OE | SRAM_WE | LATCH_CLK );
	SRAM_CONTROL_PORT |= ( SRAM_OE | SRAM_WE | LATCH_CLK );
	SRAM_CONTROL_PORT &= ~(SRAM_A16 | SRAM_EN );
	ADDR_DDR = 0xff;
	DATA_DDR = 0xff;
}

void sram_select(void) {
	SRAM_CONTROL_PORT |= SRAM_EN;  // E2
	// STAM_CONTROL_PORT &= ~SRAM_OE;  // ~E1 (G)
}

void sram_deselect(void) {
	SRAM_CONTROL_PORT &= ~SRAM_EN; // E2
}

void sram_addr_out(uint32_t addr) {
	ADDR = addr>>8 & 0xff;
	SRAM_CONTROL_PORT &= ~LATCH_CLK;
	SRAM_CONTROL_PORT |= LATCH_CLK;
	ADDR = addr & 0xff;
	SRAM_CONTROL_PORT &= ~SRAM_A16;
	SRAM_CONTROL_PORT |= (addr>>16 & 1) <<2;
}

void sram_data_out(uint8_t val) {
	DATA_DDR = 0xff;
	DATA_OUT = val;
	SRAM_CONTROL_PORT &= ~SRAM_WE;
	SRAM_CONTROL_PORT |= SRAM_WE;
}

uint8_t sram_data_in(void) {
	uint8_t val;
	DATA_DDR = 0x00;
	SRAM_CONTROL_PORT &= ~SRAM_OE;
	SRAM_CONTROL_PORT |= SRAM_OE;
	val = DATA_IN;
	return val;
}

uint8_t sram_read(uint32_t addr) {
	uint8_t val;
	sram_select();
	sram_addr_out(addr);
	val = sram_data_in();
	sram_deselect();
	return val;
}

void sram_write(uint32_t addr, uint8_t val) {
	sram_select();
	sram_addr_out(addr);
	sram_data_out(val);
	sram_deselect();

}


int main(void) {
	uint32_t addr = 0;
	uint8_t val;

	sram_init();
	uart_init(19200);
	stdout = &uartout;

	printf("Hello there.\n");

	//test();
	srand(10);

	for(;;) {
		addr = rand() & 0x1ffff;
		printf("address %05lx, \n",addr);

		for(uint32_t i = 0; i < 24; i++) {
			val = rand() & 0xff;
			sram_write(addr+i, val);
			printf("%02x ",val);
		}
		printf("\n");

		for(uint32_t i = 0; i < 24; i++) {
			val = sram_read(addr+i);
			printf("%02x ",val);
		}
		printf("\n\n");

		_delay_ms(1000);
	}
	return 0;
}

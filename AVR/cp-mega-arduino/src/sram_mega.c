#include "sram.h"

#include <avr/io.h>

/* SRAM I/F Port/Pin definitions */
#define LOW4_DIR DDRC
#define LOW4_OUT PORTC
#define LOW4_IN PINC
#define LOW4_MASK 0x0f
#define HIGH4_DIR DDRD
#define HIGH4_OUT PORTD
#define HIGH4_IN PIND
#define HIGH4_MASK 0xf0
#define LATCH_CONTROL_DIR DDRD
#define LATCH_CONTROL     PORTD
#define LATCH_L           (1<<PD2)
#define LATCH_H           (1<<PD3)

#define SRAM_CONTROL PORTB
#define SRAM_CONTROL_DIR DDRB
//#define SRAM_CS (1<<PD)
#define SRAM_CS 0
#define SRAM_OE (1<<PB0)
#define SRAM_WE (1<<PB1)

void sram_init() {
	LOW4_DIR |= LOW4_MASK;
	HIGH4_DIR |= HIGH4_MASK;

	SRAM_CONTROL_DIR |= ( SRAM_CS | SRAM_OE | SRAM_WE);
	SRAM_CONTROL |= ( SRAM_CS | SRAM_OE | SRAM_WE);
	LATCH_CONTROL_DIR |= (LATCH_L | LATCH_H);
	LATCH_CONTROL |= (LATCH_L | LATCH_H);
}

inline void addr_out(uint32_t addr) {
	LOW4_DIR |= LOW4_MASK;
	HIGH4_DIR |= HIGH4_MASK;

	LOW4_OUT &= ~LOW4_MASK;
	HIGH4_OUT &= ~HIGH4_MASK;
	LOW4_OUT |= LOW4_MASK & (uint8_t) addr;
	HIGH4_OUT |= HIGH4_MASK & (uint8_t) addr;
	LATCH_CONTROL &= ~LATCH_L;
	LATCH_CONTROL |= LATCH_L;

	addr >>= 8;
	LOW4_OUT &= ~LOW4_MASK;
	HIGH4_OUT &= ~HIGH4_MASK;
	LOW4_OUT |= LOW4_MASK & (uint8_t) addr;
	HIGH4_OUT |= HIGH4_MASK & (uint8_t) addr;
	LATCH_CONTROL &= ~LATCH_H;
	LATCH_CONTROL |= LATCH_H;

	// A16 and above will always be ignored
}

unsigned char sram_read(unsigned short addr) {
	unsigned char vall4, valh4;

	addr_out(addr);  // A16 is always 0

	LOW4_DIR &= ~LOW4_MASK;
	HIGH4_DIR &= ~HIGH4_MASK;

	SRAM_CONTROL &= ~SRAM_OE;
	__asm__ __volatile__("nop");
	vall4 = LOW4_IN;
	valh4 = HIGH4_IN;
	SRAM_CONTROL |= ~SRAM_OE;
	return (vall4 & LOW4_MASK) | (valh4 & HIGH4_MASK);
}

void sram_write(unsigned short addr, unsigned char data) {
	addr_out(addr);  // A16 is always 0

	LOW4_OUT &= ~LOW4_MASK;
	HIGH4_OUT &= ~HIGH4_MASK;
	LOW4_OUT |= data & LOW4_MASK;
	HIGH4_OUT |= data & HIGH4_MASK;
	SRAM_CONTROL &= ~SRAM_WE;
	SRAM_CONTROL |= SRAM_WE;

}

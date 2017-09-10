#include "sram.h"

#include <avr/io.h>

/* SRAM I/F Port/Pin definitions */
#define BUSL4_DDR DDRC
#define BUSL4_OUT PORTC
#define BUSL4_IN  PINC
#define BUSL4_MASK 0x0f
#define BUSH4_DDR DDRD
#define BUSH4_OUT PORTD
#define BUSH4_IN  PIND
#define BUSH4_MASK 0xf0

#define SRAM_CONTROL PORTB
#define SRAM_CONTROL_DIR DDRB
//#define SRAM_CS (1<<1)
#define SRAM_OE (1<<PB0)
#define SRAM_WE (1<<PB1)

#define LATCH_CONTROL PORTD
#define LATCH_CONTROL_DDR DDRD
#define LATCH_L (1<<PD2)
#define LATCH_H (1<<PD3)

void sram_init() {
	BUSL4_DDR |= BUSL4_MASK;
	BUSH4_DDR |= BUSH4_MASK;
	SRAM_CONTROL_DIR |= (SRAM_OE | SRAM_WE);
	SRAM_CONTROL |= (SRAM_OE | SRAM_WE);
	LATCH_CONTROL_DDR |= (LATCH_L | LATCH_H);
	LATCH_CONTROL |= (LATCH_L | LATCH_H);
}

inline void addr_out(uint32_t addr) {
	BUSL4_DDR |= BUSL4_MASK;
	BUSH4_DDR |= BUSH4_MASK;

	BUSL4_OUT &= ~BUSL4_MASK;
	BUSH4_OUT &= ~BUSH4_MASK;
	BUSL4_OUT |= addr & BUSL4_MASK;
	BUSH4_OUT |= addr & BUSH4_MASK;
	LATCH_CONTROL &= ~LATCH_L;
	LATCH_CONTROL |= LATCH_L;

	addr >>= 8;

	BUSL4_OUT &= ~BUSL4_MASK;
	BUSH4_OUT &= ~BUSH4_MASK;
	BUSL4_OUT |= addr & BUSL4_MASK;
	BUSH4_OUT |= addr & BUSH4_MASK;
	LATCH_CONTROL &= ~LATCH_H;
	LATCH_CONTROL |= LATCH_H;

	// ignore A16 and above
}


unsigned char sram_read(unsigned short addr) {
  unsigned char val;
//  SRAM_CONTROL &= ~SRAM_CS;
  addr_out(addr);  // A16 is always 0

  BUSL4_DDR &= ~BUSL4_MASK;
  BUSH4_DDR &= ~BUSH4_MASK;
  SRAM_CONTROL &= ~SRAM_OE;
  __asm__ __volatile__("nop");
  __asm__ __volatile__("nop");
  val = BUSL4_IN & BUSL4_MASK;
  val |= BUSH4_IN & BUSH4_MASK;
  SRAM_CONTROL |= SRAM_OE;
//  CONTROL |= SRAM_CS;
  return val;
}

void sram_write(unsigned short addr, unsigned char data) {
//  CONTROL &= ~SRAM_CS;
	addr_out(addr);  // A16 is always 0

	BUSL4_OUT &= ~BUSL4_MASK;
	BUSH4_OUT &= ~BUSH4_MASK;
	BUSL4_OUT |= (data & BUSL4_MASK);
	BUSH4_OUT |= (data & BUSH4_MASK);

	SRAM_CONTROL &= ~SRAM_WE;
	SRAM_CONTROL |= SRAM_WE;
//  CONTROL |= SRAM_CS;
}

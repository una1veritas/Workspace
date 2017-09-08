#include "sram.h"

void sram_init() {
  ADDRL_DIR = 0xff;
  ADDRH_DIR = 0xff;
  ADDRX_DIR |= ADDRX_MASK;
  DATA_DIR = 0x00;
  CONTROL_DIR |= ( SRAM_CS | SRAM_WE | SRAM_OE ); //| LATCH_L );
  CONTROL |= ( SRAM_CS | SRAM_WE | SRAM_OE ); //| LATCH_L );
}

inline void addr_out(uint32_t addr) {
  ADDRL = addr & 0xff;
  addr >>= 8;
  ADDRH = addr & 0xff;
  addr >>= 8;
  ADDRX &= ~ADDRX_MASK;
  ADDRX |= addr & ADDRX_MASK;
}


unsigned char sram_read(unsigned short addr) {
  unsigned char val;
  CONTROL &= ~SRAM_CS;
  addr_out(addr);  // A16 is always 0
  DATA_DIR = 0x00;
  CONTROL &= ~SRAM_OE;
  CONTROL |= ~SRAM_OE;
  val = DATA_IN;
  CONTROL |= SRAM_CS;
  return val;
}

void sram_write(unsigned short addr, unsigned char data) {
  CONTROL &= ~SRAM_CS;
  addr_out(addr);  // A16 is always 0
  DATA_DIR = 0xff;
  DATA_OUT = data;
  CONTROL &= ~SRAM_WE;
  CONTROL |= SRAM_WE;
  CONTROL |= SRAM_CS;
}

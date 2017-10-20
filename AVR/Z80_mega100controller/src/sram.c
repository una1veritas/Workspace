/*
 * sram.cpp
 *
 *  Created on: 2017/09/19
 *      Author: sin
 */

#include "types.h"
#include "sram.h"

#undef READ_AFTER_OE
#define READ_AFTER_NOP

void sram_bus_init() {
  ADDRL_DIR = ADDRL_MASK; // fixed to be OUTPUT
  ADDRH_DIR = ADDRH_MASK; // fixed to be OUTPUT
  ADDRX_DIR |= ADDRX_MASK; // fixed to be OUTPUT
  DATA_DIR = DATA_MASK;
  CONTROL_DIR |= ( SRAM_WE | SRAM_OE | SRAM_CS | SRAM_ALE );
  CONTROL |= ( SRAM_WE | SRAM_OE | SRAM_CS | SRAM_ALE);
}

void sram_bus_release() {
  ADDRL_DIR = 0x00l;
  ADDRH_DIR = 0x00;
  ADDRX_DIR &= ~ADDRX_MASK;
  DATA_DIR = 0x00;
  CONTROL_DIR &= ~( SRAM_WE | SRAM_OE | SRAM_CS | SRAM_ALE );
  CONTROL |= ( SRAM_CS ); // pull-up to avoid unexpected data change
}

void sram_enable() {
  CONTROL &= ~SRAM_CS;  
}

void sram_disable() {
  CONTROL |= SRAM_CS;
}

inline void addr_set32(uint32_t addr) {
  ADDRL = (uint8) addr;
  addr >>= 8;
  ADDRH = (uint8) addr;
  addr >>= 8;
  ADDRX &= ~ADDRX_MASK;
  ADDRX |= ((uint8)addr) & ADDRX_MASK;
}

inline void sram_bank_select(uint8_t bk) {
  ADDRX &= ~ADDRX_MASK;
  ADDRX |= (bk & ADDRX_MASK);
}

inline void addr_set16(uint16_t addr) {
  ADDRL = (uint8) addr;
  addr >>= 8;
  ADDRH = (uint8) addr;
  addr >>= 8;
}

uint8_t sram_read(uint32_t addr) {
  unsigned char val;
  addr_set32(addr);
  DATA_DIR = 0x00;
  //DATA_OUT = 0xff;
  CONTROL &= ~SRAM_OE;
#ifndef READ_AFTER_OE
#ifdef READ_AFTER_NOP
  __asm__ __volatile("nop");
#endif
  val = DATA_IN;
  CONTROL |= SRAM_OE; // valid data remains while
#else
  CONTROL |= SRAM_OE; // valid data remains while
  val = DATA_IN;
#endif
  return val;
}

void sram_write(uint32_t addr, uint8_t data) {
  addr_set32(addr);
  DATA_DIR = DATA_MASK;
  DATA_OUT = data;
  CONTROL &= ~SRAM_WE;
  CONTROL |= SRAM_WE;
}

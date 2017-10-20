#ifndef _SRAM_H_
#define _SRAM_H_

#include <avr/io.h>

#include "types.h"

/*
 * pins must be defined
 * in sketch
 *
 */
#include "sram_iodef.h"

/*
#define bitset(port, bv) (port) |= (bv)
#define bitclr(port, bv) (port) &= ~(bv)
*/

void sram_bus_init(void);
void sram_bus_release(void);
void sram_enable(void);
void sram_disable(void);
uint8 sram_read(uint32 addr);
void sram_write(uint32 addr, uint8 data);

void addr_set32(uint32 addr);
void sram_bank_select(uint8 bk);
void addr_set16(uint16 addr);


#endif /* _SRAM_H_ */

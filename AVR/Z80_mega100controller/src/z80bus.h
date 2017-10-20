/*
 * z80bus.h
 *
 *  Created on: 2017/09/29
 *      Author: sin
 */

#ifndef SRC_Z80BUS_H_
#define SRC_Z80BUS_H_

#include <avr/io.h>

#include "z80_iodef.h"
#include "types.h"

uint8 z80_bus_init(void);
uint8 z80_busreq(void);

void z80_reset(void);
uint8 z80_rd();
uint8 z80_wr();
uint8 z80_m1rd();
uint8 z80_in();
uint8 z80_out();

#endif /* SRC_Z80BUS_H_ */

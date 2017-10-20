/*
 * z80_iodef.h
 *
 *  Created on: 2017/09/29
 *      Author: sin
 */

#ifndef SRC_Z80_IODEF_H_
#define SRC_Z80_IODEF_H_

#include <avr/io.h>

#define Z80_CLK_PORT 	PORTB
#define Z80_CLK_DDR 	DDRB
#define Z80_CLK 		(1<<PB5)
#define Z80_CLK_PIN 	11

#define Z80_RESET_PORT 	PORTE
#define Z80_RESET_DDR 	DDRE
#define Z80_RESET 		(1<<PE5)
#define Z80_RESET_PIN 	3

#define Z80_BUSREQ_PORT PORTH
#define Z80_BUSREQ_DDR 	DDRH
#define Z80_BUSREQ 		(1<<PH5)
#define Z80_BUSREQ_PIN 	8

#define Z80_BUSACK_PORT PORTH
#define Z80_BUSACK_DDR 	DDRH
#define Z80_BUSACK 		(1<<PH6)
#define Z80_BUSACK_PIN 	9

#define Z80_M1_PORT 	PORTB
#define Z80_M1 			(1<<PB4)
#define Z80_M1_PIN 		10

/*
#define ALLOW_Z80_MREQ_PORT	PORTH
#define ALLOW_Z80_MREQ_DDR	DDRH
#define ALLOW_Z80_MREQ 		(1<<PH4)
#define ALLOW_Z80_MREQ_PIN 	7
*/

#define Z80_RD_PORT 	PORTG
#define Z80_RD 			(1<<PG1)
#define Z80_RD_PIN		40

#define Z80_WR_PORT 	PORTG
#define Z80_WR 			(1<<PG0)
#define Z80_WR_PIN		41

#define Z80_MREQ_PORT 	PORTG
#define Z80_MREQ 		(1<<PG5)
#define Z80_MREQ_PIN 	4

#define Z80_IORQ_PORT 	PORTE
#define Z80_IORQ 		(1<<PE3)
#define Z80_IORQ_PIN 	5

#define Z80_ADDRL_PORT 	PORTA
#define Z80_ADDRH_PORT 	PORTC
#define Z80_ADDRX_PORT 	PORTL
#define Z80_ADDRL_MASK 	0xff
#define Z80_ADDRH_MASK 	0xff
#define Z80_ADDRX_MASK 	(1<<PL0)
#define Z80_DATA_PORT 	PORTF
#define Z80_DATA_MASK	0xff

#endif /* SRC_Z80_IODEF_H_ */

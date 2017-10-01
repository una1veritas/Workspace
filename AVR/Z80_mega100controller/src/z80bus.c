/*
 * z80bus.c
 *
 *  Created on: 2017/09/29
 *      Author: sin
 */

#include <avr/io.h>
#include <util/delay.h>

#include "types.h"
#include "z80bus.h"
#include "sram.h"


uint8 z80_busfree(void) {
	uint32 waitcount = 500;

	OUTPUTMODE(ALLOW_Z80_MREQ_PORT, ALLOW_Z80_MREQ);
	BCLR(ALLOW_Z80_MREQ_PORT, ALLOW_Z80_MREQ);

	OUTPUTMODE(Z80_BUSREQ_PORT, Z80_BUSREQ);
	BSET(Z80_BUSREQ_PORT, Z80_BUSREQ);

	OUTPUTMODE(Z80_CLK_PORT, Z80_CLK);

	INPUTMODE(Z80_BUSACK_PORT, Z80_BUSACK);
	BSET(Z80_BUSACK_PORT, Z80_BUSACK);
	while ( !BTEST(Z80_BUSACK_PORT, Z80_BUSACK) && waitcount ) {
		_delay_ms(10);
		waitcount--;
	}
	return waitcount != 0;
}

void busmode_z80(void) {
	INPUTMODE(ADDRL, 0xff);
	INPUTMODE(ADDRH, 0xff);
	INPUTMODE(ADDRX, ADDRX_MASK);
	INPUTMODE(DATA_OUT, 0xff);
	INPUTMODE(CONTROL, ( SRAM_WE | SRAM_OE | SRAM_CS | SRAM_ALE ));
	BSET(CONTROL, SRAM_CS); // pull-up

	INPUTMODE(Z80_M1_PORT, Z80_M1);
	INPUTMODE(Z80_IORQ_PORT, Z80_IORQ);
	INPUTMODE(Z80_MREQ_PORT, Z80_MREQ);
	INPUTMODE(Z80_RD_PORT, Z80_RD);
	INPUTMODE(Z80_WR_PORT, Z80_WR);
}


uint8 z80_busreq() {
	uint32 waitcount = 500;

	OUTPUTMODE(ALLOW_Z80_MREQ_PORT, ALLOW_Z80_MREQ);
	BSET(ALLOW_Z80_MREQ_PORT, ALLOW_Z80_MREQ);

	OUTPUTMODE(Z80_CLK_PORT, Z80_CLK);

	OUTPUTMODE(Z80_BUSREQ_PORT, Z80_BUSREQ);
	BCLR(Z80_BUSREQ_PORT, Z80_BUSREQ);

	INPUTMODE(Z80_BUSACK_PORT, Z80_BUSACK);
	while ( BTEST(Z80_BUSACK_PORT, Z80_BUSACK) && waitcount ) {
		_delay_ms(10);
		waitcount--;
	}
	return  waitcount != 0;
}

void z80_reset(void) {
	OUTPUTMODE(Z80_RESET_PORT, Z80_RESET);
//	_delay_ms(500);
	BCLR(Z80_RESET_PORT, Z80_RESET);
	_delay_ms(500);
	BSET(Z80_RESET_PORT, Z80_RESET);
}


uint8 z80_m1() {
	return BTEST(PIN(Z80_M1_PORT), Z80_M1) || BTEST(PIN(Z80_RD_PORT), Z80_RD) || BTEST(PIN(Z80_MREQ_PORT), Z80_MREQ);
}

uint8 z80_rd() {
	return BTEST(PIN(Z80_RD_PORT), Z80_RD) || BTEST(PIN(Z80_MREQ_PORT), Z80_MREQ);
}

uint8 z80_wr() {
	return BTEST(PIN(Z80_WR_PORT), Z80_WR) || BTEST(PIN(Z80_MREQ_PORT), Z80_MREQ);
}

uint8 z80_in() {
	return BTEST(PIN(Z80_RD_PORT), Z80_RD) || BTEST(PIN(Z80_IORQ_PORT), Z80_IORQ);
}

uint8 z80_out() {
	return BTEST(PIN(Z80_WR_PORT), Z80_WR) || BTEST(PIN(Z80_IORQ_PORT), Z80_IORQ);
}

/*
void init_z80io() {
  pinMode(Z80_RESET_PIN, OUTPUT);
  digitalWrite(Z80_RESET_PIN, HIGH);

  pinMode(Z80_INT_PIN, OUTPUT);
  digitalWrite(Z80_INT_PIN, HIGH);
  pinMode(Z80_NMI_PIN, OUTPUT);
  digitalWrite(Z80_NMI_PIN, HIGH);
  pinMode(Z80_WAIT_PIN, OUTPUT);
  digitalWrite(Z80_WAIT_PIN, HIGH);
  pinMode(Z80_BUSREQ_PIN, OUTPUT);
  digitalWrite(Z80_BUSREQ_PIN, HIGH);

}

void z80io_reset() {
  digitalWrite(Z80_RESET_PIN, LOW);
  delay(500);
  digitalWrite(Z80_RESET_PIN, HIGH);
}

bool z80_busreq() {
  uint16_t climit = 1000;
  digitalWrite(Z80_BUSREQ_PIN, LOW);
  while (digitalRead(Z80_BUSACK_PIN) && climit > 0) {
    delay(5);
    --climit;
  }
  return digitalRead(Z80_BUSACK_PIN) == 0;
}

bool z80_busactivate() {
  digitalWrite(Z80_BUSREQ_PIN, HIGH);
  return digitalRead(Z80_BUSACK_PIN) == 0;
}
*/

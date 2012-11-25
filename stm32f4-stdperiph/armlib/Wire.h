/*
 * Wire.h
 *
 *  Created on: 2012/11/25
 *      Author: sin
 */

#ifndef WIRE_H_
#define WIRE_H_

#include "i2c.h"

class TwoWire {
	byte address;
	byte buffer[32];
	uint16_t length;
public:
	void begin() {
		i2c_begin(100000); //I2C_TypeDef * I2Cx, uint32_t clk);
	}

	void beginTransmission(byte addr) {
		address = addr;
		length = 0;
	}

	void send(byte data) {
		buffer[length++] = data;
	}

	void endTransmission() {
		i2c_transmit(address, buffer, length);
	}
};

extern TwoWire Wire;

#endif /* WIRE_H_ */

/*
 * bcd.h
 *
 *  Created on: 2019/03/03
 *      Author: sin
 */

#ifndef SRC_BCD_H_
#define SRC_BCD_H_

#include <cinttypes>

class bcd {
	typedef uint64_t uint64;
	typedef uint16_t uint16;
	uint64 significand;
	uint16 exponent;

	bcd(void) : significand(0), exponent(0) {}
	bcd(uint64 & val) : significand(val), exponent(0) {}

	bcd operator+(bcd & a, bcd & b) {
		return bcd(a.significand + b.significand);
	}


};



#endif /* SRC_BCD_H_ */

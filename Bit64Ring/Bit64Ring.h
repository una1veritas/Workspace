/*
 * Bit64Ring.h
 *
 *  Created on: Feb 3, 2019
 *      Author: sin
 */

#ifndef BIT64RING_H_
#define BIT64RING_H_

#include <iostream>
#include <cinttypes>
#include <cstdio>

class Bit64Ring {
	uint64_t bitring;

	inline static uint64_t rotl64(uint64_t x) {
		// assert (n<32);
		return (x<<1) | (x>>(-1&63));
	}

	inline static uint64_t rotl64(uint64_t x, uint64_t n) {
		// assert (n<32);
		return (x<<n) | (x>>(-n & 63));
	}

public:
	Bit64Ring(const uint64_t & initval = 0) {
		bitring = 0;
	}

	void enqueue(int bit) {
		bitring = rotl64(bitring) | (bit ? 1 : 0);
	}

	//
	friend std::ostream & operator<<(std::ostream & stream, const Bit64Ring & b64ring) {
		uint64_t bits = b64ring.bitring;
		for (unsigned int i = 64; i > 0; --i) {
			bits = rotl64(bits);
			stream << (int)(bits & 1) << " ";
		}
		return stream;
	}
};

#endif /* BIT64RING_H_ */

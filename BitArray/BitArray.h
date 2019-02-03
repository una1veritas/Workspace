/*
 * BitArray.h
 *
 *  Created on: Feb 3, 2019
 *      Author: sin
 */

#ifndef BITARRAY_H_
#define BITARRAY_H_

#include <cinttypes>

class BitArray {
	uint64_t * u64array;
	unsigned int u64capacity;

	uint64_t cache;
	unsigned int cache_rot;

public:
	BitArray(const unsigned int & bitsize) {
		u64capacity = (bitsize>>6) + (bitsize & 0x3f ? 1 : 0);
		u64array = new uint64_t[u64capacity];

		cache = 0;
		cache_rot = 0;
	}

	~BitArray() {
		delete [] u64array;
		u64capacity = 0;
	}
};



#endif /* BITARRAY_H_ */

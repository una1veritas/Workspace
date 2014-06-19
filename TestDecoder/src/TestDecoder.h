/*
 * TestCode.h
 *
 *  Created on: 2014/06/18
 *      Author: sin
 */

#ifndef TESTCODE_H_
#define TESTCODE_H_

#ifdef _cc_
typedef uint8_t boolean;
enum {
	false = 0,
	true = 0xff,
};
#else
typedef bool boolean;
#endif

typedef unsigned int uint;

struct Permutation {
	uint permsize, anchor;
	uint perm[128];

	Permutation() {
		permsize = 0;
		anchor = 0;
	}

	Permutation(uint size, uint limit = 0) {
		init(size, limit);
	}

	void init(uint size, uint limit = 0) {
		anchor = limit;
		permsize = size;
		for(uint i = 0; i < permsize; i++)
			perm[i] = i;
	}

	boolean next();
	uint operator[](const uint i) const {
		return perm[i];
	}
};

struct Mapping {
	char alphabet[128];
	unsigned int size;
	Permutation transfer;

	static const char terminator = '~';

	Mapping() {
		set("abcdefghijklmnopqrstuvwxyz");
	}

	void set(const char a[], const uint lim = 0);
	void translate(char str[]) const;
	char operator[](const char c) const {
		uint t;
		for(t = 0; t < size ; t++)
			if ( c == alphabet[t] )
				return alphabet[transfer[t]];
		return c;
	}
};

void asort(uint array[], uint s, uint n );
void dsort(uint array[], uint s, uint n );




#endif /* TESTCODE_H_ */

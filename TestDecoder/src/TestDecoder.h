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

class Permutation {
	uint length;
	uint perm[128];

public:
	Permutation() {
		length = 0;
	}

	Permutation(uint sz) {
		init(sz);
	}

	void init(uint sz) {
		length = sz;
		for(uint i = 0; i < length; i++)
			perm[i] = i;
	}

	boolean next(const uint startpos = 0);
	const uint & operator[](const uint i) const {
		return perm[i];
	}
	uint & operator[](const uint i) {
		return perm[i];
	}
};

class Mapping {
	char charfrom[128];
	char charto[128];
	int charmap[256];
	uint alphsize;
	Permutation rotor;
	uint permlimit;

public:
	static const char terminator = '~';

	Mapping() {
		init("abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz");
	}

	void init(const char a[], const char r[], const uint lim = 0);

	const char * domain() const { return charfrom; }
	const char * range() const { return charto; }
	const uint alphabetSize() const { return alphsize; }

	void translate(char str[]) const;
//	char operator[](const char c) const;
//	uint order(const char c) const;
//	void setTranslate(const char orig[], const char trans[]);

	boolean next();
};

void asort(uint array[], uint s, uint n );
void dsort(uint array[], uint s, uint n );


#endif /* TESTCODE_H_ */

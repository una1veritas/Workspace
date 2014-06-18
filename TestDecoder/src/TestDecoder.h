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

struct Mapping {
	char alphabet[128];
	unsigned int size;
	int transfer[128];

	Mapping() {
		set("abcdefghijklmnopqrstuvwxyz");
	}

	void set(const char a[]);
	boolean hasNext();
	boolean next();
	void translate(char str[]) const;
};

void asort(int array[], int s, int n );
void dsort(char array[], int s, int n );




#endif /* TESTCODE_H_ */

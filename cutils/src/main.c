/*
 * main.c
 *
 *  Created on: 2017/01/22
 *      Author: sin
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cutils.h"

uint32 dummy(uint32 x) {
	return x;
}

uint32 clz0(uint32 x) {
	int count;
	uint32_t bit = 0x80000000;
	for(count = 0; bit != 0; ++count, bit >>= 1) {
		if ( (x & bit) != 0 )
			break;
	}
	return count;
}

int main(void) {
	uint32 r[1000000];
	time_t lap;
	long dummysum;

	printf("Hello.\n");

	srand(113);
	for(int i = 0; i < 1000000; i++)
		r[i] = rand();

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += dummy(r[i]);
	}
	lap = clock() - lap;
	printf("dummy\n");
	printf("lap = %ld, sum = %ld\n", (long) lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += clz(r[i]);
	}
	lap = clock() - lap;
	printf("clz:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += clz0(r[i]);
	}
	lap = clock() - lap;
	printf("clz0\n");
	printf("lap = %ld, sum = %ld\n", (long) lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += ceil2pow(r[i]);
	}
	lap = clock() - lap;
	printf("ceil2pow:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	return EXIT_SUCCESS;
}


/*
 ============================================================================
 Name        : cutils.c
 Author      : Sin Shimozono
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cutils.h"

uint32 clz0(uint32 x) {
	int count;
	uint32_t bit = 0x80000000;
	for(count = 0; bit != 0; ++count, bit >>= 1) {
		if ( (x & bit) != 0 )
			break;
	}
	return count;
}

uint32 clz(uint32 x) {
	double d = x;
	uint32 *p = ((uint32*) &d) + 1;
	if ( x == 0 )
		return 32;
	return 0x41e - (*p>>20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}

uint32 dummy(uint32 x) {
	return x;
}

uint32 ceil2pow(uint32 x) {
	if (x == 0)
		return 0;
	return 1<<(32 - clz(x-1));

}

int main(void) {
	uint32 r[1000000];
	time_t lap;
	long dummysum;

	for(int i = 0; i < 1000000; i++)
		r[i] = random();

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += dummy(r[i]);
	}
	lap = clock() - lap;
	printf("dummy\n");
	printf("lap = %ld, sum = %ld\n",lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += clz(r[i]);
	}
	lap = clock() - lap;
	printf("clz:\n");
	printf("lap = %ld, sum = %ld\n",lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += clz0(r[i]);
	}
	lap = clock() - lap;
	printf("clz0\n");
	printf("lap = %ld, sum = %ld\n",lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < 1000000; i++) {
		dummysum += ceil2pow(r[i]);
	}
	lap = clock() - lap;
	printf("ceil2pow:\n");
	printf("lap = %ld, sum = %ld\n",lap, dummysum);

	return EXIT_SUCCESS;
}

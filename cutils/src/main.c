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


int main(void) {
	const int NUM = 2000000;
	uint32 r[NUM];
	time_t lap;
	long dummysum;

	int32 x = 2;
	printf("Hello. %04x, %04ux\n", -x, x);

	srand(113);
	for(int i = 0; i < NUM; i++)
		r[i] = rand();

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += dummy(r[i]);
	}
	lap = clock() - lap;
	printf("dummy\n");
	printf("lap = %ld, sum = %ld\n", (long) lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += clz(r[i]);
	}
	lap = clock() - lap;
	printf("clz:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += clz0(r[i]);
	}
	lap = clock() - lap;
	printf("clz0\n");
	printf("lap = %ld, sum = %ld\n", (long) lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += ceil2pow(r[i]);
	}
	lap = clock() - lap;
	printf("ceil2pow:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += popc0(r[i]);
	}
	lap = clock() - lap;
	printf("popc0:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += popc(r[i]);
	}
	lap = clock() - lap;
	printf("popc:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += ctz(r[i]);
	}
	lap = clock() - lap;
	printf("ctz:\n");
	printf("lap = %ld, sum = %ld\n", (long)lap, dummysum);

	return EXIT_SUCCESS;
}


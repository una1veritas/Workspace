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
	printf("Hello. %08x, %08x\n", -x, x);

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
	printf("lap = %ld, sum = %ld\n\n", (long) lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += nlz32_(r[i]);
	}
	lap = clock() - lap;
	printf("nlz32_:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += nlz32_IEEE(r[i]);
	}
	lap = clock() - lap;
	printf("nlz32_IEEE\n");
	printf("lap = %ld, sum = %ld\n\n", (long) lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += ceil2pow32(r[i]);
	}
	lap = clock() - lap;
	printf("ceil2pow32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += pop32_(r[i]);
	}
	lap = clock() - lap;
	printf("pop32_:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += pop32_SSE42(r[i]);
	}
	lap = clock() - lap;
	printf("pop32_SSE42:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		dummysum += ntz32(r[i]);
	}
	lap = clock() - lap;
	printf("ntz32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		dummysum += (r[i] >= r[i+1] ? r[i] : r[i+1] );
	}
	lap = clock() - lap;
	printf("max ( ? : ):\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		dummysum += min32(r[i], r[i+1]);
	}
	lap = clock() - lap;
	printf("min32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		dummysum += MIN(r[i], r[i+1]);
	}
	lap = clock() - lap;
	printf("MIN:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	int32 a, b, t;
	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		a = r[i];
		b = r[i+1];
		//
		t = a;
		a = b;
		b = t;
		dummysum += b;
	}
	lap = clock() - lap;
	printf("swap by temp:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		a = r[i];
		b = r[i+1];
		//
		SWAP(a,b);
		dummysum += b;
	}
	lap = clock() - lap;
	printf("SWAP:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	dummysum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		a = r[i];
		b = r[i+1];
		//
		swap32(&a, &b);
		dummysum += b;
	}
	lap = clock() - lap;
	printf("swap32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, dummysum);

	return EXIT_SUCCESS;
}


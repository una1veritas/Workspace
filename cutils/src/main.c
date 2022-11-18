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

uint32_t dummy(uint32_t x) {
	return x;
}


int main(int argc, char * argv[]) {
	const int NUM = 1000000;
	int r[NUM];
	time_t lap;
	unsigned long sum;
	int seed = 113;

	printf("%u\n", clog32(57));

	if ( argc == 2 )
		seed = atol(argv[1]);

	srand(seed);
	for(int i = 0; i < NUM; i++)
		r[i] = rand();

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += dummy(r[i]);
	}
	lap = clock() - lap;
	printf("dummy\n");
	printf("lap = %ld, sum = %ld\n\n", (long) lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += nlz32_IEEEFP(r[i]);
	}
	lap = clock() - lap;
	printf("nlz32 IEEEFP:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += nlz32_ABM(r[i]);
	}
	lap = clock() - lap;
	printf("nlz32 ABM:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += nlz32_Harley(r[i]);
	}
	lap = clock() - lap;
	printf("nlz32_Harley:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += c2pow32(r[i]);
	}
	lap = clock() - lap;
	printf("c2pow32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += pop32_HD(r[i]);
	}
	lap = clock() - lap;
	printf("pop32_HD:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += pop32_SSE42(r[i]);
	}
	lap = clock() - lap;
	printf("pop32 SSE42:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += ntz32_HD(r[i]);
	}
	lap = clock() - lap;
	printf("ntz32_HD:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += ntz32_BMI1(r[i]);
	}
	lap = clock() - lap;
	printf("ntz32_BMI1:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += ntz32_pop32(r[i]);
	}
	lap = clock() - lap;
	printf("ntz32_pop32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM; i++) {
		sum += ntz32_nlz32(r[i]);
	}
	lap = clock() - lap;
	printf("ntz32_nlz32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		sum += (r[i] <= r[i+1] ? r[i] : r[i+1] );
	}
	lap = clock() - lap;
	printf("min ( ? : ):\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		sum += min32(r[i], r[i+1]);
	}
	lap = clock() - lap;
	printf("min32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		sum += MIN(r[i], r[i+1]);
	}
	lap = clock() - lap;
	printf("MIN:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	int32_t a, b, t;
	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		a = r[i];
		b = r[i+1];
		//
		t = a;
		a = b;
		b = t;
		sum += b;
	}
	lap = clock() - lap;
	printf("swap by temp:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		a = r[i];
		b = r[i+1];
		//
		SWAP(a,b);
		sum += b;
	}
	lap = clock() - lap;
	printf("SWAP:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	sum = 0;
	lap = clock();
	for(int i = 0; i < NUM-1; i++) {
		a = r[i];
		b = r[i+1];
		//
		swap32_(&a, &b);
		sum += b;
	}
	lap = clock() - lap;
	printf("swap32:\n");
	printf("lap = %ld, sum = %ld\n\n", (long)lap, sum);

	return EXIT_SUCCESS;
}


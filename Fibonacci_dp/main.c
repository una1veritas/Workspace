/*
 * main.c
 *
 *  Created on: 2013/12/12
 *      Author: sin
 */

/*
 * Hi.
input = 37
result = 39088169
.fibo called 78176337 time.
result by dp = 24157817
.fibo_dp looped 35 time.
 *
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

uint32_t counter_fibo = 0;
uint32_t counter_fibo_dp = 0;

uint32_t fibo(uint32_t n) {
	counter_fibo++;

	if ( n == 1 )
		return 1;
	if ( n == 2 )
		return 1;
	return fibo(n-1) + fibo(n-2);
}

uint32_t fibo_dp(uint32_t n) {
	uint32_t d = 1;
	uint32_t e = 1;
	uint32_t f;

	if ( n < 2 )
		return e;
	uint32_t i;
	for(i = 2; i < n; i++) {
		counter_fibo_dp++;

		f = d + e;
		d = e;
		e = f;
	}
	return f;
}

int main(int argc, char * argv[]) {

	uint32_t n, result;

	printf("Hi.\n");

	n= atol(argv[1]);
	printf("input = %d\n", n);

	result = fibo(n);
	printf("result = %d\n.", result);
	printf("fibo called %d time.\n", counter_fibo);

	result = fibo_dp(n);
	printf("result by dp = %d\n.", result);
	printf("fibo_dp looped %d time.\n", counter_fibo_dp);

	return 0;
}



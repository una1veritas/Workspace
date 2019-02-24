/*
 * sieve.c
 *
 *  Created on: 2017/07/19
 *      Author: sin
 */

#include <iostream>

#include "sieve.h"

long long sieve(const long long & sieve_size) {
	long long i, l, n;
	long long bit_mask, bit_shifts;
	long long lastprime = 2;

	if ( sizeof(long long) == 8) {
		bit_shifts = 6;
		bit_mask = 63;
	} else if ( sizeof(long long) == 4) {
			bit_shifts = 5;
			bit_mask = 31;
	} else {
		std::cout << "size of long long does not match the size expected." << std::endl;
		return -1;
	}

	long long wordarray_size = ((sieve_size>>bit_shifts) > 1 ? (sieve_size>>bit_shifts) : 1);
	long long * wordarray = new long long[wordarray_size];

	//std::cout << wordarray.size() << std::endl;
	for(long long i = 0; i < wordarray_size; ++i) {
		wordarray[i] = ~((long long)0); 	// all set
	}
	std::cout << std::endl;

	wordarray[0>>bit_shifts] &= ~((long long)1)<<(0 & bit_mask); // 0 no use, must be ignored, bit clear
	wordarray[1>>bit_shifts] &= ~((long long)1)<<(1 & bit_mask); // 1 is not prime, bit clear
	//std::cout << std::hex << wordarray[0] << ", ";

	for (i = 2; i < sieve_size; i++) {
		for (l = 2; (n = i*l) < sieve_size; l++) {
			wordarray[n>>bit_shifts] &= ~(((long long)1)<<(n & bit_mask));
			//std::cout << std::hex << wordarray[0] << ", ";
		}
		//std::cout << std::endl;
	}

	for (i = sieve_size - 1; i > 0; --i) {
		if ( wordarray[i>>bit_shifts] & ((long long)1)<<(i & bit_mask) )
			break;
	}
	lastprime = i;
	delete [] wordarray;

	return lastprime;
}

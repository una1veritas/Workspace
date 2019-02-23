/*
 * sieve.c
 *
 *  Created on: 2017/07/19
 *      Author: sin
 */

#include <iostream>
#include <vector>

#include <time.h>
#include <cstdlib>
#include <cinttypes>

typedef unsigned long ulong;

const ulong LIMIT = 0x8000000UL;

int main(int argc, char * argv[]) {
	ulong sieve_size;
	if ( argc == 1 ) {
		sieve_size = LIMIT;
	} else {
		sieve_size = atol(argv[1]);
		sieve_size = sieve_size < LIMIT ? sieve_size : LIMIT;
	}
	ulong i, l, n;
	ulong ulong_nbits, ulong_mask, ulong_shifts;

	std::cout << "Sieve size : " << sieve_size << ", " << std::endl;
	std::cout << "The bibt-size of ulong : " << sizeof(ulong)*8 << ". " << std::endl;

	if ( sizeof(ulong) == 8) {
		ulong_nbits = 64;
		ulong_shifts = 6;
		ulong_mask = 63;
	} else {
		std::cout << "size of ulong does not match the expected." << std::endl;
		return EXIT_FAILURE;
	}
	std::vector<ulong> wordarray((sieve_size>>ulong_shifts) > 1 ? (sieve_size>>ulong_shifts) : 1);

	//std::cout << wordarray.size() << std::endl;
	for(ulong i = 0; i < wordarray.size(); ++i) {
		wordarray[i] = ~((ulong)0); 	// all set
	}
	std::cout << std::endl;

	wordarray[0>>ulong_shifts] &= ~((ulong)1)<<(0 & ulong_mask); // 0 no use, must be ignored, bit clear
	wordarray[1>>ulong_shifts] &= ~((ulong)1)<<(1 & ulong_mask); // 1 is not prime, bit clear
	//std::cout << std::hex << wordarray[0] << ", ";
	std::cout << "Sieve initialized." << std::endl;

	clock();
	for (i = 2; i < sieve_size; i++) {
		for (l = 2; (n = i*l) < sieve_size; l++) {
			wordarray[n>>ulong_shifts] &= ~(((ulong)1)<<(n & ulong_mask));
			//std::cout << std::hex << wordarray[0] << ", ";
		}
		//std::cout << std::endl;
	}
	  printf("sieve loop finished.\n");
	std::cout << ((double)clock()/CLOCKS_PER_SEC) << " sec." << std::endl;

	n = 0;
  for (i = sieve_size - 1; i > 0 && n < 100; --i) {
	  if ( wordarray[i>>ulong_shifts] & ((ulong)1)<<(i & ulong_mask) ) {
		  ++n;
		  std::cout << i << ", ";
	  }
  }
  std::cout << std::endl;
  printf("detection finished.\n");

  return EXIT_SUCCESS;
}

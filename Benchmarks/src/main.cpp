/*
 * sieve.c
 *
 *  Created on: 2017/07/19
 *      Author: sin
 */

#include <iostream>
#include <iomanip>
#include <vector>

#include <time.h>
#include <cstdlib>
#include <cinttypes>

//#include "bcd.h"
#include "pi.h"
#include "sieve.h"

const long long LIMIT = 0x40000000UL;

int main(int argc, char * argv[]) {
	long long limit_size, largest_prime;
	long double pi;

	if ( argc == 1 ) {
		limit_size = LIMIT;
	} else {
		limit_size = atol(argv[1]);
		limit_size = limit_size < LIMIT ? limit_size : LIMIT;
	}

	std::cout << "Sieve size : " << limit_size << ", " << std::endl;
	std::cout << "long long size (bits) : " << sizeof(long long)*8 << ". " << std::endl;

	clock();
	largest_prime = sieve(limit_size);
	std::cout << ((double)clock()/CLOCKS_PER_SEC) << " sec." << std::endl;
	std::cout << "The last prime = " << largest_prime << std::endl;

	std::cout << "Pi by Leibniz_pi " << limit_size << " loops." << std::endl;
	clock();
	pi = Leibniz_pi(limit_size);
	std::cout << ((double)clock()/CLOCKS_PER_SEC) << " sec." << std::endl;
	std::cout << "Pi = " << std::fixed << std::setprecision(16) << pi << std::endl;

	std::cout << "Pi by Wallis_pi " << limit_size << " loops." << std::endl;
	clock();
	pi = Wallis_pi(limit_size);
	std::cout << ((double)clock()/CLOCKS_PER_SEC) << " sec." << std::endl;
	std::cout << "Pi = " << std::fixed << std::setprecision(16) << pi << std::endl;

	return EXIT_SUCCESS;
}

/*
 *
Sieve size : 200000000,
long long size (bits) : 64.

16.3871 sec.
The last prime = 199999991
Pi by Leibniz_pi 200000000 loops.
17.3201 sec.
Pi = 3.1415926535897957
Pi by Wallis_pi 200000000 loops.
18.2517780000000016 sec.
Pi = 3.1415926496629787

 *
 *
 */

//============================================================================
// Name        : picalc.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <iomanip>
using namespace std;

int main(const int argc, const char * const argv[]) {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	long n = 1000; // default
	char * ptr;
	if ( argc >= 2 ) {
		long t = strtol(argv[1], &ptr, 10);
		if ( ptr != argv[1] )
			n = t;
	}
	cout << "n = " << n << endl;

	long double psum = 4.0;
	long double prev;
	long double s = - 4.0;

	for(long i = 1; i <= n; ++i) {
		prev = psum;
		psum = psum + s/(2*i+1);
		s = -s;
		if ( i % 1000 == 0 ) {
			cout << "i: "<< i << " pi = " << fixed << setprecision(16) << (prev + psum)/2 << endl;
		}
	}
	cout << "-----------" << endl;

	return EXIT_SUCCESS;
}

//============================================================================
// Name        : ModPower.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

long modpow(long x, long p, long m);

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	cout << " modpow(2, 7, 31) = " << modpow(2, 7, 31) << endl << endl;

	return 0;
}

long modpow(long x, long p, long m) {
	long t = 1;
	long xp = x;
	for ( ; p > 0 ; p>>=1) {
		if ( p % 1 ) {
			t = t * xp;
			t %= m;
		}
		xp *= xp;
	}
	return t;
}

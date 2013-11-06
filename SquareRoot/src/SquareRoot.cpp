//============================================================================
// Name        : SquareRoot.cpp
// Author      : Sin
// Version     : 1
// Copyright   :
// Description : Calculate the square root without explicit multiplication
//============================================================================

#include <iostream>
using namespace std;

unsigned long sqrt(unsigned long sqr) {
	unsigned long root = 0;
	unsigned long sum = 0;
	do {
		root++;
		sum += (root<<1) - 1;
		if ( sum == sqr)
			break;
	} while ( sum < sqr);
	return root;
}

int main(int argc, char * argv[]) {
	unsigned long sqr, nthpower;
	unsigned int fracdigits;

	if ( !(argc > 1) ) {
		cout << "argument integer is required." << endl;
		return 0;
	}
 	sqr = atol(argv[1]);
	cout << "Input = " << sqr << endl;
	if ( argc > 2 ) {
		fracdigits = atoi(argv[2]);
		cout << "Fractional digits = " << fracdigits << endl;
	} else {
		fracdigits = 0;
	}

	nthpower = 1;
	if ( fracdigits > 0 ) {
		for(int i = 0; i < fracdigits; i++) {
			nthpower = nthpower * 10;
		}
	}
	sqr = sqr * nthpower * nthpower;

	unsigned long root = sqrt(sqr);

	cout << "Answer: the squre root of " << sqr/nthpower/nthpower;
	if ( root*root == sqr )
		cout << " equals " << root/nthpower << "." << endl;
	else
		cout << " is between " << (root - 1)/(double)nthpower << " and " << (double)root/nthpower << "." << endl;
	cout << endl;
	return 0;
}

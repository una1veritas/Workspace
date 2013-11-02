//============================================================================
// Name        : SquareRoot.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int main(int argc, char * argv[]) {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	if ( !(argc > 1) ) {
		cout << "argument integer is required." << endl;
		return 0;
	}

	cout << "size of long is " << sizeof(unsigned long) << "." << endl;

 	unsigned long sqr = atol(argv[1]);
	cout << "Val = " << sqr << endl;

	unsigned long root = 0;
	unsigned long sum = 0;
	do {
		root++;
		sum += 2*root-1;
//		cout << root << " -> " << sum << endl;
		if ( sum == sqr)
			break;
	} while ( sum < sqr);
	cout << "Answer: the squre root of " << sqr;
	if ( sum == sqr )
		cout << " equals to " << root << "." << endl;
	else
		cout << " is between " << root - 1 << " and " << root << "." << endl;
	cout << endl;
	return 0;
}

//============================================================================
// Name        : QuickTest.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

typedef uint8_t uint8;
typedef uint16_t uint16;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	uint16 hist[16+1];
	uint16 bits;
	for(int i = 0; i < 16+1; i++) {
		hist[i] = 0;
	}

	srandom(time(NULL));
	for (uint16 t = 0; t < 300; t++) {
		bits = t;
		bits = random() % 0x10000;
		int cZero = 0, cOne = 0;
		for (int p = 0; p < 16; p++) {
			if (bits >> p & 1) {
				cOne++;
			} else {
				cZero++;
			}
		}
		// cout << cOne << " bits are 1, and " << cZero << " bits are 0." << endl;
		hist[cOne]++;
	}

	for(int i = 0; i <= 16; i++) {
		cout << i << ",\t" << hist[i] << "\t";
		for(int c = 0; c < hist[i]; c++) {
			cout << '*';
		}
		cout << endl;
	}

	return 0;
}

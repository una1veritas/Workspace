//============================================================================
// Name        : FixedMath_Test.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <inttypes.h>
#include <math.h>
#include <iostream>
using namespace std;

typedef uint32_t	uint32;
typedef int32_t 	int32;

struct fixedpoint {
	int32 intpart; 		// 32 lsb-bits of the 33-bit fractional part
	uint32 fracpart; 	// 30 bits integerpart and the msb of the 33 bit fractional part

	fixedpoint(void) : intpart(0), fracpart(0) {}
	fixedpoint(const double & val) {
		double i, f;
		f = modf((val < 0 ? -val: val), &i);
		intpart = (int32) i;
		fracpart = 0;
		for(int bit = 0; bit < 32; bit++) {
			f += f;
			if ( f >= 1.0f ) {
				fracpart |= 1<<(31-bit);
				f -= 1.0f;
			}
		}
	}

	ostream & printOn(ostream & out) const {
		uint32 tmp = 0, half = 500000000UL;
		out << intpart << ".";

		for(int bit = 31; bit >= 0; bit--) {
			if ( fracpart & (1<<bit) ) {
				tmp += half;
			}
			half >>= 1;
		}
		out << tmp << " ";
		return out;
	}

	friend ostream & operator<<(ostream & out, const fixedpoint & fp) {
		return fp.printOn(out);
	}
};

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	fixedpoint x(38), y(3.1415926339887);

	cout << "x = " << x << ", " << endl;
	cout << "y = " << y << ". " << endl;

	return 0;
}

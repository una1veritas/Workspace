//============================================================================
// Name        : Bitrotate.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

#include "font.h"

int main() {
	int ch, r, c;
	int width = font[2];
	int height = font[3];
	int bytesperchar = ceil(width/8) * height;
	uint8_t * bitmap = font + 6;
	uint32_t setbit, colbits[width];

	cout << "width " << width << ", height " << height << ", "
			<< bytesperchar << " bytes per letter." << endl;
	for(ch = 0; ch < font[5]; ch++) {
		for (c = 0; c < width; c++) {
			colbits[c] = 0;
			setbit = 1<<height;
			for (r = height-1; r >= 0; r--) {
				setbit >>= 1;
				if ( bitmap[ch*bytesperchar + r]>>(7-c) & 1 ) {
					colbits[c] |= setbit;
				}
			}
			colbits[c] >>= 1; // move bits to lower
		}
		for(r = 0; r < height; r += 8) {
			for(c = 0; c+2 < width; c++) {
				cout << "0x" << std::setw(2) << std::setfill('0')
					 << std::hex << ((colbits[c]>>r) & 0xff) << ", ";
			}
		}
		cout << " // " << (char) (ch+font[4]) << endl;
	}

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}

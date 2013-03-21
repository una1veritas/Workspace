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

#include "Chicago12.h"

int main() {
	int maxwidth = 20;
	int maxheight = 30;
	int basechar = 0x20;
	int numchar = 0x60;
	int bytespercolumn = ceil((float)maxheight/8);
	int bytesperchar = bytespercolumn * maxwidth;
	const uint8_t * bitmap = pix_Chicago20x30; // + 6;

//	cout << sizeof(Chicago10x12) << endl;
	cout << "max width " << maxwidth << ", max height " << maxheight << ", "
			<< bytesperchar << " bytes per letter." << endl;

	int fontdatapos = 0;
	int fontwidth;
	uint32_t colbits[maxwidth];
	uint8_t sizes[numchar+1];

	for(int i = 0; i < numchar; i++) {
		cout << "// "<< hex << (basechar+i);
		fontdatapos = i * (bytesperchar + 1);
		fontwidth = bitmap[fontdatapos];
		cout << " (width " << fontwidth << ")" << endl;
//		cout << "0x" << setw(2) << setfill('0')
//							 << hex << (fontwidth>>1) << ", " << endl;
		sizes[i] = fontwidth>>1;
		for(int col = 0 ; col < fontwidth; col++) {
			colbits[col] = 0;
			for(int bpos = 0; bpos < bytespercolumn; bpos++) {
				colbits[col] = colbits[col]<<8 | bitmap[fontdatapos + 1 + bytespercolumn*col + bpos];
			}
//			cout << "0x" << setw(8) << setfill('0')
//					 << hex << colbits[col];
			if ( col & 1 == 0 ) {
				uint32_t shrinked = 0;
				for(int bit = 0; bit < maxheight/2; bit++) {
					shrinked |= (colbits[col]>>(bit*2) & 1 ? 1<<bit : 0 );
				}
//				cout << " -> ";
				for(int pos = 0; pos < bytespercolumn>>1; pos++) {
					cout << "0x" << setw(2) << setfill('0')
								 << hex << (shrinked>>(pos*8)&0xff)<< ", ";
				}
			} else {
//				cout << endl;
			}
		}
		cout << endl;
	}
	cout << "// sizes " << endl;
	for(int i = 0; i < numchar; i++) {
		cout << "0x" << setw(2) << setfill('0') << hex << (uint16_t)sizes[i] << ", ";
	}
	cout << endl;

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}

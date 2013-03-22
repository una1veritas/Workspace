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
	uint32_t fdata[numchar+1][maxwidth];
	uint8_t fwidth[numchar+1];

	for(int i = 0; i < numchar; i++) {
//		cout << "// "<< hex << (basechar+i);
		fontdatapos = i * (bytesperchar + 1);
		int fontwidth = bitmap[fontdatapos];
//		cout << " (width " << fontwidth << ")" << endl;
//		cout << "0x" << setw(2) << setfill('0')
//							 << hex << (fontwidth>>1) << ", " << endl;
		fwidth[i] = fontwidth>>1;
		for(int col = 0 ; col < fontwidth; col++) {
			uint32_t colbits = 0;
			for(int bpos = 0; bpos < bytespercolumn; bpos++) {
				colbits = colbits<<8 | bitmap[fontdatapos + 1 + bytespercolumn*col + bpos];
			}
//			cout << "0x" << setw(8) << setfill('0')
//					 << hex << colbits[col];
			if ( (col & 1) == 0 ) {
				uint32_t shrinked = 0;
				for(int bit = 0; bit < maxheight/2; bit++) {
					shrinked |= (colbits>>(bit*2) & 1 ? 1<<bit : 0 );
				}
				fdata[i][col>>1] = shrinked;
			}
//			cout << endl;
		}
		// cout << endl;
	}
	cout << "// sizes " << endl;
	for(int i = 0; i < numchar; i++) {
		if ( i%16 == 0 )
			cout << endl;
		cout << "0x" << setw(2) << setfill('0') << hex << (uint16_t)fwidth[i] << ", ";
	}
	cout << endl;
	cout << "	// font data" << endl;
	for(int i = 0; i < numchar; i++) {
		for(int bpos = 0; bpos < ceil((float)maxheight/16); bpos++) {
			for(int col = 0 ; col < fwidth[i]; col++) {
				cout << "0x" << setw(2) << setfill('0') << hex
						<< (uint16_t)(fdata[i][col]>>((2-bpos-1)*8)&0xff) << ", ";
			}
			cout << endl;
		}
		cout << "// "<< i+basechar << endl;
	}
	cout << endl;

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}

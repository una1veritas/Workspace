/*
 * main.cpp
 *
 *  Created on: 2017/05/22
 *      Author: sin
 */

#include <iostream>
#include <string>

int main(const int argc, const char * argv[]) {
	const unsigned int DATA_MAXSIZE = 1024*8;
	int * data;
	unsigned int count;
	std::string buf;
	char pulse;

	if ( !(argc > 1) ) {
		data = new int[DATA_MAXSIZE];
		count = 0;
		while ( ! std::cin.eof() ) {
			std::cin >> buf;
			data[count++] = std::strtol(buf.c_str(), NULL, 16);
		}
	} else if ( std::string(argv[1]) == "-pc1500") {
		std::cout << "PC-1500" << std::endl;
		data = new int[DATA_MAXSIZE];
		count = 0;
		unsigned int qbits = 0;
		unsigned int bitCount = 0;
		unsigned int pulseCount = 0;
		while ( !std::cin.eof() ) {
			std::cin >> pulse;

			if ( pulseCount == 0 ) {
				if ( pulse == '0' )
					pulseCount = 1;
				else
					pulseCount = 0x11;
			} else {
				if ( pulse == '0' && (pulseCount & 0x10) == 0 ) {
					pulseCount++;
				} else if ( pulse == '1' && (pulseCount & 0x10) != 0 ) {
					pulseCount++;
				} else {
					std::cout << "error " << std::hex << pulseCount << std::endl;
					if ( pulse == '0' )
						pulseCount = 1;
					else if ( pulse == '1' )
						pulseCount = 0x11;
				}
			}

			if ( (pulseCount == 0x04) || (pulseCount == 0x18) ) {
				if ( pulseCount == 0x04 ) {
					// bit 0 out
					qbits = (qbits>>1);
					std::cout << "0 ";
				} else if ( pulseCount == 0x18 ) {
					// bit 1 out
					qbits = (qbits>>1) | 0x40;
					std::cout << "1 ";
				}
				/*
				bitCount++;
				if ( bitCount >= 7 ) {
					if ((qbits & 0x61) == 0x60) {
						std::cout << std::hex << ((qbits>>1) & 0x0f) << " ";
					} else {
						std::cout << std::hex << qbits << " ";
					}
					bitCount = 0;
					qbits = 0;
				}
				*/
				pulseCount = 0;
			}
		}
		std::cout << std::endl;
	} else {
		return 0;
	}

	std::cout << "Data " << count << " bytes." << std::endl;
	for(int i = 0; i < count; i++) {
		if ( (i & 0x0f) == 0 )
			std::cout << std::endl;
		else
			std::cout << " ";
		std::cout << "0x" << std::hex << (0x0f & (data[i]>>4))
						<< std::hex << (0x0f & data[i]);
	}
	std::cout << std::endl;
	delete [] data;

	return 0;
}



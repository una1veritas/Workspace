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

	data = new int[DATA_MAXSIZE];
	count = 0;
	char pulse;
	int bits = 0;
	unsigned int val;
	unsigned int hilo = 0;
	while ( ! std::cin.eof() ) {
		std::cin >> pulse;
		if ( pulse == '0' || pulse == '1' ) {
			if ( bits == 0 && pulse == '0') { // start bit
				val = 0;
				bits = 1;
			} else if ( bits == 0 && pulse == '1' ) {
				// start bit is continuing...
				continue;
			} else if ( bits >= 1 && bits <= 4 ) {
				val >>= 1;
				if ( pulse == '1') {
					val |= 0x08;
				}
				bits++;
			} else if ( bits >= 5 && pulse == '1' ) { // the first stop bit
				std::cout << std::hex << val;
				bits = 0;
				std::cout << " ";
			} else {
				std::cout << "error ";
			}
		}
	}

	std::cout << std::endl;
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



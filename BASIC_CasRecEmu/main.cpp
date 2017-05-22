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
	unsigned int count, i;
	std::string buf;
	char tmpchar;

	if ( !(argc > 1) ) {
		data = new int[DATA_MAXSIZE];
		count = 0;
		while ( ! std::cin.eof() ) {
			std::cin >> buf;
			data[count++] = std::strtol(buf.c_str(), NULL, 16);
		}
	} else if ( std::string(argv[1]) == "-pc1500") {
		std::cout << "1500!!" << std::endl;
		data = new int[DATA_MAXSIZE];
		count = 0;
		unsigned char lastchar;
		unsigned int pulses = 0;
		while ( !std::cin.eof() ) {
			std::cin >> tmpchar;
			++pulses;
			if ( pulses == 1 ) {
				lastchar = tmpchar;
			} else if ( pulses == 4 && lastchar == '0' && lastchar == tmpchar ) {
				std::cout << lastchar << " ";
				pulses = 0;
			} else if ( pulses == 8 && lastchar == '1' && lastchar == tmpchar ) {
				std::cout << lastchar << " ";
				pulses = 0;
			} else if ( pulses > 1) {
				if ( lastchar != tmpchar ) {
					std::cout << std::endl << "error ";
					std::cout << lastchar << "(" << pulses - 1 << "), " << std::endl;
					lastchar = tmpchar;
					pulses = 1;
				}
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



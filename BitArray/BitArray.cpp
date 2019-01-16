//============================================================================
// Name        : BitArray.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
//============================================================================

#include <iostream>
#include <vector>

int main(void) {

	std::cout << "Hello World!!!" << std::endl;

	std::vector<bool> barray(1024*1024);
	unsigned long val = 4465;
	unsigned char * p = (unsigned char *) &val;
	for (int i = 0; i < 8; ++i)
		std::cout << std::hex << static_cast<unsigned int>(*p++) << ' ';
	std::cout << std::endl;

	std::cout << "size of barray = " << sizeof(barray) << std::endl;
	unsigned long bit;
	unsigned int pos;
	for (bit = 1, pos = 0; pos < 31; bit <<= 1, ++pos) {
		barray[pos] = (bit & val) ? 1 : 0 ;
		std::cout << (int)((bit & val) ? 1 : 0) << " ";
	}
	std::cout << std::endl;
	char t[16];
	std::cin.getline(t,15);
	return EXIT_SUCCESS;
}


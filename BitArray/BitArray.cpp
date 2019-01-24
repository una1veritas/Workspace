//============================================================================
// Name        : BitArray.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
//============================================================================

#include <iostream>
#include <vector>

#include <cinttypes>

typedef std::vector<bool> BitArray;
typedef uint32_t uint32;


uint32 nlz32(uint32 x)
{
	// Hacker's Delight 2nd by H. S. Warren Jr., 5.3, p. 104 --
	double d = (double) x + 0.5;
	uint32 *p = ((uint32*) &d) + 1;
	return 0x41e - (*p>>20);  // 31 - ((*(p+1)>>20) - 0x3FF)
}
/*
uint32 nlz32(uint32 x)
{
    int ret;
    __asm__ volatile ("lzcnt %1, %0" : "=r" (ret) : "r" (x) );
    return ret;
}
*/

bool encode(BitArray & barray, uint32 & intval) {
	unsigned int digits;
	unsigned long bitmask;
	switch ( intval ) {
	case 0:
		barray.push_back(0);
		barray.push_back(0);
		break;
	case 1:
		barray.push_back(0);
		barray.push_back(1);
		break;
	default:
		digits = 31 - nlz32(intval);
		std::cout << digits << std::endl;
		for(unsigned int i = 0; i< digits; ++i)
			barray.push_back(1);
		barray.push_back(0);
		bitmask = 1<<(digits-1);
		for(unsigned int i = 0; i< digits; ++i) {
			std::cout << std::hex << bitmask << std::endl;
			barray.push_back( (bitmask & intval ? 1 : 0) );
			bitmask >>= 1;
		}
	}
	return true;
}

int main(void) {

	std::cout << "Hello World!!!" << std::endl;

	BitArray barray;
	const char str[] = "Go to the hell 3092!";
	std::cout << "size of barray = " << sizeof(barray) << std::endl;

	for(const char * p = str; *p; ++p) {
		uint32 c = *p;
		std::cout << std::hex << c << std::endl;
		encode(barray, c);
	}
	std::cout << std::endl;
	//char t[16];
	//std::cin.getline(t,15);
	for (auto i = barray.begin(); i != barray.end() ; ++i) {
		std::cout << *i << " ";
	}
	std::cout << std::endl;
	return EXIT_SUCCESS;
}


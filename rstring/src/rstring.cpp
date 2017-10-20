//============================================================================
// Name        : rstring.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <random>

void get_alphabet(unsigned char alphabet[], const char * ptr);

int main(const int argc, char * const argv[]) {
	unsigned char alphabet[256];
	get_alphabet(alphabet, argv[1]);
	unsigned int alphabet_size = strlen((const char*)alphabet);
	std::random_device rdev;
	std::mt19937 mt(rdev()), mt_len(rdev());
	std::uniform_int_distribution<unsigned char> ascode(0,alphabet_size);
	std::uniform_int_distribution<double> lenrand(0,1.0f);

	for(int i = 0; i < 100; i++) {
		for(int l = 0; l < 32; l++) {
			std::cout << alphabet[ascode(mt)];
			if ( lenrand(mt_len) < 0.1 )
				break;
		}
		std::cout << std::endl;
	}
	return 0;
}

void get_alphabet(unsigned char alph[], const char *argv) {
	unsigned char * ptr = alph;
	if ( argv == NULL || *argv == '\0' ) {
		for (unsigned int i = 0x20; i < 0x7f; i++) {
			*ptr++ = (unsigned char) i;
		}
		*ptr = '\0';
	}
	return;
}

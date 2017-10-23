//============================================================================
// Name        : rstring.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <random>

#include <cstdlib>
#include <cctype>

#include <math.h>

unsigned int get_alphabet(unsigned char alphabet[], const int argc, char * ptr[]);

int main(const int argc, char * argv[]) {
	unsigned char alphabet[256+1];
	unsigned int alph_size;
	unsigned int max_length;
	alph_size = get_alphabet(alphabet, argc, argv);
	std::random_device rdev;
	std::mt19937 mt(rdev()), mt_len(rdev()), mt_rand(rdev());
	std::uniform_int_distribution<unsigned char> ralph(0,alph_size);
	std::uniform_int_distribution<unsigned int> rlength(1,32);
	std::uniform_real_distribution<double> random(0,1.0);

	for(int i = 0; i < 1000; ) {
		int len = rlength(mt_len);
		if ( random(mt_rand) > pow((double)len/32,3)/(exp( len/(double)32)-1) )
			continue;
		for(int l = 0; l < len; l++) {
			std::cout << alphabet[ralph(mt)];
		}
		++i;
		std::cout << std::endl;
	}
	return 0;
}

unsigned int get_alphabet(unsigned char alph[], const int argc, char *argv[]) {
	unsigned char * ptr = alph;
	char * argptr = NULL;
	unsigned int argpos;
	for(argpos = 1; argpos < argc; argpos++) {
		if ( strncmp(argv[argpos], "-a", 2) )
			continue;
		if ( strlen(argv[argpos]+2) == 0 ) {
			if ( (argpos+1 < argc) && (argv[argpos+1] != NULL) )
				argptr = argv[++argpos];
		} else {
			std::cout << "here2" << std::endl;
			argptr = argv[argpos]+2;
		}
		break;
	}
	if ( argptr == NULL ) {
		for (unsigned int i = 0x30; i < 0x7f; i++) {
			if ( isalnum((char)i) )
				*ptr++ = (unsigned char) i;
		}
		*ptr = '\0';
	} else {
		strncpy((char *)alph, argptr, 256);
		alph[256] = '\0';
	}
	return strlen((char*)alph);
}

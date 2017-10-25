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
int get_int(const int argc, char *argv[], const char * opt, const int defval) ;

int main(const int argc, char * argv[]) {
	unsigned char alphabet[256+1];
	unsigned int alph_size;
	unsigned int max_length, mode_length, outnum;
	alph_size = get_alphabet(alphabet, argc, argv);
	max_length = get_int(argc, argv, "-l", 32);
	mode_length = get_int(argc, argv, "-l", max_length*3/5);
	outnum = get_int(argc, argv, "-n", 1000);

	std::random_device rdev;
	std::mt19937 mt(rdev()), mt_len(rdev()), mt_rand(rdev());
	std::uniform_int_distribution<unsigned char> ralph(0,alph_size);
	std::uniform_int_distribution<unsigned int> rlength(1,32);
	std::uniform_real_distribution<double> random(0,1.0);

	for(int i = 0; i < outnum; ) {
		int rlen = rlength(mt_len);
		double sg = 8.0;
		double pr = 1/(2.5f*sg)*exp(-pow((double)rlen - (double)mode_length,2)/pow(2*sg,2));
		if ( random(mt_rand) > pr )
			continue;
		for(int l = 0; l < rlen; l++) {
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

int get_int(const int argc, char *argv[], const char * opt, const int defval) {
	int length = defval;
	char * argptr = NULL;
	unsigned int argpos;
	for(argpos = 1; argpos < argc; argpos++) {
		if ( strncmp(argv[argpos], opt, 2) )
			continue;
		if ( strlen(argv[argpos]+2) == 0 ) {
			if ( (argpos+1 < argc) && (argv[argpos+1] != NULL) )
				argptr = argv[++argpos];
		} else {
			argptr = argv[argpos]+2;
		}
		break;
	}
	if ( argptr != NULL )
		length = atoi(argptr);
	return length;
}

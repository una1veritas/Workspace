//============================================================================
// Name        : permutator.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <set>

#include <cctype>

#include "Permutation.h"

int main(const int argc, const char * argv[]) {

	if ( !(argc > 1) )
		return EXIT_FAILURE;

	std::string str(argv[1]);
	std::cout << "input: " << str << std::endl;

	Permutation<char> perm(str);

	do {
		std::string t = str;
		std::cout << perm << ": " << perm.map(t) << std::endl;
	} while (perm.next());


	std::cout << std::endl << std::endl;
	return EXIT_SUCCESS;
}

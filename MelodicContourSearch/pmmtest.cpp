/*
 * pmmtest.cpp
 *
 *  Created on: 2020/02/18
 *      Author: sin
 */

#include <cstdlib>

#include <iostream>
#include "stringmatching.h"
#include "manamatching.h"

int main(const int argc, const char * argv[]) {
	if ( !(argc > 2) )
		return EXIT_FAILURE;
	const char * p = argv[1], * t = argv[2];
	std::cout << "pattern = '" << argv[1] << "', " << std::endl
			<< "text = '" << argv[2] << "'." << std::endl;

	kmp kmp(p);
	manakmp mkmp(p);
	int pos;
	std::cout << "kmp: " << kmp << std::endl;
	pos = kmp.find(t);
	std::cout << pos << "/" << strlen(t) << std::endl;

	std::cout << "manakmp: " << mkmp << std::endl;
	pos = mkmp.search(t);
	std::cout << pos << "/" << strlen(t) << std::endl;
	return EXIT_SUCCESS;
}

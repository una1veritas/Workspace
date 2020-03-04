/*
 * dirlister.cpp
 *
 *  Created on: 2020/03/01
 *      Author: Sin Shimozono
 */

#include <iostream>
#include <cstdlib>
#include <regex>

#include "dirlister.h"

int main( void ) {
	std::regex regpat(".*\\.(mid|MID)");
	dirlister dl(".");
	unsigned int counter = 0;
	while ( dl.get_next_entry(regpat) ) {
		//std::cout << dl.entry_name() << std::endl;
		std::cout << dl.entry_fullpath() << std::endl;
		counter += 1;
	}
	std::cout << "found " << counter << " files." << std::endl << "finished." << std::endl;
	return EXIT_SUCCESS;
}

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
	std::regex regpat(".*\\.h"); //(mid|MID)");
	dirlister dl(".");
	while ( dl.get_next_entry(regpat) ) {
		//std::cout << dl.entry_name() << std::endl;
		std::cout << dl.entry_fullpath() << std::endl;
	}
	std::cout << "finished." << std::endl;
	return EXIT_SUCCESS;
}

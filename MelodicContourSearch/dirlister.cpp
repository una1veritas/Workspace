/*
 * dirlister.cpp
 *
 *  Created on: 2020/03/01
 *      Author: Sin Shimozono
 */

#include <iostream>
#include <cstdlib>
#include <io.h>
#include <regex>

#include "dirlister.h"

int main( void ) {
	std::regex regpat(".*\\.h");
	dirlister dl(".", regpat);
	while ( dl.get_next_entry() ) {
		std::cout << dl.entry_name() << std::endl;
	}
	std::cout << "finished." << std::endl;
	return EXIT_SUCCESS;
}

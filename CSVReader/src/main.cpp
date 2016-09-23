/*
 * main.cpp
 *
 *  Created on: 2016/09/23
 *      Author: Sin Shimozono
 */

#include <cstdlib>

#include <iostream>
#include <fstream>
//#include <string>

#include "main.h"


int main(const int argc, const char * argv[]) {

	std::string filename = "./test.csv";

	std::fstream csvfile;
	csvfile.open("/Users/sin/Documents/Workspace/permutator/words.txt", std::fstream::in);
	if ( csvfile.is_open() ) {
		std::cout << "reading data..." << std::endl;
		load_dict(table, csvfile);
		csvfile.close();
	} else {
		std::cout << "Failed opening the data file " << filename << std::endl << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void load_dict(std::set<std::string> & list, std::istream & indata) {
	list.clear();
	while ( !indata.eof() ) {
		std::string word;
		indata >> word;
		if ( word.length() > 0 ) {
			list.insert(std::string(word));
		}
	}

	return;
}

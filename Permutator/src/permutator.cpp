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

void load_dict(std::set<std::string> &, std::istream &);

int main(const int argc, const char * argv[]) {
	if ( argc == 0 )
		return EXIT_FAILURE;
	std::string str(argv[1]);
	for(int i = 0; i < str.length(); ++i)
		str[i] = tolower(str[i]);
	Permutation perm(str.length());

	//std::cout << "!!!Hello World!!! " << std::endl; // prints !!!Hello World!!!

	std::fstream dictfile;
	std::set<std::string> worddict;
	dictfile.open("/Users/sin/Documents/Workspace/permutator/words.txt", std::fstream::in);
	if ( dictfile.is_open() ) {
		std::cout << "reading words..." << std::endl;
		load_dict(worddict, dictfile);
		dictfile.close();
	} else {
		std::cout << "Failed opening the words file." << std::endl << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << worddict.size() << " words: " << std::endl;
	std::set<std::string>::iterator it;
	for(it = worddict.begin(); it != worddict.end(); it++) {
		std::string word = *it;
		std::cout << word << std::endl;
	}
	std::cout << std::endl;

	std::string matched = "";
	std::cout << perm << std::endl << std::endl;
	do {
		std::string t = str;
		perm.map(t);
		std::set<std::string>::iterator pos = worddict.find(t);
		if ( pos != worddict.end() ) {
			matched = t;
		}
		std::cout << t << std::endl;
	} while (perm.next());

	if ( matched == "" ) {
		std::cout << std::endl << "No suggestion." << std::endl << std::endl;
	} else {
		std::cout << std::endl  << "Suggestion: " << matched << std::endl << std::endl;
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
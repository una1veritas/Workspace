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
unsigned int prefixmatch(const std::string & t, const std::string & p);

int main(const int argc, const char * argv[]) {
	struct {
		bool all = false;
	} options;

	if ( argc == 0 )
		return EXIT_FAILURE;
	std::string str(argv[1]);
	if ( str == "-a" ) {
		options.all = true;
		str = argv[2];
	}
	for(int i = 0; (unsigned int)i < str.length(); ++i)
		str[i] = tolower(str[i]);
	std::sort(&str[0], &str[str.length()], [](char a, char b) { return a < b; } );

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
//	std::set<std::string>::iterator it;
//	for(it = worddict.begin(); it != worddict.end(); it++) {
//		std::string word = *it;
//		std::cout << word << std::endl;
//	}
//	std::cout << std::endl;

	std::vector<std::string> matched;
	double bestpoint = 0;
	std::cout << perm << std::endl << std::endl;
	do {
		std::string t = str;
		perm.map(t);
		if ( options.all )
			std::cout << t << std::endl;
		for(std::set<std::string>::iterator elem = worddict.begin(); elem != worddict.end(); elem++) {
			double point = prefixmatch(t, *elem)/(double)(t.length() > elem->length() ? t.length() : elem->length() );
			if ( point == 0 )
				continue;
			if ( point > bestpoint ) {
				if ( bestpoint < 1.0 ) {
					if ( matched.size() > 0 ) // must be equal to 1
						matched.pop_back();
				}
				std::cout << *elem << " <-> " << t;
				std::cout << " with " << (int)(point*100) << "%" << std::endl;;
				if ( t.length() < elem->length() )
					matched.push_back(*elem);
				else
					matched.push_back(t);
				bestpoint = point;
			}
		}
	} while (perm.next());

	std::cout << std::endl  << "Suggestion: " << std::endl;
	for(std::vector<std::string>::iterator it = matched.begin();
			it != matched.end(); ++it ) {
		std::cout << *it << ", ";
	}
	std::cout << std::endl << std::endl;
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

unsigned int prefixmatch(const std::string & str1, const std::string & str2) {
	unsigned int matchlen;
	unsigned int pattlen = ((str1.length() < str2.length()) ? str1.length() : str2.length() );
	for ( matchlen = 0; matchlen < pattlen; matchlen++ ) {
		if ( str1[matchlen] != str2[matchlen] )
			break;
	}

	return matchlen;
}

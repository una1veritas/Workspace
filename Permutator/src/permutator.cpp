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
#include <unordered_map>
#include <cctype>

#include "Permutation.h"

#include <ncurses.h>

typedef uint32_t uint32;
typedef std::set<std::string> wordset;
typedef std::unordered_map<uint32,std::vector<std::string> > dict;

void load_dict(dict &, std::istream &);
unsigned int prefixmatch(const std::string & t, const std::string & p);
uint32 hashCode(const std::string & );

int main(const int argc, const char * argv[]) {
//	WINDOW * mywin;
	struct {
		bool all;
	} options = { false };

	if ( argc == 0 )
		return EXIT_FAILURE;
	std::string str(argv[1]);
	if ( str == "-all" ) {
		options.all = true;
		str = argv[2];
	}
	for(int i = 0; (unsigned int)i < str.length(); ++i)
		str[i] = tolower(str[i]);

	Permutation<char> perm(str);

	//std::cout << "!!!Hello World!!! " << std::endl; // prints !!!Hello World!!!

	std::fstream dictfile;
	dict worddict;
	dictfile.open("/Users/sin/Documents/Workspace/permutator/words.txt", std::fstream::in);
	if ( dictfile.is_open() ) {
		std::cout << "reading words..." << std::endl;
		load_dict(worddict, dictfile);
		dictfile.close();
	} else {
		std::cout << "Failed opening the words file." << std::endl << std::endl;
		return EXIT_FAILURE;
	}

	/*
	if ( (mywin = initscr()) == NULL ) {
		std::cerr << "ncurses start failed." << std::endl;
		return EXIT_FAILURE;
	}
	move(0, 0);
	printw("%6d words found in the dictionary", worddict.size());
	refresh();
	*/
	std::cout << worddict.size() << " words found in the dictionary." << std::endl;
//	std::set<std::string>::iterator it;
//	for(it = worddict.begin(); it != worddict.end(); it++) {
//		std::string word = *it;
//		std::cout << word << std::endl;
//	}
//	std::cout << std::endl;

	std::vector<std::string> matched;
	long update = 0;
	double bestpoint = 0;
	std::cout << perm << std::endl << std::endl;
	uint32 keycode = hashCode(str.c_str());
	std::cout << "Will compare with " << worddict[keycode].size() << " words." << std::endl;
	do {
		std::string t = str;
		perm.map(t);
		if ( options.all ) {
			//move(1,0);
			//printw("%s hash code = %08x", t.c_str(), hashCode(t));
			//move(2,0);
			//printw("%d words", worddict[keycode].size());
			//refresh();
			std::cout << t << std::endl;
		}
		for(std::vector<std::string>::iterator elem = worddict[keycode].begin();
				elem != worddict[keycode].end(); elem++) {
			double point = prefixmatch(t, *elem)/(double)(t.length() > elem->length() ? t.length() : elem->length() );
			if ( point == 0 )
				continue;
			if ( (bestpoint < 1.0 && point > bestpoint) || point == 1.0 ) {
				if ( point > bestpoint ) {
					if ( matched.size() > 0 ) // must be equal to 1
						matched.pop_back();
				}
				//move(3+update, 0);
				std::cout << *elem << " <-> " << t;
				std::cout << " with " << (int)(point*100) << "%" << std::endl;;
				//printw("%s <-> %s (%3.1f%%)", elem->c_str(), t.c_str(), point*100);
				//refresh();
				if ( t.length() < elem->length() )
					matched.push_back(*elem);
				else
					matched.push_back(t);
				bestpoint = point;
				update++;
			}
		}
	} while (perm.next());

	/*
	delwin(mywin);
	endwin();
	refresh();
	 */
	std::cout << std::endl  << "Suggestion: " << std::endl;
	for(std::vector<std::string>::iterator it = matched.begin();
			it != matched.end(); ++it ) {
		std::cout << *it << ", ";
	}
	std::cout << std::endl << std::endl;


	return EXIT_SUCCESS;
}

void load_dict(dict & wdic, std::istream & indata) {
	wordset words;
	while ( !indata.eof() ) {
		std::string word;
		indata >> word;
		if ( word.length() > 0 ) {
			words.insert(word);
		}
	}

	wdic.clear();
	for(wordset::iterator it = words.begin();
			it != words.end(); it++) {

		wdic[hashCode(it->c_str())].push_back(*it);
		//std::cout << *it << ", 0x" << std::hex << hashCode(it->c_str()) << std::dec << std::endl;

		/*
		for(std::vector<std::string>::iterator wit = wdic[hashCode(it->c_str())].begin();
				wit != wdic[hashCode(it->c_str())].end(); wit++) {
			std::cout << *wit << ", ";
		}
		std::cout << std::endl;
		*/
/*
	dict::iterator index = list.find(hashCode(word.c_str()));
	if ( index == list.end() ) {
		index->second.push_back(word);
	} else {
		std::vector<std::string> newlist;
		newlist.push_back(word);
		list.insert(std::pair<uint32,std::vector<std::string>>(hashCode(word),newlist));
	}
*/
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

uint32 hashCode(const std::string & str) {
	uint32 tmp = 0;
	char c;
	for(int pos = 0; pos < str.length(); pos++) {
		if ( isalnum(str[pos]) ) {
			c = tolower(str[pos]) - 'a';
			tmp |= 1<<c;
		}
	}
	return tmp;
}

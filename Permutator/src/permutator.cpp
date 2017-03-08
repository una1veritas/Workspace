//============================================================================
// Name        : permutator.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <cctype>

#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_map>


#ifndef MAX
#define MAX(x, y)   ((x) < (y)? (y) : (x))
#endif
#ifndef MIN
#define MIN(x, y)   ((x) > (y)? (y) : (x))
#endif
#ifndef ABS
#define ABS(x)  ((x) < 0 ? -(x) : (x))
#endif //ABS

typedef unsigned int uint;
typedef uint32_t uint32;
typedef std::set<std::string> wordset;
typedef std::unordered_map<uint32,std::vector<std::string> > dict;

void setframe(uint * frame, const uint n, const uint m);
uint dp_edist(uint * frame, uint * table, const std::string & t, const std::string & p);

void load_dict(dict &, std::istream &);
uint prefixmatch(const std::string & t, const std::string & p);
uint32 hashCode(const std::string & );

uint32 pop32_SSE42(uint32 u32)
{
    int ret;
    __asm__ volatile ("popcntl %1, %0" : "=r"(ret) : "r"(u32) );
    return ret;
}

int main(const int argc, const char * argv[]) {
	std::string perm;

	struct {
		bool all;
	} options = { false };

	if ( argc == 1 )
		return EXIT_FAILURE;
	if ( argc == 2 ) {
		perm = argv[1];
	} else
	if ( std::string(argv[1]) == "-all" ) {
		options.all = true;
		perm = argv[2];
	}

	for(uint32 i = 0; i < perm.length(); ++i)
		perm[i] = tolower(perm[i]);
	std::sort(perm.begin(), perm.end());
	for(uint32 i = 0; i < perm.length(); i++) {
		std::cout << perm[i];
		if ( i + 1 < perm.length() )
			std::cout << ", ";
	}
	std::cout << std::endl;

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
	uint32 keycode = hashCode(perm);
	std::cout << "Will compare with " << worddict[keycode].size() << " words." << std::endl;
	const uint32 distlim = 2;

	while ( std::next_permutation(perm.begin(), perm.end()) ) {
		if ( options.all ) {
			std::cout << perm << std::endl;
		}
		for(uint32 bpos = 0; bpos < 27; bpos++) {
			uint32 chkcode;
			if ( bpos == 0 )
				chkcode = keycode;
			else
				chkcode = keycode ^ (uint32(1)<<(bpos-1));
			const dict::iterator itr = worddict.find(chkcode);
			if ( itr == worddict.end() )
				continue;
			for(std::vector<std::string>::iterator elem = itr->second.begin();
					elem != itr->second.end(); elem++) {
				if ( ABS(perm.length() - elem->length()) > distlim )
					continue;
				double point = prefixmatch(perm, *elem)/(double)MAX(perm.length(), elem->length() );
				if ( point == 0 )
					continue;
				if ( (bestpoint < 1.0 && point > bestpoint) || point == 1.0 ) {
					if ( point > bestpoint ) {
						if ( matched.size() > 0 ) // must be equal to 1
							matched.pop_back();
					}
					//move(3+update, 0);
					std::cout << *elem << " <-> " << perm;
					std::cout << " with " << (int)(point*100) << "%" << std::endl;;
					matched.push_back(*elem);
					bestpoint = point;
					update++;
				}
			}
		}
	};

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

uint prefixmatch(const std::string & str1, const std::string & str2) {
	static uint table[16*16];
	static uint frame[16+16+1];
	uint matchlen;
	const std::string * text;
	const std::string * patt;

	if ( str1.length() >= str2.length() ) {
		text = &str1;
		patt = &str2;
	} else {
		text = &str2;
		patt = &str1;
	}

	if ( text->length() < 16 ) {
		// do approximate match
		setframe(frame, text->length(), patt->length());
		matchlen = dp_edist(frame, table, *text, *patt);
		matchlen = text->length() - matchlen;
	} else {
		std::cout << "!";
		for ( matchlen = 0; matchlen < patt->length(); matchlen++ ) {
			if ( text->at(matchlen) != patt->at(matchlen) )
				break;
		}
	}
	return matchlen;
}

uint32 hashCode(const std::string & str) {
	uint32 tmp = 0;
	uint32 c;
	for(unsigned int pos = 0; pos < str.length(); pos++) {
		c = str[pos];
		if ( isalpha(c) ) {
			tmp |= (uint32(1) << (c - 'A'));
		} else {
			tmp |= (c & 0x1f);
		}
	}
	return tmp;
}

void setframe(uint * frame, const uint n, const uint m) {
	for (long i = 0; i < n + m + 1; i++) {
		frame[i] = ABS(m - i);  // m (pattern, left) frame
	}
}

uint dp_edist(uint * frame, uint * dist, const std::string & t, const std::string & p) {
	long col, row;
	long ins, del, repl, result;
	const int lcs_switch = 0;
	const uint n = t.length();
	const uint m = p.length();

	//printf("text %s (%d), pattern %s (%d)\n", t, n, p, m);

	if ( dist == NULL )
		return n+m+1; // error

	// n -- the number of columns (the text length), m -- the number of rows (the pattern length)
	//table calcuration
	// let and top frames of the table corresponds to col = -1, row = -1
	for(col = 0; col < n; col++) { // column, text axis
		for (row = 0; row < m; row++) {  // row, pattern axis
			ins = ( col == 0 ? frame[(m-1)-row] : dist[row + m*(col-1)] ) + 1;
			del = ( row == 0 ? frame[m+1+col] : dist[(row-1) + m*col] ) + 1;
			repl = ( col == 0 || row == 0 ?
					( col == 0 ? frame[m-row] : frame[m+col] ) :
					dist[(row-1) + m*(col-1)] );
			//
			if ( t[col] == p[row] ) {
				if ( ins < del && ins < repl ) {
					repl = ins;
				} else if (del < ins && del < repl ) {
					repl = del;
				}
			} else {
				repl += 1;
				if ( ins <= del && (lcs_switch || ins < repl) ) {
					repl = ins;
				} else if (del < ins && (lcs_switch || del < repl) ) {
					repl = del;
				}
			}
			dist[row + m*col] = repl;
			//printf("[%ld,%ld] %c|%c : %ld/%ld/%ld+%d >%ld,\n", col, row, t[col], p[row], del,ins, repl, (t[col] != p[row]), dist[col*m+row]);
		}
	}

	result = dist[n*m-1];

	return result;
}

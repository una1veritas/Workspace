/*
 * nexttoken.cpp
 *
 *  Created on: 2014/08/30
 *      Author: sin
 */

#include <iostream>
#include <string>

namespace {
using namespace std;

void nextToken(istream& is, string& result, const string& word) {
	char c;
	int i;
	string buf("");
	result.clear();
	for( i = 0; !is.eof() ; ) {
		is >> noskipws >> c;
		if ( word[i] == c ) {
			i++;
			buf.append(1,c);
			if ( i == word.length() ) {
				break;
			}
		} else {
			if ( i > 0 ) {
				result.append(buf);
				buf.clear();
			}
			i = 0;
			result.append(1,c);
		}
	}
	return;
}
}

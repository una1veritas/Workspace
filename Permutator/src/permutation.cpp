/*
 * Permutation.cpp
 *
 *  Created on: 2016/07/24
 *      Author: sin
 */

#include "Permutation.h"

std::string & Permutation::map(std::string & str) const {
	std::string tmp(str);
	str.clear();
	for(int unsigned i = 0; i < size; i++) {
		if ( perm[i] < tmp.length() ) {
			str.push_back(tmp[perm[i]]);
		}
	}
	return str;
}

const bool Permutation::next(void) {
	unsigned int boundary = size - 1;
	for ( ; boundary > 0 && perm[boundary-1] > perm[boundary]; --boundary) {}
	//std::cout << "boundary = " << boundary << ", " << *this << std::endl;
	if ( boundary == 0 )
		return false;
	//sort
	std::sort(perm.begin()+boundary, perm.end(), [](int a, int b) { return a < b; } );
	//std::cout << "sorted: " << *this << std::endl;
	// find the next for perm[top-1]
	unsigned int larger;
	for(larger = boundary; larger < size; ++larger) {
		if ( perm[boundary-1] < perm[larger] )
			break;
	}
	if ( !(larger < size) ) {
		std::cerr << "next error!" << std::endl;
		return false;
	}
	unsigned int t = perm[boundary-1];
	perm[boundary-1] = perm[larger];
	perm[larger] = t;
	//std::cout << *this << std::endl;
	return true;
}

std::ostream & Permutation::printOn(std::ostream & out) const {
	out << "(";
	for(unsigned int i = 0; i < size; i++) {
		out << perm[i];
		if ( i+1 < size )
			out << ", ";
	}
	out << ") ";
	return out;
}
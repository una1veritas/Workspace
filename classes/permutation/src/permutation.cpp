/*
 * Permutation.cpp
 *
 *  Created on: 2016/07/24
 *      Author: sin
 */

#include "Permutation.h"

template <typename T>
std::string & Permutation<T>::map(std::string & str) const {
	const size_t orglen = str.length();
	for(int unsigned i = 0; i < size; i++) {
		str.push_back(str[perm[i]]);
	}
	return str.erase(0,orglen);
}

template <>
std::string & Permutation<char>::map(std::string & str) const {
	for(int unsigned i = 0; i < size; i++) {
		str[i] = perm[i];
	}
	return str;
}


template <>
const bool Permutation<unsigned int>::next(void) {
	unsigned int boundary = size - 1;
	for ( ; boundary > 0 && perm[boundary-1] >= perm[boundary]; --boundary) {}
	//std::cout << "boundary = " << boundary << ", " << *this << std::endl;
	if ( boundary == 0 )
		return false;
	//sort
	std::sort(perm.begin()+boundary, perm.end(), [](unsigned int a, unsigned int b) { return a < b; } );
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

template <>
const bool Permutation<char>::next(void) {
	unsigned int boundary = size - 1;
	for ( ; boundary > 0 && perm[boundary-1] >= perm[boundary]; --boundary) {}
	// std::cout << "boundary = " << boundary << ", " << *this << std::endl;
	if ( boundary == 0 )
		return false;
	//sort
	std::sort(perm.begin()+boundary, perm.end(), [](char a, char b) { return a < b; } );
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


template <typename T>
std::ostream & Permutation<T>::printOn(std::ostream & out) const {
	out << "(";
	for(unsigned int i = 0; i < size; i++) {
		out << perm[i];
		if ( i+1 < size )
			out << ", ";
	}
	out << ") ";
	return out;
}

template class Permutation<unsigned int>;
template class Permutation<char>;

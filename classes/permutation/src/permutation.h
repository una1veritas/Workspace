/*
 * Permutation.h
 *
 *  Created on: 2016/07/24
 *      Author: sin
 */

#ifndef PERMUTATION_H_
#define PERMUTATION_H_

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

template <typename T>
class Permutation {
	const unsigned int size;
	std::vector<T>  perm;

public:
	Permutation(const unsigned int n) : size(n), perm(n) {
		init(n);
	}

	Permutation(const std::string str) : size(str.length()) {
		perm = std::vector<T>(size);
		init(str);
	}

	void init(unsigned int n) {
		for(unsigned int i = 0; i < size; i++)
			perm[i] = i;
	}

	void init(const std::string str) {
		for(unsigned int i = 0; i < size; i++)
			perm[i] = str[i];
		std::sort(perm.begin(), perm.end(), [](char a, char b) { return a < b; } );

	}

	const bool next(void);

	std::string & map(std::string & str) const;

	std::ostream & printOn(std::ostream & out) const;

	friend std::ostream & operator<<(std::ostream & out, const Permutation & p) {
		return p.printOn(out);
	}
};



#endif /* PERMUTATION_H_ */

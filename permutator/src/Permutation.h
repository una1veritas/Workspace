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

class Permutation {
	const unsigned int size;
	std::vector<unsigned int> perm;

public:
	Permutation(const unsigned int n) : size(n), perm(size) {
		init();
	}

	void init(void) {
		for(unsigned int i = 0; i < size; i++)
			perm[i] = i;
	}

	const bool next(void);

	std::string & map(std::string & str) const;

	std::ostream & printOn(std::ostream & out) const;

	friend std::ostream & operator<<(std::ostream & out, const Permutation & p) {
		return p.printOn(out);
	}
};



#endif /* PERMUTATION_H_ */
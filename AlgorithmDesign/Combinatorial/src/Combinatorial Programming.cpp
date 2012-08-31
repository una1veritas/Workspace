//============================================================================
// Name        : Combinatorial.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

struct Combination {
	int * values;
	int varieties;
	int * dimensions;
	int size;

	Combination(int vals[], int vars, int sz) {
		values = vals;
		varieties = vars;
		dimensions = new int[sz];
		size = sz;
	}

	~Combination() {
		delete []dimensions;
	}

	void init() {
		for (int i = 0; i < size; i++ ) {
			dimensions[i] = values[0];
		}
	}

	// Streaming
	//
	friend std::ostream& operator<<(std::ostream& out, const Combination & c) {
		out << "Combination(";
		for(int i = 0; i < c.size; i++) {
			out << c.values[c.dimensions[i]] << ", ";
		}
		out << ") ";
		return out;
	}

};

/*
int main() {
	int a[] = {0,1};
	Combination combi(a, 2, 3);

	cout << combi << std::endl;
	return 0;
}
*/

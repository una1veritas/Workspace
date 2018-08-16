//============================================================================
// Name        : VectorTest.cpp
// Author      : Sin Shimozono
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

int main(const int argc, char * argv[]) {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	vector<pair<int,string>> varray;
	struct compare {
		bool operator()(const pair<int,string> & left, const pair<int,string> & right) {
			return left.first < right.first;
		}
	};
	for (int i = 1; i < argc; ++i) {
		int x;
		x = argv[i][0];
		pair<int,string> p(x,argv[i]);
		vector<pair<int,string>>::iterator itr = lower_bound(varray.begin(), varray.end(), p);
		if ( itr == varray.end() ) {
			varray.insert(itr,p);
		} else if ( itr->first != x ) {
			varray.insert(itr,p);
		}
	}

	for(auto itr = varray.begin(); itr != varray.end(); ++itr) {
		cout << itr->first << ": " << itr->second << ", ";
	}
	cout << endl;
	return 0;
}

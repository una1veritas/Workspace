//============================================================================
// Name        : SubsetEnum.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int * bestSubset(const int b, const int list[], const int n);

int main(int argc, char * argv[]) {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	int n = 0;
	int budget = 500;

	if ( !(argc >= 3) ) {
		cerr << "budget, item1, item2, ...." << endl << endl;
		return -1; // input format error
	}
	budget = atoi(argv[1]);
	n = argc - 2;
	int * list = new int[n+1];
	for(int i = 0; i < n; ++i) {
		list[i] = atoi(argv[i+2]);
	}
	list[n] = 0;

	cout << "Given items: ";
	for (int i = 0; list[i] != 0; ++i) {
		cout << list[i] << ", ";
	}
	cout << endl << "Size: " << n << endl;
	cout << "Budget: " << budget << endl;

	int * cart = bestSubset(budget, list, n);
	int price = 0;

	cout << endl << "In the best cart: ";
	for (int i = 0; i < n; ++i) {
		if ( cart[i] != 0 ) {
			cout << "[" << i << "] " << list[i] << ", ";
			price += list[i];
		}
	}
	cout << endl;
	cout << "Price: " << price << endl;

	delete[] list;
	delete[] cart;
	return 0;
}

int * bestSubset(const int b, const int list[], const int n) {
	int subset[n];
	int sum = 0;
	int * bestcart = new int[n];

	for(int i = 0; i < n; ++i)
		subset[i] = 0;

	while (true) {
		int d;
		for(d = n-1; d >= 0 && subset[d] != 0; --d) {}
		if (d < 0) break;
		subset[d] = 1;
		d += 1;
		for( ; d < n; ++d)
			subset[d] = 0;
		//
		int temp = 0;
		for(int i = 0; i < n; ++i)
			if ( subset[i] != 0 ) { temp += list[i]; }
		if ( temp <= b && temp > sum ) {
			sum = temp;
			for(int i = 0; i < n; ++i)
				bestcart[i] = subset[i];
		}
	}

	return bestcart;
}

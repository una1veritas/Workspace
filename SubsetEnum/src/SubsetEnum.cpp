//============================================================================
// Name        : SubsetEnum.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

int * subsetenum(const int b, const int list[], const int n);

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	int list[] = { 45, 98, 103, 128, 38, 198, 75, 0 };
	int n = 0;
	int budget = 500;

	cout << "Given items: ";
	for (int i = 0; list[i] != 0; ++i) {
		cout << list[i] << ", ";
		n++;
	}
	cout << endl << "Size: " << n << endl;
	cout << "Budget: " << budget << endl;

	int * cart = subsetenum(budget, list, n);
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

	return 0;
}

int * subsetenum(const int b, const int list[], const int n) {
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
		for(int i = 0; i < n; ++i) {
			if ( subset[i] != 0 ) { temp += list[i]; }
			cout << subset[i] << " ";
		}
		cout << ", " << temp << endl;
		if ( temp <= b && temp > sum ) {
			sum = temp;
			for(int i = 0; i < n; ++i)
				bestcart[i] = subset[i];
		}
	}

	return bestcart;
}

//============================================================================
// Name        : SubsetEnum.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

#define USE_PASSCOUNTER
#ifdef USE_PASSCOUNTER
long passcounter = 0;
#endif

int bestSubset(const int b, const int list[], const int n, int cart[]);

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
#ifdef USE_PASSCOUNTER
		++passcounter;
#endif
	}
	list[n] = 0;
	int * cart = new int[n];

	cout << "Given items: ";
	for (int i = 0; list[i] != 0; ++i) {
		cout << list[i] << ", ";
	}
	cout << endl;
	cout << "Budget: " << budget << endl;
	cout << "Num. of items: " << n << endl;

	int price = bestSubset(budget, list, n, cart);

	cout << endl << "The best shopping list: ";
	for (int i = 0; i < n; ++i) {
		if ( cart[i] != 0 ) {
			cout << "Item " << i << " (" << list[i] << " yen), ";
		}
	}
	cout << endl;
	cout << "The price: " << price << endl;

#ifdef USE_PASSCOUNTER
	cerr << endl << "num. of passes = " << passcounter << ". " << endl;
#endif

	delete[] list;
	delete[] cart;
	return 0;
}

int bestSubset(const int b, const int list[], const int n, int cart[]) {
	int subset[n];
	int sum = 0;

	for(int i = 0; i < n; ++i)
		subset[i] = 0;

	while (true) {
		int d;											/* ２進数のインクリメント */
		for(d = n-1; d >= 0 && subset[d] != 0; --d) {} 	/* d = "0 である最下位桁" */
		if (d < 0) break;								/* すべて 1, 終了 */
		subset[d] = 1;									/* d 桁目を 1 に */
		for( d += 1 ; d < n; ++d)						/* その下の桁すべてを 0 に */
			subset[d] = 0;
		//
		int temp = 0;
		for(int i = 0; i < n; ++i) {
			if ( subset[i] != 0 ) { temp += list[i]; }
#ifdef USE_PASSCOUNTER
			++passcounter;
#endif
		}
		if ( temp <= b && temp > sum ) {
			sum = temp;
			for(int i = 0; i < n; ++i)
				cart[i] = subset[i];
		}
	}

	return sum;
}

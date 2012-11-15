//============================================================================
// Name        : Knapsack.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <iomanip>

using namespace std;

#include <time.h>

typedef bool boolean;
typedef unsigned char byte;

int list4[] = { 108, 78, 78, 64, 0 };
int list6[] = { 108, 78, 78, 58, 58, 68, 0 };
int list10[] = { 108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 0 };
int list20[] = { 108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78,
		78, 58, 158, 68, 128, 98, 0 };
int list25[] = { 108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78,
		78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228,  0 };
int list30[] = { 108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78,
		78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228, 238, 198, 298, 64,
		178, 0 };
int list32[] = { 108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78,
		78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228, 238, 198, 298, 64,
		178, 115, 138, 0 };
int list34[] = { 108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78,
		78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228, 238, 198, 298, 64,
		178, 115, 138, 148, 88, 0 };

int * plist = list34;
const int B = 300;

int best(int lastitem, int budget, boolean cart[]) {
//	for(int i = 0; plist[i]; i++) {
//		cout << (cart[i]? 'T' : 'F') << ", ";
//	}
	cout.flush();

	if (lastitem < 0)
		return 0;
	if (lastitem == 0 and plist[lastitem] <= budget) {
		cart[0] = true;
		return plist[lastitem];
	} else if (lastitem == 0 and plist[lastitem] > budget) {
		cart[0] = false;
		return 0;
	}
	// lastitem >= 1

	boolean cart_bought[lastitem], cart_not_bought[lastitem];
	int bought = plist[lastitem]
			+ best(lastitem - 1, budget - plist[lastitem], cart_bought);
	int not_bought = best(lastitem - 1, budget, cart_not_bought);
	if (bought <= budget and bought > not_bought) {
		for (int i = 0; i < lastitem; i++) {
			cart[i] = cart_bought[i];
//			cout << (cart[i]?'T':'F') << ", ";
		}
		cart[lastitem] = true;
//		cout << (cart[lastitem]?'T':'F') << " total = " << bought << endl;
		return bought;
	} else {
		for (int i = 0; i < lastitem; i++) {
			cart[i] = cart_not_bought[i];
//			cout << (cart[i]? 'T' : 'F') << ", ";
		}
		cart[lastitem] = false;
//		cout << (cart[lastitem]?'T':'F') << " total = " << not_bought << endl;
		return not_bought;
	}
}

int main(int argc, char ** argv) {
	int number;

	cout << "Budget: " << B << "," << endl;
	for (number = 0; plist[number] > 0; number++)
		;
	/*
	 cout << number << " Items: " << endl;
	 for (number = 0; plist[number] > 0; number++) {
	 cout << plist[number] << ", ";
	 }
	 cout << endl;
	 */

	boolean shoppingCart[number];

	clock_t swatch = clock();
	int result = best(number - 1, B, shoppingCart);
	swatch = clock() - swatch;

	cout << number << " Items: " << endl;
	for (number = 0; plist[number] > 0; number++) {
		cout << plist[number] << ", ";
	}
	cout << endl;
	cout << "Recommended total price is " << result << " with these items ";
	for (number = 0; plist[number] > 0; number++) {
		if (shoppingCart[number]) {
			cout << number << " ";
		}
	}
	cout << "." << endl;

	cout << "Elapsed " << double(swatch) / 1000000L << ". " << endl;

	return 0; // finished without errors.
}

/*
Budget: 300,
4 Items:
108, 78, 78, 64,
Recommended total price is 264 with these items 0 1 2 .
Elapsed 4e-06.

Budget: 300,
6 Items:
108, 78, 78, 58, 58, 68,
Recommended total price is 292 with these items 0 3 4 5 .
Elapsed 6e-06.

Budget: 300,
10 Items:
108, 78, 78, 58, 58, 68, 38, 39, 58, 128,
Recommended total price is 300 with these items 1 3 4 5 6 .
Elapsed 4.4e-05.

Budget: 300,
20 Items:
108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78, 78, 58, 158, 68, 128, 98,
Recommended total price is 300 with these items 1 3 4 5 6 .
Elapsed 0.045501.

Budget: 300,
25 Items:
108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78, 78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228,
Recommended total price is 300 with these items 1 3 4 5 6 .
Elapsed 1.2132.

Budget: 300,
30 Items:
108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78, 78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228, 238, 198, 298, 64, 178,
Recommended total price is 300 with these items 1 3 4 5 6 .
Elapsed 37.806.


Budget: 300,
32 Items:
108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78, 78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228, 238, 198, 298, 64, 178, 115, 138,
Recommended total price is 300 with these items 1 3 4 5 6 .
Elapsed 159.271.

Budget: 300,
34 Items:
108, 78, 78, 58, 58, 68, 38, 39, 58, 128, 158, 108, 138, 78, 78, 58, 158, 68, 128, 98, 158, 118, 128, 328, 228, 238, 198, 298, 64, 178, 115, 138, 148, 88,
Recommended total price is 300 with these items 1 3 4 5 6 .
Elapsed 647.102.
 */

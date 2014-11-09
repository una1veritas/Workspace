//============================================================================
// Name        : PeriodicalCustom.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

struct curl { int period, tail; };
curl findperiod(const char list[]);

int main(int argc, char * argv[]) {
	cout << "Hello." << endl << endl;

	if (argc <= 1)
		return -1; // error.
	cout << "Input: " << argv[1] << endl; // prints !!!Hello World!!!
	cout << endl;

	char * inputstr = argv[1];
	int n = strlen(inputstr);

	curl p = findperiod(inputstr);

	cout << "pattern: " << endl;
	if ( p.period != 0 ) {
		cout << "[";
		for(int i = 0; i < p.period; ++i) {
			cout << inputstr[i];
		}
		cout << "]^(";
		cout << (n - p.tail)/p.period;
		if ( (n - p.tail) % p.period != 0 ) {
			cout << "+" << ((n-p.tail) % p.period) << "/" << p.period;
		}
		cout << ") ";
	}
	for(int i = n - p.tail; i < n; ++i) {
		cout << inputstr[i];
	}
	cout << endl;
	cout << "period: " << p.period << endl;
	cout << "size: " << p.period + p.tail << endl;
	return 0;
}

curl findperiod(const char list[]) {
	int n = strlen(list);
	curl best = { 0, n };
	for (int p = 1 ; p < n; ++p) {
		int d = n - p;
		for(; d > 0; --d) {
			if ( list[n-d] != list[(n-d) % p])
				break;
		}
		if ( (d < n - p) && (p + d < best.period + best.tail) ) {
			best.period = p; best.tail = d;
		}
	}
	return best;
}

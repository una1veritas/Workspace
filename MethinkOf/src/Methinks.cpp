//============================================================================
// Name        : MethinkOf.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cctype>

using namespace std;

int main() {
	int counter[256];

	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	for(int i = 0; i < 256; ++i) {
		counter[i] = 0;
	}

	char c;
	while ( !cin.eof() ) {
		cin >> c;
		c = toupper(c);
		++counter[int(c)];
	}

	for(int i = 0x20; i < 127; ++i) {
		if ( !isalpha(i) )
			continue;
		if ( islower(i) )
			continue;
		cout << " char " << char(i) <<  "(" << int(i) << ")  = " << counter[i] << endl;
	}

	return 0;
}

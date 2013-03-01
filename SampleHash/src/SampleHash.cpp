//============================================================================
// Name        : SampleHash.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
using namespace std;

int main() {
	cout << "Hello World!!!" << endl; // prints Hello World!!!

	char buf[32];
	while (!cin.eof()) {
		cin >> buf;
		int m, d = 0;
		m = atoi(buf);
		int i;
		for(i = 0; buf[i] != '/' && buf[i]; i++ );
		//cout << "pos = " << i << endl;
		if ( buf[i] == '/' )
			d = atoi(buf+i+1);
		cout << m << " / " << d << ", ";

		int m1, m2, d1, d2;
		m1 = m/10;
		m2 = m - m*10;
		d1 = d/10;
		d2 = d - d1*10;
		cout << "strange = " << (m1+m2)*(d1+d2) << endl;
	}
	return 0;
}

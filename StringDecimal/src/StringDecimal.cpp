//============================================================================
// Name        : StringDecimal.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "stringdecimal.h"
using namespace std;

int main(const int argc, const char * argv[]) {
	cout << "Hello World!!!" << endl; // prints Hello World!!!

	StringDecimal d = "100";
	StringDecimal a = argv[1];
	StringDecimal x = 1.10582;

	cout << "d = " << d << std::endl;
	cout << "a = " << a << std::endl;
	cout << "x = " << x << std::endl;
	cout << a.add(x) << std::endl;
	return 0;
}

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

	StringDecimal d = "-1200";
	StringDecimal a = -1294L;
	StringDecimal x = 1.10582;

	cout << "d = " << d << std::endl;
	cout << "a = " << a << std::endl;
	cout << "x = " << x << std::endl;
	cout << (d < a) << std::endl;
	cout << (a < d) << std::endl;
	cout << (a < x) << std::endl;
	cout << (d < x) << std::endl;
	cout << (x < a) << std::endl;
	return 0;
}

//============================================================================
// Name        : StringStreamTest.cpp
// Author      : Sin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <sstream>

using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	string buf = "abc def,ghi";

	buf.append("tea? ");
	buf.append("coffie? ");

	stringstream ss(buf);
	string token;
	while (ss >> token)
	{
	    printf("%s\n", token.c_str());
	}

	return 0;
}

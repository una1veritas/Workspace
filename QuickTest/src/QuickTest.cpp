/*
 ============================================================================
 Name        : QuickTest.cpp
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C++,
 ============================================================================
 */

#include <iostream>
#include <stdlib.h>

using namespace std;

int main(void) {
	char buf[32];
	long val;

	cout << "Hello World" << endl; /* prints Hello World */
	cin >> buf;
	cout << "Input: " << buf << endl;
	val = strtol(buf, NULL, 16);
	cout << "Interpreted to: " << val << endl;
	return 0;
}

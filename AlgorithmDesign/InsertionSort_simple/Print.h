/*
 *  WProgram.h
 *  InsertionSort_simple
 *
 *  Created by ���� �^�� on 10/10/09.
 *  Copyright 2010 ��B�H�Ƒ�w���H�w��. All rights reserved.
 *
 */

#include <iostream>

void print(const char c) {
	std::cout << c;
}
void print(const long i) {
	std::cout << i;
}
void print(const int i) {
	std::cout << i;
}
void print(const char * s) {
	std::cout << s;
}

void println() {
	std::cout << std::endl;
}

void println(const int i)	{
	print(i);
	println();
}

void println(const char * s)	{
	print(s);
	println();
}



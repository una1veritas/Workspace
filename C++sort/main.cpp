/*
 * main.cpp
 *
 *  Created on: 2022/05/24
 *      Author: sin
 */
#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>
using namespace std;

bool compare(const string & lhd, const string & rhd) {
	for(int i = 0; i < 3 and lhd[i] != 0 and rhd[i] != 0; ++i) {
		if ( toupper(lhd[i]) < toupper(rhd[i]) )
			return true;
	}
	return false;
}

int main(int argc, char * argv[]) {
	if ( argc <= 1 ) {
		printf("ソートする要素を一つ以上引数で与えてください．\n");
		return EXIT_FAILURE;
	}
	int n = argc - 1;
	char * a[n];
	for(int i = 0; i < n; ++i) {
		a[i] = argv[i+1];
		printf("%s, ", a[i]);
	}
	printf("\n");

	std::sort(a, a+n, compare);

	for(int i = 0; i < n; ++i) {
		printf("%s, ", a[i]);
	}
	printf("\n");
	return EXIT_SUCCESS;
}

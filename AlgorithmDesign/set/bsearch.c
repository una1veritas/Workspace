/*
 ============================================================================
 Name        : bsearch.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
//#include <iostream>

int binary_search(float a[], float target, int aStart, int aEnd) {
	int index, start = aStart, end = aEnd;
	while ( start < end ) {
		index = (start + end) / 2;
		if ( a[index] == target ) {
			return index;
		} else if ( a[index] > target ) {
			end = index - 1;
		} else {
			start = index + 1;
		}
	}
	return aEnd;
}

void bubble(float a[], int start, int end) {
	float t;
	for(int i = 0; i < end-1; i++) {
		for(int j = 0; j < end-i-1; j++) {
			if ( !(a[j] < a[j+1]) ) {
				t = a[j+1];
				a[j+1] = a[j];
				a[j] = t;
			}
		}
	}
}

int main(int argc, char * argv[]) {

	//init();
	// program body

	printf("Hi friends!\n");
//	_delay_ms(500);

	float a[10] = {
			0.4,
			0.33,
			2.221,
			3.141592,
			2.7182818,
			6.02,
			1.4141356,
			0.1,
			0.991,
			1.333
	};
	bubble(a, 0, 10);
	printf("Bubble sort:");
	for(int i = 0; i < 10; i++) {
		printf("%f,\n", a[i]);
	}

	printf("\n");
//	_delay_ms(500);
	printf("searching...\n");

	int place = binary_search(a, 0.991, 0, 10);

	printf("Result: \n");
	printf("%d.\n",place);

	return 0;
}

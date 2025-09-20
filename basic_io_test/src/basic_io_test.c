/*
 ============================================================================
 Name        : basic_io_test.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

char upper_case(char a) {
	if ( a < 'a' )
		return a;
	if ( a < 'z'+1 )
		return a & 0xdf;
	return a;
}

char hex2nib(char a) {
	if ( a < '0' )
		return a | 0x80;
	if (a < '9' + 1) {
		a -= '0';
		return a;
	}
	a = upper_case(a);
	if ( a < 'A' )
		return a | 0x80;
	if ( a < 'F' + 1) {
		a -= 'A' - 10;
		return a;
	}
	return a | 0x80;
}

int main(const int argc, const char * argv[]) {
	puts("Hello World!!!"); /* prints Hello World!!! */

	if (argc > 1) {
		printf("input %s\n", argv[1]);
		for(const char * ptr = argv[1]; *ptr; ++ptr) {
			unsigned char u = hex2nib(*ptr);
			printf("%d ", u);
		}
	}

	printf("\nfor all the chars.\n");
	for(unsigned char c = 0x20; c < 0x7f; ++c) {
		unsigned char u = hex2nib(c);
		printf("%c --> %d\n", c, u);
	}

	return EXIT_SUCCESS;
}

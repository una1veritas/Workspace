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

char hextonibble(const char c) {
	char a = c;
	if (a > 'a' - 1)
		a &= 0xdf;
	if (a < '9' + 1) {
		a -= '0';
		if ( a < 0 )
			return -1;
	} else if (a < 'F' + 1) {
		a -= 'A';
		if ( a < 0 )
			return -1;
		a += 10;
	}
	if ( a > 15 )
		return -1;
	return a;
}

int main(const int argc, const char * argv[]) {
	puts("Hello World!!!"); /* prints Hello World!!! */

	if (argc <= 1)
		return EXIT_SUCCESS;
	printf("input %s\n", argv[1]);
	for(const char * ptr = argv[1]; *ptr; ++ptr) {
		char u = hextonibble(*ptr);
		printf("%d ", u);
	}

	printf("\nfor all the chars.\n");
	for(char c = 0x20; c < 0x7f; ++c) {
		char u = hextonibble(c);
		if ( u == -1 ) {
			printf("%c --> error\n", c);
		} else {
			printf("%c --> %d\n", c, u);
		}
	}

	return EXIT_SUCCESS;
}

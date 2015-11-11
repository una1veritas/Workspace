/*
 ============================================================================
 Name        : Merge_Sort_wo_recursion.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
	int input_size = 0;

	if (argc < 2) {
		puts("Too few arguments.\n");
		return EXIT_FAILURE;
	}

	printf("argc = %d.\n", argc);
	for(int i = 1; i < argc; i++) {
		puts("yah.\n");
		input_size++;
	}

	printf("input size: %d.\n", input_size);

	return EXIT_SUCCESS;
}

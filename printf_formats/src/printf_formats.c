/*
 ============================================================================
 Name        : printf_formats.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
	const char * str = "Hello World!!!"; /* prints Hello World!!! */

	char buf[64];
	int c = 5;
	int len = snprintf(buf, 12, "%*c%-20s", c, ' ', str);
	printf("len = %d\n", len);
	printf("strlen = %d\n", (int) strlen(buf));
	puts(buf);
	return EXIT_SUCCESS;
}

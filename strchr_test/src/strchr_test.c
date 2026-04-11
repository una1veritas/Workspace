/*
 ============================================================================
 Name        : strchr_test.c
 Author      : Sin Shimozono
 Version     :
 Copyright   : GPL
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define YES 1
#define NO 0

int any1(char c, char * str) {
	if ( strchr(str, c) == NULL)
		return NO;
	return YES;
}

int any2(
char	c,
char	*str
) {
	while(*str != 0)
		if(*str++ == c)
			return(YES);
	return(NO);
}

int main(void) {
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	char target[256];
	for(int i = 0; i < 256; ++i){
		target[i] = (char) (i + 1) & 0xff;
	}

	for(char c = '\r'; c <= '~'; c++){
		printf("any1 = %c %d, ", c, any1(c, target));
		printf("any2=  %c %d\n", c, any2(c, target));
	}
	return EXIT_SUCCESS;
}

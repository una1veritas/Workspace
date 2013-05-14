/*
 ============================================================================
 Name        : Sample_1.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
/*
 	if ( argc != 2 ) {
		fprintf(stderr,"bad arguments %d.\n", argc);
		return EXIT_FAILURE;
	}

	fprintf(stdout, "input: %s\n", argv[1]);
*/
	int count = 0;
	char * p = argv[1];
	for( ; *p ; p++) {
		if ( *p == '0' )
			count--;
		else
			count++;
		//
		printf("%c [%+d], ", *p, count);
	}
	//
	printf("\n");
	if ( count == 0 )
		printf("accept\n");
	else
		printf("reject\n");
	return EXIT_SUCCESS;
}

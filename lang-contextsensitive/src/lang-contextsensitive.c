/*
 ============================================================================
 Name        : lang-contextsensitive.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {

	char * inputstr;
	if (argc != 2) {
		puts("Give me a string.\n");
		return EXIT_FAILURE;
	}

	inputstr = argv[1];
	printf("Input: %s\n", inputstr);

	int wlen = strlen(inputstr)+2;
	char * work = (char*) malloc(sizeof(char)*wlen);
	for(int i = 0; i < wlen; i++)
		work[i] = 0;

	char * whead = work+1;
	char * inhead = inputstr;

	while ( *inputstr == 'a' ) {
		*whead = 'a';
		inputstr++;
		whead++;
	}
	whead--;
	while ( *whead != 0)
		whead--;
	whead++;
	printf("1 %s\n",whead);
	while ( *inputstr == 'b' && *whead == 'a') {
			inputstr++;
			whead++;
	}
	if ( *inputstr != 'c' || *whead != 0 )
		goto exit_reject;

	whead--;
	while ( whead != 0)
		whead--;
	whead++;
	printf("2 %s\n",whead);
	while ( *inputstr == 'c' && *whead == 'a') {
			inputstr++;
			whead++;
	}
	if ( *inputstr != 0 || *whead != 0 )
		goto exit_reject;

	puts("Accept.\n");

	free(work);
	return EXIT_SUCCESS;

exit_reject:
	free(work);
	puts("Reject.\n");
	return EXIT_FAILURE;
}

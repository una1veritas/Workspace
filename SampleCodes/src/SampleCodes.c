/*
 ============================================================================
 Name        : SampleCodes.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {
	const char * filename;
	FILE * infp;
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	if ( argc <= 1 ) { // given no arguments
		fprintf(stdout, "No arguments given.");
		//readData(stdin);
	} else { // supplied arguments
		filename = argv[1];
		fprintf(stdout, "Trying to open file \"%s\"...\n", filename);
		infp = fopen(filename, "r");
		if ( infp == NULL ) {
			fprintf(stderr, "Opening file \"%s\" failed!\n", filename);
			goto BAD_ENDING;
		}
		readData(infp);
		fclose(infp);
	}

BAD_ENDING:
	return EXIT_SUCCESS;
}

int readData(FILE * fp) {
	int counter = 0;
	char buff[64];
	char * ptr;
	long monl, datel;

	while ( !feof(fp) ) {
		fgets(buff, 64, fp);
		fprintf(stdout, "contents: %s\n", buff);
		ptr = buff;
		monl = strtol(ptr, ptr, 10);
		if ( *ptr == '/') ptr++;
		datel = strtol(ptr, ptr, 10);
		fprintf(stdout, "%d, ", (int)val);
	}
	return 0;
}

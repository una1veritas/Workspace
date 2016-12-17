/*
 * textfromfile.c
 *
 *  Created on: 2016/11/20
 *      Author: Sin Shimozono
 */

#include "textfromfile.h"

int textfromfile(const char * filename, const unsigned int maxsize, char * text) {
	FILE * fp;
	int pos;
	char * ptr;

	fp = fopen(filename, "r");
	if ( fp == NULL ) {
		fprintf(stderr, "error: open file %s failed.\n", filename);
		fflush(stderr);
		return EXIT_FAILURE;
	} else {
		for(ptr = text, pos = 0; pos < maxsize; ++ptr, ++pos) {
			int c = fgetc(fp);
			if ( c == EOF )
				break;
			*ptr = (char) c;
		}
		if (pos < maxsize) {
			*ptr = 0;
		} else {
			text[maxsize-1] = 0;
		}
		return EXIT_SUCCESS;
	}
}

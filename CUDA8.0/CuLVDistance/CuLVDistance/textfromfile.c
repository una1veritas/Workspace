/*
 * textfromfile.c
 *
 *  Created on: 2016/11/20
 *      Author: Sin Shimozono
 */

#include "textfromfile.h"

unsigned long textfromfile(const char * filename, const unsigned long maxsize, char * text) {
	FILE * fp;
	unsigned long count = 0;
	char * ptr;

	fp = fopen(filename, "r");
	if ( fp == NULL ) {
		fprintf(stderr, "error: open file %s failed.\n", filename);
		fflush(stderr);
		return count;
	} else {
		for(ptr = text, count = 0; count < maxsize; ++ptr, ++count) {
			int c = fgetc(fp);
			if ( c == EOF )
				break;
			*ptr = (char) c;
		}
		if (count < maxsize) {
			*ptr = 0;
		}
		return count;
	}
}

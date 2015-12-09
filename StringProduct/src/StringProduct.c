/*
 ============================================================================
 Name        : StringProduct.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <math.h>

unsigned long iproduct(const char * p, const char * t, const unsigned int len);

int main(int argc, char * argv[]) {

	// prepare
	const int arg_count = argc - 1;
	char ** arg_val = argv + 1;

	if ( arg_count < 2 ) {
		printf("Unexpected number of args.\n");
		return EXIT_FAILURE;
	}
	const char * pattern = arg_val[0];
	const unsigned int plen = strlen(pattern);
	const char * text = arg_val[1];
	const unsigned int tlen = strlen(text);

	printf("pattern = %s (%d), text = %s (%d)\n", pattern, plen, text, tlen);
	if ( plen > tlen) {
		printf("the pattern is longer than the text.\n");
		return EXIT_FAILURE;
	}

	// main
	const unsigned long pnorm_sq = iproduct(pattern, pattern, plen);

	for(unsigned int pos = 0; pos < tlen - plen + 1; pos++) {
		printf("pos = %d: ", pos);

		printf("'");
		for(int i = 0; i < plen; i++) {
			printf("%c", text[pos + i]);
		}
		printf("'\n");
		const unsigned long tnorm_sq = iproduct(text+pos, text+pos, plen);
		const unsigned long iprod = iproduct(pattern, text+pos, plen);
		printf("pn %lu, tn %lu, product: %5.4f\n", pnorm_sq, tnorm_sq, (double) iprod / sqrt(pnorm_sq) / sqrt(tnorm_sq) ) ;
	}

	printf("end.\n\n");
	return EXIT_SUCCESS;
}

unsigned long iproduct(const char * p, const char * t, const unsigned int len) {
	unsigned long sum = 0;

	for(int i = 0; i < len; i++) {
		sum = ((unsigned long) p[i]) * ((unsigned long) t[i]);
	}
	return sum;
}

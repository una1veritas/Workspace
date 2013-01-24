/*
 * main.c
 *
 *  Created on: 2013/01/24
 *      Author: sin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define NOERROR 0
unsigned char modpow(unsigned char c, long p, unsigned char d);
char * modpowString(char * str, long p, unsigned char d);

int main(int argc, char * argv[]) {
	char * mess = argv[1];
	int messlen = strlen(mess);
//	char * str = (char*) malloc(messlen);
//	char * ptr;
	int i, t;

	printf("input `%s' of length %d,\n", mess, messlen);

	char c;
	for(i = 0, t = 0; mess[i] ; i++) {
		c = mess[i];
		if ( isalpha(c) ) {
			mess[t++] = toupper(c);
		} else if ( c == ' ' ) {
			mess[t++] = '@';
		}
	}
	mess[t] = 0;
	printf("transformed into `%s'.\n", mess);

	modpowString(mess, 15, 221);
	printf("encoded as `%s'.\n", mess);
	modpowString(mess, 11, 221);
	printf("decoded to `%s'.\n", mess);

//	free(str);
	return NOERROR;
}

unsigned char modpow(unsigned char c, long p, unsigned char d) {
	long t = c;
	long i;
	for(i = 0; i < p; i++) {
		t = (t*c) % d;
	}
	return (unsigned char) t;
}

char * modpowString(char * str, long p, unsigned char d) {
	int i;
	for ( i = 0; str[i]; i++) {
		str[i] = modpow((unsigned char) (str[i] - '@' + 2), p, d) + '@' - 2;
	}
	return str;
}


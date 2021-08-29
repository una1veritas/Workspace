/*
 ============================================================================
 Name        : dfa.c
 Author      : Sin Shimozono
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* dfa で使用する定数 */
#define TRANSITION_NOT_DEFINED 	0
#define STATE_NOT_FINAL 		0
#define STATE_FINAL 			1
#define STATE_LIMIT	 			128
#define ALPHABET_LIMIT 			128

typedef struct {
	/* the set of states must be a subset of {1,...,127}. */
	/* an element of the finite alphabet must be a char in (1,...,127). */
	char delta[STATE_LIMIT][ALPHABET_LIMIT];
	char initial;
	char finals[STATE_LIMIT]; /* 0 -> not a final state, 1 -> a final state. */

	char current;
} dfa;

void dfa_define(dfa * mp,
		char * trans,
		char * initial,
		char * finals) {
	char triple[7];
	char * ptr = trans;
	/* データ構造の初期化 */
	for(int i = 0; i < STATE_LIMIT; ++i) { /* = 1 からでもよいが */
		for(int a = 0; a < ALPHABET_LIMIT; ++a) { /* = 1 からでもよいが */
			mp->delta[i][a] = TRANSITION_NOT_DEFINED;
		}
	}
	for(int i = 0; i < STATE_LIMIT; ++i) {
		mp->finals[i] = STATE_NOT_FINAL;
	}
	/* 定義の読み取り */
	while ( sscanf(ptr, "%[^,]", triple) ) {
		//printf("%s\n", triple);
		mp->delta[(int)triple[0]][(int)triple[1]] = triple[2];
		ptr += strlen(triple);
		if (*ptr == 0) break;
		++ptr;
		//printf("%s\n", ptr);
	}
	mp->initial = *initial;
	for(ptr = finals; *ptr ; ++ptr) {
		mp->finals[(int)*ptr] = STATE_FINAL;
	}
}

void dfa_reset(dfa * mp) {
	mp->current = mp->initial;
}

char dfa_transfer(dfa * mp, char a) {
	mp->current = mp->delta[(int)mp->current][(int)a];
	return mp->current;
}

int dfa_accepting(dfa * mp) {
	return mp->finals[(int)mp->current] == STATE_FINAL;
}

void dfa_print(dfa * mp) {
	char states[STATE_LIMIT];
	char alphabet[ALPHABET_LIMIT];

	for(int i = 0; i < STATE_LIMIT; ++i) {
		states[i] = 0;
	}
	for(int a = 0; a < ALPHABET_LIMIT; ++a) {
		alphabet[a] = 0;
	}
	for(int i = 0; i < STATE_LIMIT; ++i) {
		for(int a = 0; a < ALPHABET_LIMIT; ++a) {
			if ( mp->delta[i][a] ) {
				states[i] = 1;
				states[(int)mp->delta[i][a]] = 1;
				alphabet[a] = 1;
			}
		}
	}
	printf("dfa(\n");
	printf("states = {");
	int the1st = 1;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if (states[i]) {
			if ( !the1st ) {
				printf(", ");
			}
			printf("%c", (char) i);
			the1st = 0;
		}
	}
	printf("},\n");
	printf("alphabet = {");
	the1st = 1;
	for(int i = 0; i < ALPHABET_LIMIT; ++i) {
		if (alphabet[i]) {
			if ( !the1st ) {
				printf(", ");
			}
			printf("%c", (char) i);
			the1st = 0;
		}
	}
	printf("},\n");
	printf("delta = \n");
	printf("state symbol| next\n");
	printf("------------+------\n");
	for(int i = 0; i < STATE_LIMIT; ++i) {
		for(int a = 0; a < ALPHABET_LIMIT; ++a) {
			if ( mp->delta[i][a] ) {
				printf("  %c  ,  %c   |  %c\n",i,a,mp->delta[i][a]);
			}
		}
	}
	printf("------------+------\n");
	printf("initial state = %c\n", mp->initial);
	printf("accepting states = {");
	the1st = 1;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if (mp->finals[i]) {
			if ( !the1st ) {
				printf(", ");
			}
			printf("%c", (char) i);
			the1st = 0;
		}
	}
	printf("}\n)\n");
}

int dfa_run(dfa * mp, char * inputstr) {
	char * ptr = inputstr;
	dfa_reset(mp);
	printf("     -> %c", mp->current);
	for ( ; *ptr; ++ptr) {
		dfa_transfer(mp, *ptr);
		printf(", -%c-> %c", *ptr, mp->current);
	}
	if ( dfa_accepting(mp) ) {
		printf(", accepted.\n");
		return STATE_FINAL;
	} else {
		printf(", rejected.\n");
		return STATE_NOT_FINAL;
	}
}

int main(int argc, char **argv) {
	dfa M;
	// printf("M is using %lld bytes.\n", sizeof(M));
	dfa_define(&M, "0a1,0b2,1a2,1b0", "0", "0");
	dfa_print(&M);
	dfa_run(&M, "ababab");
	printf("bye.\n");
	return EXIT_SUCCESS;
}

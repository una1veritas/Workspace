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

typedef struct {
	/* the set of states must be a subset of {1,...,127}. */
	/* an element of the finite alphabet must be an ascii char in (1,...,127). */
	char delta[32767];
	char initialstate;
	char finalstate[128]; /* 0 -> not a final state, 1 -> a final state. */

	char currentstate;
} dfa;
const int TRANSITION_LIMIT = 0x7fff;
const int TRANSITION_NOT_DEFINED = 0;
const char STATE_NOT_FINAL = 0;
const char STATE_FINAL = 1;
const char STATE_LIMIT = 0x7f;
const char ALPHABET_LIMIT = 0x7f;

void dfa_initialize(dfa * mp) {
	mp->initialstate = 0;
	for(unsigned int i = 0; i <= TRANSITION_LIMIT; ++i) {
		mp->delta[i] = TRANSITION_NOT_DEFINED;
	}
	for(unsigned int i = 0; i <= STATE_LIMIT; ++i) {
		mp->finalstate[i] = STATE_NOT_FINAL;
	}
}

void dfa_define(dfa * mp,
		char * trans,
		char * initial,
		char * finals) {
	char triple[7];
	char * ptr = trans;
	while ( sscanf(ptr, "%[^,]", triple) ) {
		printf("%s\n", triple);
		mp->delta[((unsigned int)triple[0])<<8 | triple[1]] = triple[2];
		ptr += strlen(triple);
		if (*ptr == 0) break;
		++ptr;
		//printf("%s\n", ptr);
	}
	mp->initialstate = *initial;
	for(ptr = finals; *ptr ; ++ptr) {
		mp->finalstate[(int)*ptr] = STATE_FINAL;
	}
}

void dfa_run(dfa * mp, char * inputstr) {
	char * ptr = inputstr;
	mp->currentstate = mp->initialstate;
	printf("state: %c\n", mp->currentstate);
	for ( ; *ptr; ++ptr) {
		mp->currentstate = mp->delta[((unsigned int)mp->currentstate)<<8 | *ptr];
		printf("state: %c\n", mp->currentstate);
	}
	if (mp->finalstate[(int)mp->currentstate]) {
		printf("accepted.\n");
	}
}

int main(int argc, char **argv) {
	dfa M;
	// printf("M is using %lld bytes.\n", sizeof(M));
	printf("initializing ... ");
	dfa_initialize(&M);
	printf("done.\n");

	dfa_define(&M, "0a1,0b2,1a2,1b0", "0", "0");

	dfa_run(&M, "ababab");
	printf("bye.\n");
	return EXIT_SUCCESS;
}

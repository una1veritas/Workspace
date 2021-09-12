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

/* 便宜的に設定する上限 */
#define STATE_LIMIT	 			128
#define ALPHABET_LIMIT 			128

typedef struct {
	/* 状態は ASCII 文字, 状態の集合は char 型の {1,...,127} の部分集合に限定. */
	/* 状態 0 は black hole state ブラックホールとして扱う．  */
	/* 文字は ASCII 文字, 有限アルファベットは char 型の {1,...,127} の部分集合に限定. */
	/* 文字 0 は文字列終端記号（ヌル文字）として処理されるため使用しない. */
	char delta[STATE_LIMIT][ALPHABET_LIMIT];	/* 遷移関数 */
	char initial; 								/* 初期状態 */
	char finals[STATE_LIMIT]; 					/* 最終状態を表すフラグの表 */

	char current;
} dfa;

/* マーカーとして使用する定数 */
#define TRANSITION_NOT_DEFINED 	0
#define STATE_IS_NOT_FINAL 		0
#define STATE_IS_FINAL 			1

/* 定義文字列から dfa を初期化 */
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
		mp->finals[i] = STATE_IS_NOT_FINAL;
	}
	/* 定義の三つ組みを読み取る */
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
		mp->finals[(int)*ptr] = STATE_IS_FINAL;
	}
}

void dfa_reset(dfa * mp) {
	mp->current = mp->initial;
}

char dfa_transfer(dfa * mp, char a) {
	if (mp->delta[(int)mp->current][(int)a] != 0) /* defined */
		mp->current = mp->delta[(int)mp->current][(int)a];
	/* else
		if omitted, go to and self-loop in the black hole state. */
	return mp->current;
}

/* 受理状態にあるか */
int dfa_accepting(dfa * mp) {
	return mp->finals[(int)mp->current] == STATE_IS_FINAL;
}

/* dfa の定義を印字 */
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
			printf("'%c'", (char) i);
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
			printf("'%c'", (char) i);
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
				printf(" '%c' , '%c'  | '%c'\n",i,a,mp->delta[i][a]);
			}
		}
	}
	printf("------------+------\n");
	printf("initial state = '%c'\n", mp->initial);
	printf("accepting states = {");
	the1st = 1;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if (mp->finals[i]) {
			if ( !the1st ) {
				printf(", ");
			}
			printf("'%c'", (char) i);
			the1st = 0;
		}
	}
	printf("}\n)\n");
}

/* 文字列にたいして dfa を走らせる */
int dfa_run(dfa * mp, char * inputstr) {
	char * ptr = inputstr;
	printf("On input '%s' :\n", ptr);
	dfa_reset(mp); 	/* 状態を初期状態にする */
	printf("   -> '%c'", mp->current);
	for ( ; *ptr; ++ptr) {
		dfa_transfer(mp, *ptr); 	/* 遷移する */
		printf(", -'%c'-> '%c'", *ptr, mp->current);
	}
	if ( dfa_accepting(mp) ) {
		/* 受理した */
		printf(", \naccepted.\n");
		return STATE_IS_FINAL;
	} else {
		/* 却下した */
		printf(", \nrejected.\n");
		return STATE_IS_NOT_FINAL;
	}
}

int command_arguments(int , char ** , char * , char * , char * , char *);

int main(int argc, char **argv) {
	char * delta = "0a1,0b2,1a2,1b0", *initial = "0", *finals = "0";
	char input_buff[1024] = "abaababbab";
	if ( command_arguments(argc, argv, delta, initial, finals, input_buff) )
		return 1;

	dfa M;
	printf("M is using %0.2f Kbytes.\n\n", (double)(sizeof(M)/1024) );
	dfa_define(&M, delta, initial, finals);
	dfa_print(&M);
	if (strlen(input_buff))
		dfa_run(&M, input_buff);
	else {
		/* 標準入力から一行ずつ，入力文字列として走らせる */
		while( fgets(input_buff, 1023, stdin) ) {
			for(char * p = input_buff+strlen(input_buff); *--p == '\n'; *p = '\0') ; /* 改行は無視 */
			dfa_run(&M, input_buff);
		}
	}
	printf("Bye.\n");
	return 0;
}


int command_arguments(int argc, char * argv[], char * delta, char * initial, char * finals, char * input) {
	if (argc > 1) {
		if (strcmp(argv[1], "-h") == 0 ) {
			printf("usage: command \"transition triples\" \"initial state\" \"final states\" (\"input string\")\n");
			printf("example: dfa.exe \"%s\" \"%s\" \"%s\"\n", delta, initial, finals);
			return 1;
		} else if (argc == 4 || argc == 5 ) {
			delta = argv[1]; initial = argv[2]; finals = argv[3];
			if (argc == 5 )
				strcpy(input, argv[4]);
			else
				input[0] = '\0';
		} else {
			printf("Illegal number of arguments.\n");
			return 1;
		}
	} else {
		printf("define M by buily-in example: \"%s\" \"%s\" \"%s\"\n", delta, initial, finals);
		printf("(Use 'command -h' to get a help message.)\n");
	}
	return 0;
}


/*
 ============================================================================
 Name        : nfa.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <ctype.h>

/* 便宜的に設定した上限 */
#define STATE_LIMIT	 			64
#define ALPHABET_LIMIT 			128

typedef uint64_t bset64; 	/* 符号なし整数型をビット表現で集合として使用する */
typedef struct {
	/* 状態は 数字，英大文字を含む空白 (0x20) から _ (0x5f) までの一文字で表す
	 * 正の整数 {0,...,63} の要素に限定. */
	/* 文字は ASCII 文字, 有限アルファベットは char 型の {1,...,127} の要素に限定. */
	bset64 delta[STATE_LIMIT][ALPHABET_LIMIT];	/* 遷移関数 : Q x Σ -> 2^Q*/
	char  initial; 								/* 初期状態 */
	bset64 finals;			 					/* 最終状態を表すフラグの表 */

	bset64 current;
} nfa;

// ユーティリティ
#define char2state(x)  ((x) - 0x30)
#define state2char(x)  ((x) + 0x30)
char * bset64_str(bset64 bits, char * buf) {
	char * ptr = buf;
	ptr += sprintf(ptr, "{");
	int cnt = 0;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if (bits>>i & 1) {
			if (cnt) ptr += sprintf(ptr, ", ");
			ptr += sprintf(ptr, "%c", state2char(i));
			++cnt;
		}
	}
	sprintf(ptr, "}");
	return buf;
}

/* マーカーとして使用する定数 */
#define TRANSITION_NOT_DEFINED 	0
#define STATE_IS_NOT_FINAL 		0
#define STATE_IS_FINAL 			1

/* 定義文字列から nfa を初期化 */
void nfa_define(nfa * mp,
		char * trans,
		char * initial,
		char * finals) {
	char triplex[72];
	//char buf[72];
	char * ptr = trans;
	/* データ構造の初期化 */
	for(int i = 0; i < STATE_LIMIT; ++i) {
		for(int a = 0; a < ALPHABET_LIMIT; ++a) { /* = 1 からでもよいが */
			mp->delta[i][a] = 0; 	/* 空集合に初期化 */
		}
	}
	mp->finals = 0;
	/* 定義の三つ組みを読み取る */
	while ( sscanf(ptr, "%[^,]", triplex) ) {
		//printf("def: %s ", triplex);
		int stat = char2state(triplex[0]);
		int symb = triplex[1];
		for(char * x = triplex+2; *x; ++x) { 	/* 遷移先は複数記述可能 */
			mp->delta[stat][symb] |= 1<<char2state(*x);
		}
		//printf("%x, %c -> ", stat, symb);
		//printf("%s\n", bset64_str(mp->delta[stat][symb], buf));
		ptr += strlen(triplex);
		if (*ptr == 0) break;
		++ptr; 	/* , を読み飛ばす */
		//printf("%s\n", ptr);
	}
	mp->initial = char2state(*initial); 	/* 初期状態は１つ */
	for(ptr = finals; *ptr ; ++ptr) {
		mp->finals |= 1<<char2state(*ptr);
	}
}

void nfa_reset(nfa * mp) {
	mp->current = 1<<mp->initial;
}

bset64 nfa_transfer(nfa * mp, char a) {
	bset64 next = 0;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if ((mp->current & (1<<i)) != 0) {
			if (mp->delta[i][(int)a] != 0) /* defined */
				next |= mp->delta[i][(int)a];
			//else /* if omitted, go to and self-loop in the ghost state. */
		}
	}
	return mp->current = next;
}

int nfa_accepting(nfa * mp) {
	return (mp->finals & mp->current) != 0;
}

void nfa_print(nfa * mp) {
	bset64 states;
	char alphabet[ALPHABET_LIMIT];
	char buf[160];

	states = 0;
	for(int a = 0; a < ALPHABET_LIMIT; ++a) {
		alphabet[a] = 0;
	}
	for(int i = 0; i < STATE_LIMIT; ++i) {
		for(int a = 0; a < ALPHABET_LIMIT; ++a) {
			if ( mp->delta[i][a] ) {
				states |= 1<<i;
				states |= (int)mp->delta[i][a];
				alphabet[a] = 1;
			}
		}
	}
	printf("nfa(\n");
	printf("states = %s\n", bset64_str(states, buf));
	printf("alphabet = {");
	int the1st = 1;
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
				printf("  %c  ,  %c   | %s \n",state2char(i), a, bset64_str(mp->delta[i][a], buf));
			}
		}
	}
	printf("------------+------\n");
	printf("initial state = %x\n", mp->initial);

	printf("accepting states = %s\n", bset64_str(mp->finals, buf));

	printf("\n");
}


int nfa_run(nfa * mp, char * inputstr) {
	char * ptr = inputstr;
	char buf[128];
	printf("run on '%s' :\n", ptr);
	nfa_reset(mp);
	printf("     -> %s", bset64_str(mp->current, buf));
	for ( ; *ptr; ++ptr) {
		nfa_transfer(mp, *ptr);
		printf(", -%c-> %s", *ptr, bset64_str(mp->current, buf));
	}
	if ( nfa_accepting(mp) ) {
		printf(", \naccepted.\n");
		return STATE_IS_FINAL;
	} else {
		printf(", \nrejected.\n");
		return STATE_IS_NOT_FINAL;
	}
}

int command_arguments(int argc, char * argv[], char * delta, char * initial, char * finals, char * input);

int main(int argc, char **argv) {
	char * delta = "0a01,0b0,1b2,2b3,3a3,3b3", *initial = "0", *finals = "3";
	char input_buff[1024] = "abaababaab";
	if ( command_arguments(argc, argv, delta, initial, finals, input_buff) )
		return 1;

	nfa M;
	printf("M is using %0.2f Kbytes.\n\n", (double)(sizeof(M)/1024) );
	nfa_define(&M, delta, initial, finals);
	nfa_print(&M);
	if (strlen(input_buff))
		nfa_run(&M, input_buff);
	else {
		/* 標準入力から一行ずつ，入力文字列として走らせる */
		while( fgets(input_buff, 1023, stdin) ) {
			for(char * p = input_buff+strlen(input_buff); *--p == '\n'; *p = '\0') ; /* 改行は無視 */
			nfa_run(&M, input_buff);
		}
	}
	printf("bye.\n");
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

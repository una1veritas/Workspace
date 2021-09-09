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
#define STATE_LIMIT	 			16
#define ALPHABET_LIMIT 			128

typedef uint16_t set16; 	/* 符号なし整数型をビット表現で集合として使用する */
typedef struct {
	/* 状態は16進数1桁の整数, 状態の集合は {0,...,15} の部分集合に限定. */
	/* 文字は ASCII 文字, 有限アルファベット Σ は char 型の {1,...,127} の部分集合に限定. */
	set16 delta[STATE_LIMIT][ALPHABET_LIMIT];	/* 遷移関数 : Q x Σ -> 2^Q*/
	char  initial; 								/* 初期状態 */
	set16 finals;			 					/* 最終状態を表すフラグの表 */

	set16 current;
} nfa;

// ユーティリティ
#define hexchar2int(x)  ((x) <= '9' ? (x) - '0' : toupper(x) - 'A' + 10)
char * set16tostr(set16 bits, char * buf) {
	char * ptr = buf;
	ptr += sprintf(ptr, "{");
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if (bits>>i & 1)
			ptr += sprintf(ptr, "%x, ", i);
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
	char triplex[24];
	char buf[48];
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
		printf("def: %s ", triplex);
		int stat = hexchar2int(triplex[0]);
		int symb = triplex[1];
		for(char * x = triplex+2; *x; ++x) { 	/* 遷移先は複数記述可能 */
			mp->delta[stat][symb] |= 1<<hexchar2int(*x);
		}
		printf("%x, %c -> ", stat, symb);
		printf("%s\n", set16tostr(mp->delta[stat][symb], buf));
		ptr += strlen(triplex);
		if (*ptr == 0) break;
		++ptr; 	/* , を読み飛ばす */
		//printf("%s\n", ptr);
	}
	mp->initial = hexchar2int(*initial); 	/* 初期状態は１つ */
	for(ptr = finals; *ptr ; ++ptr) {
		mp->finals |= 1<<hexchar2int(*ptr);
	}
}

void nfa_reset(nfa * mp) {
	mp->current = 1<<mp->initial;
}

char nfa_transfer(nfa * mp, char a) {
	uint64_t next = 0;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if ((mp->current & (1<<i)) != 0) {
			next |= mp->delta[i][(int)a];
		}
	}
	return mp->current = next;
}

int nfa_accepting(nfa * mp) {
	return (mp->finals & mp->current) != 0;
}

void nfa_print(nfa * mp) {
	uint64_t states;
	char alphabet[ALPHABET_LIMIT];

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
	printf("states = {");
	int the1st = 1;
	for(int i = 0; i < STATE_LIMIT; ++i) {
		if ((states & (1<<i)) != 0) {
			if ( !the1st ) {
				printf(", ");
			}
			printf("%x", i);
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
	//printf("state symbol| next\n");
	//printf("------------+------\n");
	for(int i = 0; i < STATE_LIMIT; ++i) {
		for(int a = 0; a < ALPHABET_LIMIT; ++a) {
			if ( mp->delta[i][a] ) {
				printf("  %x  ,  %c   | ",i,a);
				printf("{");
				for(int b = 0; b < STATE_LIMIT; ++b) {
					if ( (mp->delta[i][a]>>b & 1) != 0 ) {
						printf("%x, ", b);
					}
				}
				printf("}\n");
			}
		}
	}
	printf("------------+------\n");
	printf("initial state = %x\n", mp->initial);
	/*
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
	*/
	printf("}\n)\n");
}

/*
int nfa_run(nfa * mp, char * inputstr) {
	char * ptr = inputstr;
	printf("dfa runs on '%s' :\n", ptr);
	nfa_reset(mp);
	printf("     -> %c", mp->current);
	for ( ; *ptr; ++ptr) {
		nfa_transfer(mp, *ptr);
		printf(", -%c-> %c", *ptr, mp->current);
	}
	if ( nfa_accepting(mp) ) {
		printf(", \naccepted.\n");
		return STATE_IS_FINAL;
	} else {
		printf(", \nrejected.\n");
		return STATE_IS_NOT_FINAL;
	}
}
*/

int main(int argc, char **argv) {
	nfa M;
	printf("M is using %lld bytes.\n", sizeof(M));
	nfa_define(&M, "0a1,0b2,1a2,1b0", "0", "0");
	nfa_print(&M);
	//nfa_run(&M, "ababab");
	printf("bye.\n");
	return EXIT_SUCCESS;
}

/*
 ============================================================================
 Name        : TestCode.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
/*
 * NSA @NSACareers
 * フォローする
 * "tpfccdlfdtte pcaccplircdt dklpcfrp?qeiq lhpqlipqeodf gpwafopwprti izxndkiqpkii krirrifcapnc dxkdciqcafmd vkfpcadf."
 * #MissionMonday #NSA #news
 */
// "tpfccdlfdttepcaccplircdtdklpcfrp?qeiqlhpqlipqeodfgpwafopwprtiizxndkiqpkiikrirrifcapncdxkdciqcafmdvkfpcadf."

#include <stdint.h>
typedef uint8_t boolean;
enum {
	false = 0,
	true = 0xff,
};

boolean lexconext(int map[], int n);
void asort(int array[], int s, int n );
void dsort(char array[], int s, int n );
void translate(char * str, int map[], char alp[]);

int main(int argc, char * argv[]) {
	char * ptr;
	int count[128];
	char alphabet[128] = "abcdefghijklmnopqrstuvwxyz";
	char talphabet[128], transed[256];
	int alsize;
	int map[128];
	char stop[128] = "";

	long counter;
	struct {
		char word[8];
		int weight;
	} keywords[8];
	int threshold = 0;
	int i, n;

	printf("Hi.\n");

	if ( argc == 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}
	ptr = argv[1];
	printf("Input string: \"%s\" of length %d.\n", ptr, (int)strlen(ptr));
	if ( argc >= 3 ) {
		if ( strlen(argv[2]) > 0 )
			strcpy(alphabet, argv[2]);
	}
	alsize = strlen(alphabet);
	if ( argc >= 4 )
		strcpy(stop, argv[3]);
	if ( strlen(stop) != alsize )
		stop[0] = 0;
	if ( argc >= 5 ) {
		printf("argc = %d\n", argc);
		n = 0;
		for (i = 4; i + 1 < argc; i += 2) {
			strcpy(keywords[n].word, argv[i]);
			keywords[n].weight = atoi(argv[i+1]);
			n++;
		}
		keywords[n].word[0] = 0;
		if ( argv[i] )
			threshold = atoi(argv[i]);

		printf("threshold = %d, ", threshold);
		for(i = 0; keywords[i].word[0]; i++) {
			printf("%s (%d), ", keywords[i].word, keywords[i].weight);
		}
		printf("\n");
	} else {
		keywords[0].word[0] = 0;
		keywords[0].weight = 0;
	}

	for(i = 0; i < 128; i++)
		count[i] = 0;
	for(i = 0; i < strlen(ptr); i++) {
		count[(int)ptr[i]]++;
	}
	printf("\nFrequency:\n");
	for(i = 0; i < 128; i++) {
		if ( count[i] != 0 ) {
			printf("%c: %d\n", (char)i, count[i]);
		}
	}
	printf("\n");


	for (i = 0; i < alsize; i++) {
		map[i] = i;
	}
	map[alsize] = 0;

	printf("alphabet = %s, size = %d\n", alphabet, alsize);

	for (counter = 0; ; counter++) {
		printf("%012ld: ", counter);
		strcpy(talphabet, alphabet);
		//printf("%s ", talphabet);
		translate(talphabet, map, alphabet);
		for(i = 0; i < alsize ; i++) {
			if ( isprint(talphabet[i]) ) {
				printf("%c", talphabet[i]);
			} else {
				printf(" ");
			}
		}

		strcpy(transed, ptr);
		translate(transed, map, alphabet);
		printf("; %s", transed);

		n = 0;
		for(i = 0; keywords[i].word[0]; i++) {
			if ( strstr(transed, keywords[i].word) )
				n += keywords[i].weight;
		}
		if ( n > threshold ) {
			printf(" @%d", n);
		}
		printf("\n");

		fflush(stdout);

		if ( !lexconext(map, alsize) )
			break;

		if ( strcmp(talphabet, stop) == 0 )
			break;
	}

	printf("\n");
	printf("Bye.\n\n");
	return 0;
}

boolean lexconext(int a[], int n) {
	int i, j, t;

	for(i = n - 1; i > 0; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( a[i-1] < a[i] ) {
			asort(a, i, n);
			for (j = i; a[j] < a[i-1]  ; j++);
			t = a[i-1];
			a[i-1] = a[j];
			a[j] = t;
			// DEBUG printf("i = %d\n", i);
			return true;
		}
	}
	return false;
}

void asort(int array[], int s, int n ) {
	int t;
	int i, j;
	for(i = s; i < n - 1; i++) {
		for(j = i + 1; j < n; j++)
			if ( array[i] > array[j] ) {
				t = array[i];
				array[i] = array[j];
				array[j] = t;
			}
		//printf("%d, ", array[i]);
	}
	//puts("\n");
}

void dsort(char array[], int s, int n ) {
	char t;
	int i, j;
	for(i = 0; i < n - 1; i++)
		for(j = i + 1; j < n; j++)
			if ( array[i] < array[j] ) {
				t = array[i];
				array[i] = array[j];
				array[j] = t;
			}
}

void translate(char * str, int map[], char alph[]) {
	int i, t;
	for(i = 0; i < strlen(str); i++) {
		for(t = 0; alph[t] && ((char)(str[i]) != alph[t]); t++);
		if ( alph[t] )
			str[i] = (char) alph[map[(int)t]];
	}
}

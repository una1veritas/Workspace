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

boolean next(char * a, int n);
void asort(char array[], int s, int n );
void dsort(char array[], int n );
void translate(char * str, char map[], char alphabet[], int asize);

int main(int argc, char * argv[]) {
	char * ptr;
	int count[128];
	char map[128], alphabet[128] = "abcdefghijklmnopqrstuvwxyz";
	int asize = strlen(alphabet);

	printf("Hi.\n");

	if ( argc == 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}
	ptr = argv[1];
	printf("Input string: \"%s\" of length %ld.\n", ptr, strlen(ptr));
	if ( argc == 3 ) {
		strcpy(alphabet, argv[2]);
		asize = strlen(alphabet);
		strcpy(map, alphabet);
	}

	int i, j;
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

	strcpy(map, alphabet);
	asort(map, 0, asize);
	printf("alphabet: %s\n", alphabet);

	char transed[256];
	long counter;
	for (counter = 0; ; counter++) {
		printf("%012ld: ", counter);
		for(i = 0; i < asize; i++) {
			if ( isprint(map[i]) ) {
				printf("%c", map[i]);
			} else {
				printf(" ");
			}
		}
		strcpy(transed, ptr);
		translate(transed, map, alphabet, asize);
		printf(": %s\n", transed);
		fflush(stdout);

		if ( !next(map, asize) )
			break;
	}

	printf("\n");
	printf("Bye.\n\n");
	return 0;
}

boolean next(char * a, int n) {
	int i, t;

	for(i = n-1; i > 0; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( a[i-1] < a[i] ) {
			t = a[i-1];
			a[i-1] = a[i];
			a[i] = t;
			// DEBUG printf("i = %d\n", i);
			asort(a, i, n);
			return true;
		}
	}
	return false;
}

void asort(char array[], int s, int n ) {
	char t;
	int i, j;
	for(i = s; i < n - 1; i++)
		for(j = i + 1; j < n; j++)
			if ( array[i] > array[j] ) {
				t = array[i];
				array[i] = array[j];
				array[j] = t;
			}
}

void dsort(char array[], int n ) {
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

void translate(char * str, char map[], char alphabet[], int asize) {
	int i, target;
	for(i = 0; i < strlen(str); i++) {
		for(target = 0; target < asize; target++)
			if ( (char)(str[i]) == alphabet[target] )
				break;
		if ( target < asize )
			str[i] = (char) map[(int)target];
	}
}

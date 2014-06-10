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
		str[i] = (char) map[(int)target];
	}
}

int main(int argc, char * argv[]) {
	char * ptr;
	int count[128];

	puts("Hi.\n");

	if ( argc == 1 )
		return 1;
	ptr = argv[1];
	printf("Input: %s\n", ptr);

	int i;
	for(i = 0; i < 128; i++)
		count[i] = 0;
	for(i = 0; i < strlen(ptr); i++) {
		count[(int)ptr[i]]++;
	}
	puts("\nFrequency:\n");
	for(i = 0; i < 128; i++) {
		if ( count[i] != 0 ) {
			printf("%c: %d\n", (char)i, count[i]);
		}
	}
	puts("\n\n");

	char map[128], alphabet[128] = "abcdefghijklmnopqrstuvwxyz.?";
	int asize = strlen(alphabet);

	asort(alphabet, 0, asize);
	memcpy(map, alphabet, asize+1);
	printf("alphabet: %s\n", alphabet);

	char transed[256];
	long counter;
	for (counter = 0; ; counter++) {
		printf("%012ld: ", counter);
		for(i = 0; i < asize; i++) {
			printf("%c", map[i]);
		}
		strcpy(transed, ptr);
		translate(transed, map, alphabet, asize);
		printf(": %s\n", transed);

		if ( !next(map, asize) )
			break;
	}

	printf("\n");
	puts("Bye.\n\n");
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

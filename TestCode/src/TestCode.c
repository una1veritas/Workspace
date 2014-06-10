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

int main(int argc, char * argv[]) {
	char * ptr;
	int count[128];

	puts("Hi.\n");

	if ( argc == 1 )
		return 1;
	ptr = argv[1];
	printf("Input: %s\n", ptr);

	int i, j, t;
	int index[128];

	for(i = 0; i < 128; i++)
		count[i] = 0;
	for(i = 0; i < strlen(ptr); i++) {
		count[(int)ptr[i]]++;
	}
	puts("\n\n");

	for(i = 0; i < 128; i++)
		index[i] = i;
	for(i = 0; i + 1 < 128; i++) {
		for(j = i + 1; j < 128; j++) {
			if ( count[index[i]] < count[index[j]] ) {
				t = index[i];
				index[i] = index[j];
				index[j] = t;
			}
		}
	}

	char map[128], alphabet[128] = ".?abcdefghijklmnopqrstuvwxyz";
	int asize = strlen(alphabet);
	memcpy(map, alphabet, asize+1);

	for(i = 0; alphabet[i]; i++) {
		printf("%c", alphabet[i]);
	}
	printf("\n");

	char transed[256];
	char target;
	long counter;
	for (counter = 0; ; counter++) {
		printf("%012ld: ", counter);
		for(i = 0; i < asize; i++) {
			printf("%c", map[i]);
		}
		for(i = 0; i <= strlen(ptr); i++) {
			for(target = 0; target < asize; target++)
				if ( ptr[i] == alphabet[(int)target] )
					break;
			transed[i] = (char) map[(int)target];
		}
		printf(": %s\n", transed);

		if ( !next(map, asize) )
			break;
	}

	printf("\n");
	puts("Bye.\n\n");
	return 0;
}

boolean next(char * a, int n) {
	int i, j, k, t;

	for(i = n-1; i > 0; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( a[i-1] < a[i] ) {
			t = a[i-1];
			a[i-1] = a[i];
			a[i] = t;
			// DEBUG printf("i = %d\n", i);
			for(j = i; j < n; j++) {
				for(k = j+1; k < n; k++) {
					if ( a[j] > a[k] ) {
						t = a[j];
						a[j] = a[k];
						a[k] = t;
						// DEBUG printf("j, k = %d, %d\n", j, k);
					}
				}
			}
			return true;
		}
	}
	return false;
}

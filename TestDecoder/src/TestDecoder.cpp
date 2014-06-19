/*
 ============================================================================
 Name        : TestCode.c
 Author      : Sin
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include <stdint.h>
}

#include "TestDecoder.h"

/*
 * NSA @NSACareers
 * "tpfccdlfdtte pcaccplircdt dklpcfrp?qeiq lhpqlipqeodf gpwafopwprti izxndkiqpkii krirrifcapnc dxkdciqcafmd vkfpcadf."
 * #MissionMonday #NSA #news
 */
// "tpfccdlfdttepcaccplircdtdklpcfrp?qeiqlhpqlipqeodfgpwafopwprtiizxndkiqpkiikrirrifcapncdxkdciqcafmdvkfpcadf."
// "cpfttolfocceptattplirtocoklptfrp?qeiqlhpqlipqedofgpwafdpwprciizxnokiqpkiikrirriftapntoxkotiqtafmovkfptaof."


int main(int argc, char * argv[]) {
	char * ptr;
	int count[128];
	char transed[256];
	long counter;
	uint i, t, lim;
	Mapping map;

	printf("Hi.\n");

	// setting the default values

	if ( argc == 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}
	ptr = argv[1];
	printf("Input string: \"%s\" of length %d.\n", ptr, (int)strlen(ptr));
	if ( argc >= 3 && strlen(argv[2]) > 0 ) {
		for(i = 0, t = 0; argv[2][i] && argv[2][i] != '/'; i++) {
			if (argv[2][i] == '~' ) {
				lim = t-1;
			} else {
				transed[t++] = argv[2][i];
			}
		}
		if ( argv[2][i] == '/' ) {

		}
		map.set(transed, lim);
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

	printf("alphabet = %s, size = %d\n", map.alphabet, map.size);

	for (counter = 0; ; counter++) {
		printf("%012ld: ", counter);
		strcpy(transed, map.alphabet);
		map.translate(transed);
		/*
		for(i = 0; i < map.size ; i++) {
			if ( isprint(transed[i]) ) {
				printf("%c%c ", map.alphabet[i],transed[i]);
			} else {
				printf(" ");
			}
		}
		*/
		printf("%s ", transed);

		strcpy(transed, ptr);
		map.translate(transed);
		printf("; %s", transed);
		printf("\n");

		fflush(stdout);

		if ( ! map.transfer.next() )
			break;
	}

	printf("\n");
	printf("Bye.\n\n");
	return 0;
}

void Mapping::set(const char a[], const uint lim) {
	uint i; //, t;
	for(i = 0; a[i] != 0 ; i++) {
		alphabet[i] = a[i];
	}
	alphabet[i] = 0;
	size = strlen(alphabet);
	transfer.init(size, lim);
}


boolean Permutation::next() {
	uint i, j, t;

	for(i = permsize - 1; i > 0 ; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( perm[i-1] == anchor )
			break;
		if ( perm[i-1] < perm[i] ) {
			asort(perm, i, permsize);
			for (j = i; perm[j] < perm[i-1]  ; j++);
			t = perm[i-1];
			perm[i-1] = perm[j];
			perm[j] = t;
			// DEBUG printf("i = %d\n", i);
			return true;
		}
	}
	return false;
}

void asort(uint array[], uint s, uint n ) {
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

void dsort(uint array[], uint s, uint n ) {
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

void Mapping::translate(char * str) const {
	uint i; //, t;
	for(i = 0; i < strlen(str); i++) {
		str[i] = (*this)[str[i]];
	}
}

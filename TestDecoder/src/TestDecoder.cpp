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
	uint i;
	Mapping map;

	printf("Hi.\n");

	// setting the default values

	if ( argc == 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}
	ptr = argv[1];
	printf("Input string: \"%s\" of length %d.\n", ptr, (int)strlen(ptr));
	if ( argc >= 3 ) {
		if ( strlen(argv[2]) > 0 )
			map.set(argv[2]);
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
		//printf("%s ", talphabet);
		map.translate(transed);
		for(i = 0; i < map.size ; i++) {
			if ( isprint(transed[i]) ) {
				printf("%c%c ", map.alphabet[i],transed[i]);
			} else {
				printf(" ");
			}
		}

		strcpy(transed, ptr);
		map.translate(transed);
		printf("; %s", transed);
		printf("\n");

		fflush(stdout);

		if ( ! map.next() )
			break;
	}

	printf("\n");
	printf("Bye.\n\n");
	return 0;
}

void Mapping::set(const char a[]) {
	strcpy(alphabet, a);
	size = strlen(alphabet);
	for (uint i = 0; i < size; i++) {
		transfer[i] = i;
	}
	transfer[size] = 0;
}


boolean Mapping::next() {
	uint i, j, t;

	for(i = size - 1; i > 0 ; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( alphabet[transfer[i-1]] == '~' )
			break;
		if ( transfer[i-1] < transfer[i] ) {
			asort(transfer, i, size);
			for (j = i; transfer[j] < transfer[i-1]  ; j++);
			t = transfer[i-1];
			transfer[i-1] = transfer[j];
			transfer[j] = t;
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

void Mapping::translate(char * str) const {
	uint i, t;
	for(i = 0; i < strlen(str); i++) {
		for(t = 0; alphabet[t] && ((char)(str[i]) != alphabet[t]); t++);
		if ( alphabet[t] )
			str[i] = alphabet[transfer[t]];
	}
}

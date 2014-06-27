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

#include <iostream>
#include "TestDecoder.h"

/*
 * NSA @NSACareers
 * "tpfccdlfdtte pcaccplircdt dklpcfrp?qeiq lhpqlipqeodf gpwafopwprti izxndkiqpkii krirrifcapnc dxkdciqcafmd vkfpcadf."
 * #MissionMonday #NSA #news
 */
// "tpfccdlfdttepcaccplircdtdklpcfrp?qeiqlhpqlipqeodfgpwafopwprtiizxndkiqpkiikrirrifcapncdxkdciqcafmdvkfpcadf."
// "tpfcdlkriaeqxnmhzvogw~/wantokrseihcplgbxumdy"
/*
 * 000000000000: wantokrseihcplgbxumdy ; want to know what it takes to work at nsa?
 * check back each monday in may as we explore careers essential to protecting our nation.
 *
 */

int main(int argc, char * argv[]) {
	char * text;
	char tmp[256];
	int freq[128];
	long counter;
	uint i, t;
	Mapping map;

	printf("Hi.\n");

	// setting the default values

	if ( argc <= 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}

	text = argv[1];
	printf("Input string: \"%s\" of length %d.\n", text, (int)strlen(text));

	char * ptr;
	char alphabet[128];
	uint start = 0;
	if ( argc >= 3 && (strlen(argv[2]) > 0) ) {
		ptr = argv[2];
		for(i = 0, t = 0; ptr[i] && ptr[i] != '/' && ptr[i] != Mapping::terminator; i++, t++) {
			alphabet[t] = ptr[i];
		}
		if ( ptr[i] == Mapping::terminator ) {
			start = i;
			i++;
		}
		for( ; ptr[i] && ptr[i] != '/'; i++, t++) {
			alphabet[t] = ptr[i];
		}
		alphabet[t] = 0;
		if ( ptr[i] == '/' ) {
			i++;
			for(t = 0 ; ptr[i] ; i++, t++) {
				tmp[t] = ptr[i];
			}
			tmp[t] = 0;
		}
		printf("alphabet = %s, target range = %s, starts from %d.\n", alphabet, tmp, start);
		map.init(alphabet, tmp, start);
	}

	for(i = 0; i < 128; i++)
		freq[i] = 0;
	for(i = 0; i < strlen(text); i++) {
		freq[(int)text[i]]++;
	}
	printf("\nFrequency:\n");
	for(i = 0; i < 128; i++) {
		if ( freq[i] != 0 ) {
			printf("%c: %d\n", (char)i, freq[i]);
		}
	}
	printf("\n");

	printf("alphabet = %s, size = %d\n", map.domain(), map.alphabetSize());

	for (counter = 0; ; counter++) {
		printf("%012ld: ", counter);
		strcpy(tmp, map.domain());
		map.translate(tmp);
		/*
		for(i = 0; i < map.size ; i++) {
			if ( isprint(transed[i]) ) {
				printf("%c%c ", map.alphabet[i],transed[i]);
			} else {
				printf(" ");
			}
		}
		*/
		printf("%s ", tmp);

		strcpy(tmp, text);
		map.translate(tmp);
		printf("; %s", tmp);
		printf("\n");

		fflush(stdout);

		if ( ! map.next() )
			break;
	}

	printf("\n");
	printf("Bye.\n\n");
	return 0;
}

void Mapping::init(const char a[], const char r[], const uint lim) {
	strcpy(charfrom, a);
	alphsize = strlen(charfrom);
	if ( alphsize != strlen(r) )
		strcpy(charto, charfrom);
	else {
		strcpy(charto, r);

	}
	for(int i = 0; i < 256; i++)
		charmap[i] = -1;
	for(int i = 0; i < alphsize; i++) {
		charmap[(uint)charfrom[i]] = i;
	}
	permlimit = lim;
	rotor.init(alphsize);
}


boolean Permutation::next(const uint startpos) {
	uint i, j, t;

	for(i = length - 1; i > startpos ; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( perm[i-1] < perm[i] ) {
			asort(perm, i, length);
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
	uint t;
	uint i, j;
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
	uint i, j;
	for(i = 0; i < n - 1; i++)
		for(j = i + 1; j < n; j++)
			if ( array[i] < array[j] ) {
				t = array[i];
				array[i] = array[j];
				array[j] = t;
			}
}
/*
char Mapping::operator[](const char c) const {
	uint i, t;
	for(t = 0; t < size ; t++)
		if ( c == alphabet[t] ) {
			i = transfer[t];
			return alphabet[i];
		}
	return c;
}
*/

boolean Mapping::next() {
	return rotor.next(permlimit);
}

void Mapping::translate(char * str) const {
	uint i; //, t;
	for(i = 0; i < strlen(str); i++) {
		if ( charmap[(uint)str[i]] != -1 )
			str[i] = charto[rotor[charmap[(uint)str[i]]]];
	}
}

/*
void Mapping::setTranslate(const char orig[], const char trans[]) {

}
*/

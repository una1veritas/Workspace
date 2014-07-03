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
/*"toatyxpsdlionofdwurnzvupvbymvswpyutwxiosd?"
 * 000000000000: wantokrseihcplgbxumdy ; want to know what it takes to work at nsa?
 * check back each monday in may as we explore careers essential to protecting our nation.
 *
 */

int main(int argc, char * argv[]) {
	char * text;
	char tmp[256];
	int freq[128];
	long counter;
	Mapping map;

	printf("Hi.\n");

	// setting the default values

	if ( argc <= 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}

	text = argv[1];
	printf("Input string: \"%s\" of length %d.\n", text, (int)strlen(text));

	{
		char * ptr;
		char original[128] = { 0 };
		char converted[128] = { 0 };
		char permutate[128] = { 0 };
		char permconv[128] = { 0 };
		uint startpos = 0;
		uint i, t;

		if ( argc > 2 ) {
			// const replacement
			ptr = argv[2];
			for(i = 0; ptr[i] != 0 && ptr[i] != '/' ; i++) {
				original[i] = ptr[i];
			}
			original[i] = 0;
			if ( ptr[i] == '/' ) {
				i++;
			}
			for(t=0 ; ptr[i] != 0 ; i++, t++) {
				converted[t] = ptr[i];
			}
			for( ; t < strlen(original); t++) {
				converted[t] = original[t];
			}
			converted[t] = 0;
			std::cout << "original alphabet: " << original << ", replace to: " << converted << std::endl;
		}
		if ( argc > 3 ) {
			ptr = argv[3];
			for(i = 0; ptr[i] != 0 && ptr[i] != '/' ; i++) {
				permutate[i] = ptr[i];
			}
			i++;
			for(t=0 ; ptr[i] != 0 ; i++, t++) {
				permconv[t] = ptr[i];
			}
			for( ; t < strlen(original); t++) {
				permconv[t] = permutate[t];
			}
			std::cout << "permutation alphabet: " << permutate << ", replace to: " << permconv << std::endl;
		}
		startpos = strlen(original);
		strcat(original, permutate);
		strcat(converted, permconv);
		printf("alphabet = %s, target range = %s, starts from %d.\n", original, converted, startpos);
		map.init(original, converted, startpos);
	}

	uint i;
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
	for(uint i = 0; i < alphsize; i++) {
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

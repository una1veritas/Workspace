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

#include <stdint.h>

/*
 * NSA @NSACareers
 * "tpfccdlfdtte pcaccplircdt dklpcfrp?qeiq lhpqlipqeodf gpwafopwprti izxndkiqpkii krirrifcapnc dxkdciqcafmd vkfpcadf."
 * #MissionMonday #NSA #news
 */
// "tpfccdlfdttepcaccplircdtdklpcfrp?qeiqlhpqlipqeodfgpwafopwprtiizxndkiqpkiikrirrifcapncdxkdciqcafmdvkfpcadf."
// "cpfttolfocceptattplirtocoklptfrp?qeiqlhpqlipqedofgpwafdpwprciizxnokiqpkiikrirriftapntoxkotiqtafmovkfptaof."
/*
.: 1
?: 1
a: 5
c: 12
d: 10
e: 3
f: 9
g: 1
h: 1
i: 11
k: 6
l: 5
m: 1
n: 2
o: 2
p: 13
q: 6
r: 6
t: 5
v: 1
w: 2
x: 2
z: 1
 */
typedef uint8_t boolean;
enum {
	false = 0,
	true = 0xff,
};

typedef struct {
	char alphabet[128];
	int size;
	int transf[128];
} mapping;

boolean lexconext(mapping * map);
void asort(int array[], int s, int n );
void dsort(char array[], int s, int n );
void translate(char * str, mapping * map);

int main(int argc, char * argv[]) {
	char * ptr;
	int count[128];
	char transed[256];
	long counter;
	int i, t;
	mapping map;

	printf("Hi.\n");

	// setting the default values
	strcpy(map.alphabet, "abcdefghijklmnopqrstuvwxyz");
	map.size = strlen(map.alphabet);
	for (i = 0; i < map.size; i++) {
		map.transf[i] = i;
	}
	map.transf[map.size] = 0;

	if ( argc == 1 ) {
		printf("Specify input(s) please.\n\n");
		return 1;
	}
	ptr = argv[1];
	printf("Input string: \"%s\" of length %d.\n", ptr, (int)strlen(ptr));
	if ( argc >= 3 ) {
		if ( strlen(argv[2]) > 0 )
			strcpy(map.alphabet, argv[2]);
		map.size = strlen(map.alphabet);
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
		translate(transed, &map);
		for(i = 0; i < map.size ; i++) {
			if ( isprint(transed[i]) ) {
				printf("%c%c ", map.alphabet[i],transed[i]);
			} else {
				printf(" ");
			}
		}

		strcpy(transed, ptr);
		translate(transed, &map);
		printf("; %s", transed);
		printf("\n");

		fflush(stdout);

		if ( !lexconext(&map) )
			break;
	}

	printf("\n");
	printf("Bye.\n\n");
	return 0;
}

boolean lexconext(mapping * m) {
	int i, j, t;

	for(i = m->size - 1; i > 0; i--) {
		// DEBUG printf("a[i-1], a[i] = %d, %d\n", a[i-1], a[i]);
		if ( m->transf[i-1] < m->transf[i] ) {
			asort(m->transf, i, m->size);
			for (j = i; m->transf[j] < m->transf[i-1]  ; j++);
			t = m->transf[i-1];
			m->transf[i-1] = m->transf[j];
			m->transf[j] = t;
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

void translate(char * str, mapping * map) {
	int i, t;
	for(i = 0; i < strlen(str); i++) {
		for(t = 0; map->alphabet[t] && ((char)(str[i]) != map->alphabet[t]); t++);
		if ( map->alphabet[t] )
			str[i] = map->alphabet[(int)map->transf[(int)t]];
	}
}

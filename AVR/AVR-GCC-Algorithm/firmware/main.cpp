#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>

// Necessary reset/interrrupt vectors (jump tables) 
// will be automatically generated and linked with 
// your program codes. 
// Initialization routine of the stack pointer 
// register (SPL, SPH) will be provided and placed
// just after the reset event and just before your code.

#include "Wiring.h"
// typedef unsigned char byte;
// typedef unsigned int word;

#include "String.h"

void InsertionSort(char array[], int size);
void SelectionSort(char array[], int size);

int main() {
	char a[16] = {'i', 'b', '$', '&', 'A', 'b', 'r', 'Q', 0};
	InsertionSort(a,8);
	for (;;) {
	}
}

void InsertionSort(char array[], int size) {
	int index, endex;
	char tmp;
	
	for (endex = 1; endex < size; endex++) {
		tmp = array[endex];
		for ( index = endex; index >= 0; index--) {
			if ( array[index-1] < tmp )
				break;
			array[index] = array[index-1];
		}
		array[index] = tmp;
	}
}

void SelectionSort(char array[], int size) {
	int index, endex;
	char tmp;
	
	for (endex = 0; endex < size-1; endex++) {
		for (index = endex; index < size; index++) {
			tmp = array[index];
			if ( array[endex] > tmp ) {
				array[index] = array[endex];
				array[endex] = tmp;
			}
		}
	}
}

long gcd(long a, long b) {
	long t;
	while (b > 0) {
		while (a >= b) {
			a -= b;
		}
		t = a;
		a = b;
		b = t;
	}
	return a;
}

long gcd_mod(long a, long b) {
	long t;
	while (b > 0) {
		t = a % b;
		a = b;
		b = t;
	}
	return a;
}

